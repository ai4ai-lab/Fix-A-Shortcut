#!/usr/bin/env python
# mimic-cxr.py — multi-method MIMIC-CXR pipeline         (updated 2025-08-07)
# ---------------------------------------------------------------------------
# • Heavy steps (biasing, pre-training, embedding extraction) are cached.
# • Add --fresh to ignore caches.
# • NEW: --balance_labels   → matches the update in data.py so you can easily
#   create datasets whose class prior P(y=1)=0.5 before injecting the shortcut.
# ---------------------------------------------------------------------------

import argparse
import pathlib
import random
import time
import warnings
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import get_single_env_loaders
from model import ConceptEncoder, UnknownEncoder
from train import Trainer

# ────────── helpers ──────────────────────────────────────────────────


def tic(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}")


def safe_torch_load(path):
    try:
        return torch.load(path, map_location="cpu")
    except Exception:
        return torch.load(path, map_location="cpu", weights_only=False)


@torch.no_grad()
def extract_embeddings(enc_c, enc_u, loader, device):
    enc_c.eval()
    enc_u.eval()
    zc, zu, ss, ys = [], [], [], []
    for b in loader:
        if b is None:
            continue
        x = b["x"].to(device, non_blocking=True)
        y = b["y"].float().to(device)
        sex = b["gender"].float().to(device)
        age = b["age"].unsqueeze(1).to(device)
        view = b["view"].unsqueeze(1).to(device)
        u = torch.cat([age, view], dim=1)

        zc.append(enc_c(x).cpu())
        zu.append(enc_u(u).cpu())
        ss.append(sex.cpu().unsqueeze(1))
        ys.append(y.cpu())

    return (torch.cat(zc).float(),
            torch.cat(zu).float(),
            torch.cat(ss).float(),
            torch.cat(ys).float())


def make_ssl_labels(y, pct, seed):
    rng = np.random.RandomState(seed)
    mask = torch.zeros(len(y), dtype=torch.bool)
    mask[rng.choice(len(y), int((1-pct)*len(y)), replace=False)] = True
    y_ssl = y.clone()
    y_ssl[~mask] = -1.0
    return mask, y_ssl


def meta_eq(a, b): return a and b and (
    set(a) == set(b)) and all(a[k] == b[k] for k in a)


# ────────── argument parsing ─────────────────────────────────────────
p = argparse.ArgumentParser()
# dataset / bias knobs
p.add_argument("--root_dir", required=True)
p.add_argument("--patients_csv", required=True)
p.add_argument("--task", default="Edema")
p.add_argument("--source_task", default="Cardiomegaly")
p.add_argument("--p_male_pos_train", type=float, default=0.80)
p.add_argument("--p_male_neg_train", type=float, default=0.50)
p.add_argument("--p_male_pos_test",  type=float, default=0.20)
p.add_argument("--p_male_neg_test",  type=float, default=0.50)
p.add_argument("--balance_labels", action="store_true",
               help="Force P(y=1)=0.5 before shortcut biasing (from new data.py).")
p.add_argument("--limit", type=int)
p.add_argument("--verify", action="store_true")
p.add_argument("--data_seed", type=int, default=123)

# training knobs
p.add_argument("--unlabeled_pct", type=float, default=0.90)
p.add_argument("--lambda_list", default="2.0")
p.add_argument("--pretrain_epochs", type=int, default=1)
p.add_argument("--epochs", type=int, default=5)
p.add_argument("--head_lr", type=float, default=1e-2)
p.add_argument("--grad_methods",
               default="naive",
               help="Comma list: naive,fix_a_step,fix_a_step_soft,gradient_surgery")

# seeds
p.add_argument("--seeds", type=int, default=1)
p.add_argument("--start_seed", type=int, default=42)

# caching / output
p.add_argument("--out_dir", default="runs/exp")
p.add_argument("--reuse_cache", action="store_true")
p.add_argument("--fresh", action="store_true")

# loader / device
p.add_argument("--batch_size", type=int, default=16)
p.add_argument("--num_workers", type=int, default=4)

# advanced → forwarded to Trainer
p.add_argument("--shortcut_l2", type=float, default=0.0)
p.add_argument("--gs_scale_aligned", type=float, default=1.0)
p.add_argument("--scheduler_per_step", action="store_true")
p.add_argument("--te_on", choices=["test", "train"], default="test")
p.add_argument("--no_analytic_te", action="store_true")
args = p.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
out_root = pathlib.Path(args.out_dir)
out_root.mkdir(parents=True, exist_ok=True)
lambdas = [float(x) for x in args.lambda_list.split(",")]
methods = [m.strip() for m in args.grad_methods.split(",") if m.strip()]

# cache paths
cache_dir = out_root / "cache"
cache_dir.mkdir(exist_ok=True, parents=True)
emb_path = cache_dir / "embeddings.pt"
concept_p = cache_dir / "concept_encoder.pth"

meta = dict(task=args.task, source_task=args.source_task,
            p_male_pos_train=args.p_male_pos_train,
            p_male_neg_train=args.p_male_neg_train,
            p_male_pos_test=args.p_male_pos_test,
            p_male_neg_test=args.p_male_neg_test,
            balance_labels=args.balance_labels,
            limit=args.limit, data_seed=args.data_seed)

# ────────── 1) build / load embeddings ───────────────────────────────
need_embed, need_encoder = True, True

if (not args.fresh) and args.reuse_cache and emb_path.exists():
    d = safe_torch_load(emb_path)
    if meta_eq(d.get("meta"), meta):
        need_embed = False
        X_tr_full, y_tr = d["X_tr_full"].float(), d["y_tr"].float()
        X_te_full, y_te = d["X_te_full"].float(), d["y_te"].float()
        shortcut_idx = int(d["shortcut_idx"])
        corr_train = float(d["corr_train"])
        print("[cache] embeddings reused.")

if (not args.fresh) and args.reuse_cache and concept_p.exists():
    ckpt = safe_torch_load(concept_p)
    if meta_eq(ckpt.get("meta"), meta):
        need_encoder = False
        concept_enc = ConceptEncoder(pretrained=True, freeze=False).to(device)
        concept_enc.load_state_dict(ckpt["state_dict"])
        print("[cache] concept encoder reused.")

if need_embed:
    tic("Building biased loaders …")
    tr_loader, _, te_loader = get_single_env_loaders(
        root_dir=args.root_dir, patients_csv=args.patients_csv,
        task=args.task,
        p_male_pos_train=args.p_male_pos_train, p_male_neg_train=args.p_male_neg_train,
        p_male_pos_test=args.p_male_pos_test,   p_male_neg_test=args.p_male_neg_test,
        batch_size=args.batch_size, num_workers=args.num_workers,
        seed=args.data_seed, limit=args.limit, verify=args.verify,
        balance_lbl=args.balance_labels
    )
    ys, ss = [], []
    for b in tr_loader:
        if b is None:
            continue
        ys.append(b["y"].numpy())
        ss.append(b["gender"].numpy())
    corr_train = float(np.corrcoef(
        np.concatenate(ys), np.concatenate(ss))[0, 1])

    if need_encoder:
        tic("Pre-training concept encoder (unbiased)…")
        src_loader, _, _ = get_single_env_loaders(
            root_dir=args.root_dir, patients_csv=args.patients_csv,
            task=args.source_task,
            p_male_pos_train=0.5, p_male_neg_train=0.5,
            p_male_pos_test=0.5, p_male_neg_test=0.5,
            batch_size=args.batch_size, num_workers=args.num_workers,
            seed=args.data_seed+999, limit=args.limit, verify=False,
            balance_lbl=False
        )
        concept_enc = ConceptEncoder(pretrained=True, freeze=False).to(device)
        probe = nn.Linear(concept_enc.out_dim, 1).to(device)
        opt = torch.optim.Adam(
            list(concept_enc.parameters())+list(probe.parameters()), lr=1e-4)
        loss_fn = nn.BCEWithLogitsLoss()
        for ep in range(args.pretrain_epochs):
            tot, n = 0, 0
            for b in src_loader:
                if b is None:
                    continue
                x = b["x"].to(device)
                y = b["y"].float().to(device)
                opt.zero_grad()
                loss = loss_fn(probe(concept_enc(x)).view(-1), y.view(-1))
                loss.backward()
                opt.step()
                tot += loss.item()*x.size(0)
                n += x.size(0)
            print(
                f"[pretrain] epoch {ep+1}/{args.pretrain_epochs} loss={tot/n:.4f}")
        torch.save({"state_dict": concept_enc.state_dict(),
                   "meta": meta}, concept_p)

    for p in concept_enc.parameters():
        p.requires_grad_(False)
    concept_enc.eval()
    unknown_enc = UnknownEncoder(2, 64, 64).to(device).eval()

    tic("Extracting embeddings …")
    zc_tr, zu_tr, s_tr, y_tr = extract_embeddings(
        concept_enc, unknown_enc, tr_loader, device)
    zc_te, zu_te, s_te, y_te = extract_embeddings(
        concept_enc, unknown_enc, te_loader, device)

    X_tr_full = torch.cat([zc_tr, zu_tr, s_tr], 1)
    X_te_full = torch.cat([zc_te, zu_te, s_te], 1)
    shortcut_idx = zc_tr.shape[1]+zu_tr.shape[1]

    torch.save({"X_tr_full": X_tr_full, "y_tr": y_tr,
                "X_te_full": X_te_full, "y_te": y_te,
                "shortcut_idx": shortcut_idx, "corr_train": corr_train,
                "meta": meta}, emb_path)
    print(f"[saved] {emb_path}")

# ────────── 2) train linear heads ────────────────────────────────────
corr_tag = f"corr{corr_train:.3f}".replace(".", "p")
for lam in lambdas:
    lam_tag = f"lam{lam}".replace(".", "p")
    base_dir = out_root / f"{corr_tag}_{lam_tag}"
    base_dir.mkdir(exist_ok=True)
    for method in methods:
        mdir = base_dir/method
        mdir.mkdir(exist_ok=True)
        for k in range(args.seeds):
            seed = args.start_seed+k
            print(f"\n===== λ={lam} | {method} | seed={seed} =====")
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

            mask_lab, y_ssl = make_ssl_labels(y_tr, args.unlabeled_pct, seed)
            head = nn.Linear(X_tr_full.shape[1], 1).to(device)
            trainer = Trainer(
                model=head, device=device,
                train_data=X_tr_full, train_label=y_ssl,
                test_data=X_te_full,  test_label=y_te,
                train_loss_fn=nn.BCEWithLogitsLoss(),
                test_loss_fn=nn.BCEWithLogitsLoss(),
                unlab_data=X_tr_full[~mask_lab],
                lambda_unlab=lam, unlab_threshold=0.95,
                optimizer_fn=torch.optim.Adam,
                optimizer_params={"lr": args.head_lr},
                scheduler_fn=torch.optim.lr_scheduler.StepLR,
                scheduler_params={"step_size": 10_000, "gamma": 0.99},
                scheduler_per_step=args.scheduler_per_step,
                num_epochs=args.epochs, batch_size=64,
                grad_method=method,
                shortcut_l2=args.shortcut_l2,
                gs_scale_aligned=args.gs_scale_aligned,
                te_on=args.te_on,
                analytic_te_for_linear=not args.no_analytic_te
            )
            trainer.train(classification=True,
                          estimate_treatment_effect=True,
                          shortcut_variable_idx=shortcut_idx)

            out_p = mdir/f"model_{corr_tag}_{lam_tag}_seed{seed}.pt"
            torch.save({"state_dict": head.state_dict(),
                        "logs": {"train_loss": trainer.train_loss,
                                 "test_loss": trainer.test_loss,
                                 "train_auc": trainer.train_auc,
                                 "test_auc": trainer.test_auc,
                                 "shortcut_weight_history": trainer.shortcut_weight_history,
                                 "treatment_effect": trainer.test_treatment_effect},
                        "shortcut_idx": shortcut_idx,
                        "corr_train": corr_train,
                        "lambda_unsup": lam,
                        "seed": seed,
                        "grad_method": method}, out_p)
            print("[saved]", out_p)
