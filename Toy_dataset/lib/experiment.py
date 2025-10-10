#!/usr/bin/env python
# experiment.py  –
# -----------------------------------------------------

import torch
import numpy as np
from typing import Dict, Any
from data    import generate_synthetic_base, generate_training_test_data
from model   import toy_nn
from gradients import (
    train_naive_5d, train_fix_5d, train_gs_5d,
    train_fix_soft_5d, train_gt_5d
)
import copy


def synthetic_experiments(
    # ───── data-gen params ─────
    num_samples: int = 2000,
    rho_cu: float = 0.0,
    sigma_y: float = 0.1,
    shortcut_mode: str = "y_noise",
    beta: float = 1.0,
    beta_y: float | None = None,
    delta_c: float | None = None,
    delta_u: float | None = None,
    sigma_s: float = 0.1,
    shortcut_ratio: float = 1.0,
    test_size: float = 0.5,
    random_state: int = 0,
    # ───── training params ─────
    n_labeled: int = 50,
    epochs: int = 50,
    lr: float = 1e-3,
    lambda_u: float = 10.0,
    device: str = "cpu",
    to_print: bool = True,
) -> Dict[str, Any]:

    # 1) synthetic data ------------------------------------------------
    X_base, y_all, y_clean = generate_synthetic_base(
        num_samples=num_samples, seed=random_state,
        rho_cu=rho_cu, sigma_y=sigma_y
    )
    x_tr_np, y_tr_np, x_te_np, y_te_np = generate_training_test_data(
        X_base, y_all, y_clean,
        shortcut_mode=shortcut_mode, beta=beta, beta_y=beta_y,
        delta_c=delta_c, delta_u=delta_u,
        sigma_s=sigma_s, shortcut_ratio=shortcut_ratio,
        test_size=test_size, random_state=random_state
    )

    # 2) torch tensors -------------------------------------------------
    dev     = torch.device(device)
    x_tr    = torch.tensor(x_tr_np, dtype=torch.float32, device=dev)
    y_tr    = torch.tensor(y_tr_np, dtype=torch.float32, device=dev)
    x_te    = torch.tensor(x_te_np, dtype=torch.float32, device=dev)
    y_te    = torch.tensor(y_te_np, dtype=torch.float32, device=dev)

    perm    = torch.randperm(x_tr.size(0), device=dev)
    x_l, y_l = x_tr[perm[:n_labeled]], y_tr[perm[:n_labeled]]
    x_u      = x_tr[perm[n_labeled:]]

    # 3) training routines --------------------------------------------
    ssl_trainers = dict(
        naive        = train_naive_5d,
        fix_step     = train_fix_5d,
        grad_surgery = train_gs_5d,
        fix_soft     = train_fix_soft_5d,
    )

    trajectories, final_w, models, stats = {}, {}, {}, {}

    # 3a) SSL methods --------------------------------------------------
    for name, trainer in ssl_trainers.items():
        torch.manual_seed(1)
        mdl = toy_nn(x_tr.size(1), 1).to(dev)

        traj = trainer(mdl, x_l, y_l, x_u,
                       epochs=epochs, lr=lr, lambda_u=lambda_u)

        trajectories[name]  = traj
        final_w[name]       = traj[-1]
        models[name]        = mdl
        stats[name]         = traj.stats

        if to_print:
            print(f"[{name:<11s}]  conflicts={stats[name]['conflict_count']:3d}  "
                  f"‖w_fin‖={np.linalg.norm(traj[-1]):.4f}")

    # 3b) GT baseline --------------------------------------------------
    torch.manual_seed(1)
    mdl_gt = toy_nn(x_tr.size(1), 1).to(dev)
    traj_gt = train_gt_5d(mdl_gt, x_tr, y_tr, epochs=epochs, lr=lr)

    trajectories["gt"]  = traj_gt
    final_w["gt"]       = traj_gt[-1]
    models["gt"]        = mdl_gt
    stats["gt"]         = traj_gt.stats

    if to_print:
        print(f"[gt          ]  ‖w_fin‖={np.linalg.norm(traj_gt[-1]):.4f}")

    # 4) bundle --------------------------------------------------------
    return dict(
        trajectories  = trajectories,
        final_weights = final_w,
        models        = models,
        stats         = stats,
        x_test        = x_te_np,
        y_test        = y_te_np,
    )
