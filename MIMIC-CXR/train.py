#!/usr/bin/env python
# train.py  — robust trainer 

import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader, Sampler
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


# ------------------------------ Sampler ------------------------------

class LabeledBatchSampler(Sampler):
    """
    Ensures each batch contains at least one labeled example (label != -1).
    """

    def __init__(self, data_source, labels, batch_size, drop_last=False):
        self.data_source = data_source
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.drop_last = drop_last

        self.labeled_idx = np.where(self.labels != -1)[0]
        self.all_idx = np.arange(len(self.data_source))

    def __iter__(self):
        indices = self.all_idx.copy()
        np.random.shuffle(indices)

        batches = []
        for i in range(0, len(indices), self.batch_size):
            batch = list(indices[i:i+self.batch_size])
            if len(batch) < self.batch_size and self.drop_last:
                continue
            if (len(self.labeled_idx) > 0) and (not any(idx in self.labeled_idx for idx in batch)):
                replace_position = np.random.randint(len(batch))
                labeled_sample = np.random.choice(self.labeled_idx)
                batch[replace_position] = labeled_sample
            batches.append(batch)

        for batch in batches:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.data_source) // self.batch_size
        return int(np.ceil(len(self.data_source) / self.batch_size))


# ------------------------------ Utilities ------------------------------

def zero_grads(model):
    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()


def get_grads(model):
    return [p.grad.clone() if p.grad is not None else None for p in model.parameters()]


def set_grads(model, grad_list):
    for p, g in zip(model.parameters(), grad_list):
        if p is not None and g is not None:
            if p.grad is None:
                p.grad = torch.zeros_like(p)
            p.grad.copy_(g)
        elif p is not None and p.grad is not None:
            p.grad.zero_()


# ------------------------------ TE / Dosage ------------------------------

def dosage(data, model, treatment_idx, lo=0.0, hi=1.0, n=2, to_prob=True):
    """
    Average sensitivity of model output to the treatment dimension by sweeping s in [lo, hi].
    For binary s, use lo=0, hi=1, n=2 (default). Returns mean |f(x,s_hi) - f(x,s_lo)|.
    """
    device = next(model.parameters()).device
    x0 = data.to(device).clone().detach()

    with torch.no_grad():
        vals = torch.linspace(lo, hi, n, device=device)
        preds = []
        for v in vals:
            x = x0.clone()
            x[:, treatment_idx] = v
            out = model(x).squeeze()
            preds.append(torch.sigmoid(out) if to_prob else out)
        P = torch.stack(preds, dim=1)  # [N, n]
        te = (P.max(dim=1).values - P.min(dim=1).values).mean().item()
    return te


def get_linear_weight_for_index(model: nn.Module, idx: int) -> torch.Tensor:
    """
    Returns a view of the scalar weight corresponding to feature index `idx`.
    Works for a simple Linear head (preferred). If no Linear found, falls back to
    the first parameter vector flattened.
    """
    for m in model.modules():
        if isinstance(m, nn.Linear):
            return m.weight.view(-1)[idx]
    for p in model.parameters():
        return p.view(-1)[idx]
    raise RuntimeError("Could not locate any parameters in the model.")


# ------------------------------ Loss for unlabeled ------------------------------

def unlab_loss_with_threshold(logits, threshold=0.95):
    """
    Classic self-training: only confident predictions contribute to the unlabeled loss.
    """
    probs = torch.sigmoid(logits)
    mask = (probs > threshold) | (probs < (1.0 - threshold))
    if mask.sum() == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)
    pseudo = (probs >= 0.5).float()
    return nn.functional.binary_cross_entropy_with_logits(logits[mask], pseudo[mask])


# ------------------------------ Dataset ------------------------------

class MyDataset(Dataset):
    def __init__(self, data, label):
        super().__init__()
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]


# ------------------------------ Trainer ------------------------------

class Trainer:
    def __init__(self,
                 model,
                 device,
                 # labeled data
                 train_data, train_label,
                 # test data
                 test_data, test_label,
                 # main losses
                 train_loss_fn,
                 test_loss_fn,
                 # optional unlabeled data
                 unlab_data=None,
                 # weighting for unlabeled loss
                 lambda_unlab=0.1,
                 # confidence threshold for unlabeled pseudo-label
                 unlab_threshold=0.95,
                 # optimizer
                 optimizer_fn=optim.Adam,
                 optimizer_params={"lr": 0.01, "weight_decay": 0.0},
                 # lr scheduler
                 scheduler_fn=optim.lr_scheduler.StepLR,
                 scheduler_params={"step_size": 10000, "gamma": 0.99},
                 scheduler_per_step=False,
                 # training control
                 num_epochs=10,
                 save_every=10,
                 batch_size=16,
                 drop_last=False,
                 grad_method=None,
                 # TE/shortcut options
                 treatment_range=(0.0, 1.0),
                 te_on="test",  # "test" or "train"
                 analytic_te_for_linear=True,
                 shortcut_l2=0.0,  # targeted L2 penalty on the shortcut weight
                 gs_scale_aligned=1.0  # scale factor for g_unlab when grads are aligned in PCGrad
                 ):
        """
        gs_scale_aligned < 1.0 makes gradient_surgery less aggressive when labeled/unlabeled grads align.
        shortcut_l2 > 0.0 penalizes |w_s|^2 directly (in addition to global weight_decay).
        """
        self.model = model.to(device)
        self.device = device

        self.train_data = train_data
        self.train_label = train_label
        self.test_data = test_data
        self.test_label = test_label

        self.unlab_data = unlab_data
        self.lambda_unlab = lambda_unlab
        self.unlab_threshold = unlab_threshold

        self.train_loss_fn = train_loss_fn
        self.test_loss_fn = test_loss_fn

        self.optimizer_fn = optimizer_fn
        self.optimizer_params = optimizer_params
        self.scheduler_fn = scheduler_fn
        self.scheduler_params = scheduler_params
        self.scheduler_per_step = scheduler_per_step

        self.num_epochs = num_epochs
        self.save_every = save_every

        # gradient method
        self.grad_method = (grad_method.lower().strip()
                            if grad_method else os.environ.get("GRADIENT_METHOD", "naive").lower().strip())

        # dataloaders
        self.train_ds = MyDataset(train_data, train_label)
        self.train_dl = DataLoader(
            self.train_ds,
            batch_sampler=LabeledBatchSampler(
                self.train_ds, train_label, batch_size, drop_last=drop_last)
        )
        self.test_ds = MyDataset(test_data, test_label)
        self.test_dl = DataLoader(self.test_ds, batch_size=64, shuffle=False)
        self.unlab_dl = None
        if self.unlab_data is not None:
            unl_labels = torch.full(
                (len(self.unlab_data),), -1.0, dtype=torch.float32)
            self.unlab_ds = MyDataset(self.unlab_data, unl_labels)
            self.unlab_dl = DataLoader(
                self.unlab_ds, batch_size=64, shuffle=True)

        # logging
        self.train_loss = []
        self.test_loss = []
        self.train_auc = []
        self.test_auc = []
        self.test_treatment_effect = []
        self.shortcut_weight_history = []
        self.grad_cosines = []
        self.grad_angles = []

        # TE / shortcut settings
        self.treatment_range = treatment_range
        self.te_on = te_on
        self.analytic_te_for_linear = analytic_te_for_linear
        self.shortcut_l2 = float(shortcut_l2)
        self.gs_scale_aligned = float(gs_scale_aligned)

        # Will be set inside train()
        self._shortcut_idx = None

    # ------------- internal helpers -------------

    def _shortcut_weight_tensor(self, idx):
        return get_linear_weight_for_index(self.model, idx)

    def _compute_te(self):
        """
        Compute treatment effect either analytically (for linear head) or via dosage sweep.
        """
        assert self._shortcut_idx is not None, "shortcut index not set"
        if self.analytic_te_for_linear:
            with torch.no_grad():
                ws = self._shortcut_weight_tensor(self._shortcut_idx)
                return abs(ws.item())
        else:
            data_ref = self.test_data if self.te_on == "test" else self.train_data
            lo, hi = self.treatment_range
            return dosage(data_ref, self.model, self._shortcut_idx, lo=lo, hi=hi, n=2, to_prob=True)

    # ------------- main training loop -------------

    def train(self,
              classification=True,
              to_print=True,
              estimate_treatment_effect=False,
              shortcut_variable_idx=-1):

        self._shortcut_idx = shortcut_variable_idx

        optimizer = self.optimizer_fn(
            self.model.parameters(), **self.optimizer_params)
        scheduler = self.scheduler_fn(optimizer, **self.scheduler_params)

        with tqdm(total=self.num_epochs, desc="Epoch", unit="epoch") as pbar:
            for epoch in range(self.num_epochs):
                self.model.train()

                labeled_iter = iter(self.train_dl)
                unl_iter = iter(
                    self.unlab_dl) if self.unlab_dl is not None else None
                steps = max(len(self.train_dl), len(
                    self.unlab_dl) if self.unlab_dl else 0)

                for _ in range(steps):
                    # scalar placeholders that require grad
                    loss_label = torch.tensor(
                        0.0, device=self.device, requires_grad=True)
                    loss_unlab = torch.tensor(
                        0.0, device=self.device, requires_grad=True)

                    # labeled batch
                    try:
                        x_lab, y_lab = next(labeled_iter)
                        x_lab = x_lab.to(self.device)
                        y_lab = y_lab.to(self.device)
                    except StopIteration:
                        x_lab, y_lab = None, None

                    # unlabeled batch
                    x_unl = None
                    if unl_iter is not None:
                        try:
                            x_unl, _ = next(unl_iter)
                            x_unl = x_unl.to(self.device)
                        except StopIteration:
                            x_unl = None

                    # forward passes
                    if x_lab is not None:
                        logits_lab = self.model(x_lab).squeeze()
                        mask = (y_lab >= 0)
                        if mask.sum() > 0:
                            loss_label = self.train_loss_fn(
                                logits_lab[mask], y_lab[mask])

                    if x_unl is not None:
                        logits_unl = self.model(x_unl).squeeze()
                        loss_unlab = unlab_loss_with_threshold(
                            logits_unl, threshold=self.unlab_threshold)

                    # targeted regularization on the shortcut weight (optional)
                    shortcut_reg = torch.tensor(0.0, device=self.device)
                    if self.shortcut_l2 > 0.0 and self._shortcut_idx is not None and self._shortcut_idx >= 0:
                        ws = self._shortcut_weight_tensor(self._shortcut_idx)
                        shortcut_reg = self.shortcut_l2 * (ws ** 2)

                    if self.grad_method == "naive":
                        optimizer.zero_grad()
                        total_loss = loss_label + self.lambda_unlab * loss_unlab + shortcut_reg
                        total_loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), max_norm=1.0)
                        optimizer.step()
                        if self.scheduler_per_step:
                            scheduler.step()

                    else:
                        # Two-pass accumulation to access each task's grads separately
                        optimizer.zero_grad()

                        # supervised + shortcut reg in the "labeled" pass
                        if x_lab is not None:
                            (loss_label + shortcut_reg).backward(retain_graph=True)
                            g_label = get_grads(self.model)
                        else:
                            g_label = [None] * \
                                len(list(self.model.parameters()))

                        # unlabeled pass
                        zero_grads(self.model)
                        if x_unl is not None:
                            (self.lambda_unlab *
                             loss_unlab).backward(retain_graph=True)
                            g_unlab = get_grads(self.model)
                        else:
                            g_unlab = [None] * \
                                len(list(self.model.parameters()))

                        # log gradient alignment
                        flat_lab_parts = [g.view(-1)
                                          for g in g_label if g is not None]
                        flat_unl_parts = [g.view(-1)
                                          for g in g_unlab if g is not None]
                        if flat_lab_parts and flat_unl_parts:
                            flat_lab = torch.cat(flat_lab_parts)
                            flat_unl = torch.cat(flat_unl_parts)
                            denom = flat_lab.norm() * flat_unl.norm() + 1e-12
                            cos_sim = (flat_lab @ flat_unl) / denom
                            angle = torch.acos(torch.clamp(cos_sim, -1.0, 1.0))
                        else:
                            cos_sim = torch.tensor(0.0, device=self.device)
                            angle = torch.tensor(0.0, device=self.device)
                        self.grad_cosines.append(cos_sim.item())
                        self.grad_angles.append(angle.item())

                        # Combine grads per method
                        if self.grad_method == "fix_a_step":
                            dot_t = torch.zeros(1, device=self.device)
                            for gl, gu in zip(g_label, g_unlab):
                                if gl is not None and gu is not None:
                                    dot_t += (gl * gu).sum()
                            if dot_t >= 0:
                                final_grads = [(gl + gu) if (gl is not None and gu is not None) else gl
                                               for gl, gu in zip(g_label, g_unlab)]
                            else:
                                final_grads = g_label

                        elif self.grad_method == "fix_a_step_soft":
                            # global cosine
                            dot_tot = torch.zeros(1, device=self.device)
                            normL_sq = torch.zeros(1, device=self.device)
                            normU_sq = torch.zeros(1, device=self.device)
                            for gl, gu in zip(g_label, g_unlab):
                                if gl is not None and gu is not None:
                                    dot_tot += (gl * gu).sum()
                                    normL_sq += (gl * gl).sum()
                                    normU_sq += (gu * gu).sum()
                            denom = torch.sqrt(normL_sq) * \
                                torch.sqrt(normU_sq) + 1e-12
                            cos_g = dot_tot / denom
                            alpha = torch.clamp(cos_g, min=0.0).item()

                            final_grads = []
                            for gl, gu in zip(g_label, g_unlab):
                                if gl is None and gu is None:
                                    final_grads.append(None)
                                elif gl is None:
                                    final_grads.append(alpha * gu)
                                elif gu is None:
                                    final_grads.append(gl)
                                else:
                                    final_grads.append(gl + alpha * gu)

                        elif self.grad_method == "gradient_surgery":
                            # Enhanced gradient surgery with dynamic scaling and range validation
                            eps = 1e-12

                            # Calculate global cosine similarity for dynamic scaling
                            dot_tot = torch.zeros(1, device=self.device)
                            normL_sq = torch.zeros(1, device=self.device)
                            normU_sq = torch.zeros(1, device=self.device)
                            for gl, gu in zip(g_label, g_unlab):
                                if gl is not None and gu is not None:
                                    dot_tot += (gl * gu).sum()
                                    normL_sq += (gl * gl).sum()
                                    normU_sq += (gu * gu).sum()

                            nL = torch.sqrt(normL_sq) + eps
                            nU = torch.sqrt(normU_sq) + eps
                            global_cos_sim = (dot_tot / (nL * nU)).item()

                            # Per-parameter projection when conflicting
                            proj_unlab = []
                            if dot_tot < 0:
                                # Conflicting gradients: apply per-parameter projection
                                for gl, gu in zip(g_label, g_unlab):
                                    if gl is None or gu is None:
                                        proj_unlab.append(
                                            gu if gl is None else gl)
                                        continue
                                    dot = (gl * gu).sum()
                                    if dot < 0:
                                        scale = (
                                            dot / (gl.norm()**2 + eps)).item()
                                        # Check if scale is in valid range [0, 1]
                                        if 0.0 <= scale <= 1.0:
                                            gu = gu - \
                                                (dot / (gl.norm()**2 + eps)) * gl
                                        else:
                                            # Discard gU if scale is out of [0,1] range
                                            gu = torch.zeros_like(gu)
                                    proj_unlab.append(gu)
                                scale_align = 1.0  # No additional scaling for conflicting case
                            else:
                                # Aligned gradients: use cosine similarity as dynamic scaling
                                if 0.0 <= global_cos_sim <= 1.0:
                                    proj_unlab = g_unlab
                                    scale_align = global_cos_sim  # Use cosine similarity as scale
                                else:
                                    # Discard gU if cosine similarity is out of [0,1] range
                                    proj_unlab = [torch.zeros_like(
                                        gu) if gu is not None else None for gu in g_unlab]
                                    scale_align = 0.0  # Effectively no unlabeled contribution

                            final_grads = []
                            for gl, gu in zip(g_label, proj_unlab):
                                if gl is None and gu is None:
                                    final_grads.append(None)
                                elif gl is None:
                                    final_grads.append(scale_align * gu)
                                elif gu is None:
                                    final_grads.append(gl)
                                else:
                                    final_grads.append(gl + scale_align * gu)

                        else:
                            raise ValueError(
                                f"Unrecognised grad_method: {self.grad_method!r}")

                        # apply
                        zero_grads(self.model)
                        set_grads(self.model, final_grads)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), max_norm=1.0)
                        optimizer.step()
                        if self.scheduler_per_step:
                            scheduler.step()

                # ----- end inner steps -----

                # evaluate
                self.model.eval()
                with torch.no_grad():
                    train_pred = self.model(
                        self.train_data.to(self.device)).squeeze()
                    test_pred = self.model(
                        self.test_data.to(self.device)).squeeze()

                y_tr = self.train_label.to(self.device)
                mask_tr = (y_tr >= 0)
                if mask_tr.sum() > 0:
                    train_loss_val = self.test_loss_fn(
                        train_pred[mask_tr], y_tr[mask_tr])
                else:
                    train_loss_val = torch.tensor(0.0, device=self.device)
                test_loss_val = self.test_loss_fn(
                    test_pred,  self.test_label.to(self.device))

                self.train_loss.append(train_loss_val.item())
                self.test_loss.append(test_loss_val.item())

                # treatment effect (analytic or dosage on test/train)
                te_val = 0.0
                if estimate_treatment_effect and self._shortcut_idx is not None and self._shortcut_idx >= 0:
                    te_val = self._compute_te()
                self.test_treatment_effect.append(te_val)

                # shortcut weight history (absolute magnitude)
                if self._shortcut_idx is not None and self._shortcut_idx >= 0:
                    with torch.no_grad():
                        ws = self._shortcut_weight_tensor(self._shortcut_idx)
                        self.shortcut_weight_history.append(abs(ws.item()))
                else:
                    self.shortcut_weight_history.append(0.0)

                # === ORIGINAL MESSAGE STYLE ===
                if classification:
                    # get probabilities
                    train_probs = torch.sigmoid(
                        train_pred[mask_tr]).cpu().numpy()
                    test_probs = torch.sigmoid(test_pred).cpu().numpy()
                    y_train = self.train_label.cpu().numpy()
                    y_test = self.test_label.cpu().numpy()

                    # masks to select only labeled samples (label >= 0)
                    valid_train_mask = y_train >= 0
                    valid_test_mask = y_test >= 0

                    if valid_train_mask.sum() > 0:
                        train_auc = roc_auc_score(
                            y_tr[mask_tr].cpu().numpy(), train_probs) if mask_tr.sum() else 0.0
                    else:
                        train_auc = 0.0
                    if valid_test_mask.sum() > 0:
                        test_auc = roc_auc_score(
                            y_test[valid_test_mask], test_probs[valid_test_mask])
                    else:
                        test_auc = 0.0

                    self.train_auc.append(train_auc)
                    self.test_auc.append(test_auc)

                    msg = (
                        f"Epoch {epoch+1}/{self.num_epochs} " +
                        f"TrainLoss: {train_loss_val:.4f}, " +
                        f"TestLoss: {test_loss_val:.4f}, " +
                        f"TrainAUC: {train_auc:.4f}, " +
                        f"TestAUC: {test_auc:.4f}, " +
                        f"TE: {te_val:.4f}"
                    )
                else:
                    msg = (
                        f"Epoch {epoch+1}/{self.num_epochs} " +
                        f"TrainLoss: {train_loss_val:.4f}, " +
                        f"TestLoss: {test_loss_val:.4f}, " +
                        f"TE: {te_val:.4f}"
                    )

                if not self.scheduler_per_step:
                    scheduler.step()

                pbar.set_description(msg)
                pbar.update(1)
