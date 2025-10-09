#!/usr/bin/env python
# gradients.py  – semi-supervised trainers + GT  + per-step statistics
# ---------------------------------------------------------------------

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List

# ═════════════════════════════════════════════════════════════════════
# A tiny ndarray-wrapper so we can attach a .stats attribute
# ═════════════════════════════════════════════════════════════════════
class Trajectory(np.ndarray):
    """
    A NumPy array subclass that stores extra metadata (here: 'stats').
    Works transparently in numpy / sklearn calls because __array__ is inherited.
    """
    def __new__(cls, data: np.ndarray, stats: Dict):
        obj = np.asarray(data, dtype=float).view(cls)
        obj.stats = stats
        return obj


# ═════════════════════════════════════════════════════════════════════
# helpers
# ═════════════════════════════════════════════════════════════════════
def flatten_params(model: nn.Module) -> np.ndarray:
    with torch.no_grad():
        return np.concatenate([p.detach().view(-1).cpu().numpy()
                               for p in model.parameters()])


def mse_logits(model: nn.Module, x: torch.Tensor,
               y: torch.Tensor) -> torch.Tensor:
    pred = model(x).view(-1)
    return ((pred - y) ** 2).mean()


def mse_probs(model: nn.Module, x: torch.Tensor,
              target_probs: torch.Tensor) -> torch.Tensor:
    pred_probs = torch.sigmoid(model(x).view(-1))
    return ((pred_probs - target_probs) ** 2).mean()


def safe_grad(p: torch.Tensor) -> torch.Tensor:
    return p.grad.clone() if p.grad is not None else torch.zeros_like(p)


def guess_pseudo_labels(model: nn.Module,
                        x_u: torch.Tensor,
                        threshold: float = 0.60):
    with torch.no_grad():
        logits = model(x_u).view(-1)
        probs = torch.sigmoid(logits)
        mask  = (probs > threshold) | (probs < 1.0 - threshold)
        if mask.sum() == 0:
            return (torch.empty(0, device=x_u.device),
                    torch.empty((0, x_u.size(1)), device=x_u.device))
        hard = (probs[mask] > 0.5).float()
        return hard, x_u[mask]


def _init_stats() -> Dict:
    return {
        "conflict_count":    0,
        "alpha_list":        [],   # populated by Fix-Soft
        "proj_scale_list":   []    # populated by Grad-Surgery
    }


# ═════════════════════════════════════════════════════════════════════
# 1) Naïve
# ═════════════════════════════════════════════════════════════════════
def train_naive_5d(model, x_l, y_l, x_u,
                   epochs=50, lr=1e-3, lambda_u=1.0,
                   step_size=10, gamma=0.5):
    opt        = optim.Adam(model.parameters(), lr=lr)
    scheduler  = optim.lr_scheduler.StepLR(opt, step_size, gamma)
    traj, stats_list = [], _init_stats()

    for _ in range(epochs):
        opt.zero_grad()
        loss_l = mse_logits(model, x_l, y_l)
        pseudo, x_conf = guess_pseudo_labels(model, x_u)
        loss_u = mse_probs(model, x_conf, pseudo) * lambda_u if x_conf.numel() else 0.0
        (loss_l + loss_u).backward()
        opt.step()

        traj.append(flatten_params(model))

    return Trajectory(np.stack(traj), stats_list)


# ═════════════════════════════════════════════════════════════════════
# 2) Fix-Step (hard gate on conflicts)
# ═════════════════════════════════════════════════════════════════════
def train_fix_5d(model, x_l, y_l, x_u,
                 epochs=50, lr=1e-3, lambda_u=10.0,
                 step_size=10, gamma=0.5):
    opt       = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(opt, step_size, gamma)
    traj, stats = [], _init_stats()
    torch.manual_seed(0)

    for _ in range(epochs):
        pseudo, x_conf = guess_pseudo_labels(model, x_u)

        # g_L
        opt.zero_grad()
        mse_logits(model, x_l, y_l).backward(retain_graph=True)
        gL = [safe_grad(p) for p in model.parameters()]

        # g_U
        opt.zero_grad()
        if x_conf.numel():
            (mse_probs(model, x_conf, pseudo) * lambda_u).backward()
        gU = [safe_grad(p) for p in model.parameters()]

        dot = sum((a * b).sum() for a, b in zip(gL, gU))
        if dot < 0:
            stats["conflict_count"] += 1
        final = [a + b if dot >= 0 else a for a, b in zip(gL, gU)]

        opt.zero_grad()
        for p, g in zip(model.parameters(), final):
            p.grad = g
        opt.step()

        traj.append(flatten_params(model))

    return Trajectory(np.stack(traj), stats)


# ═════════════════════════════════════════════════════════════════════
# 3) Gradient Surgery (PCGrad) with optional scaling when ALIGNED
#    - If global cosine(gL, gU) <= 0 : project each gu off gl (per-parameter).
#    - Else (aligned): scale gu by gs_scale_aligned (≤1.0 to temper, >1.0 to amplify).
#    - Logs: conflict_count and the per-parameter projection scales used.
# ═════════════════════════════════════════════════════════════════════
def train_gs_5d(model, x_l, y_l, x_u,
                epochs=50, lr=1e-3, lambda_u=10.0,
                step_size=10, gamma=0.5):
    """
    Fix-a-Shortcut (GS-Align): Projection-based variant that explicitly removes 
    shortcut-aligned components while preserving causal-consistent signal.
    
    Mathematical formulation:
    - When gradients conflict (<g_L, g_U> < 0): g_U - φ * g_L where φ = <g_U, g_L> / ||g_L||^2
    - When gradients align (<g_L, g_U> >= 0): α * g_U where α = max(cos_sim, 0)
    - If |φ| ∉ [0,1]: discard g_U (set to 0)
    """
    opt       = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(opt, step_size, gamma)
    traj, stats = [], _init_stats()
    torch.manual_seed(1)

    eps = 1e-12

    for _ in range(epochs):
        pseudo, x_conf = guess_pseudo_labels(model, x_u)

        # g_L
        opt.zero_grad()
        mse_logits(model, x_l, y_l).backward(retain_graph=True)
        gL = [safe_grad(p) for p in model.parameters()]

        # g_U
        opt.zero_grad()
        if x_conf.numel():
            (mse_probs(model, x_conf, pseudo) * lambda_u).backward()
        gU = [safe_grad(p) for p in model.parameters()]

        # ---- global cosine for alignment check ----
        dot_tot = sum((gl * gu).sum() for gl, gu in zip(gL, gU))
        nL = torch.sqrt(sum((gl * gl).sum() for gl in gL)) + eps
        nU = torch.sqrt(sum((gu * gu).sum() for gu in gU)) + eps
        cos_sim = (dot_tot / (nL * nU)).item()

        # ---- per-parameter PCGrad projection when conflicting ----
        proj_unlab: List[torch.Tensor] = []
        if dot_tot < 0:
            # Conflicting gradients: apply per-parameter projection g_U - φ * g_L
            stats["conflict_count"] += 1
            for gl, gu in zip(gL, gU):
                if gl is None or gu is None:
                    proj_unlab.append(gu if gl is None else gl)
                    continue
                dot = (gl * gu).sum()
                if dot < 0:
                    phi = (dot / (gl.norm()**2 + eps))          # φ = <g_U, g_L> / ||g_L||^2
                    # Only use and log projections that are reasonable (|φ| ≤ 1.0)
                    if abs(phi.item()) <= 1.0:                  
                        stats["proj_scale_list"].append(abs(phi.item()))  # log absolute value of scale used
                        gu = gu - phi * gl                      # g_U - φ * g_L
                    else:
                        gu = torch.zeros_like(gu)               # discard if |φ| > 1
                proj_unlab.append(gu)
            scale_align = 1.0  # No additional scaling for conflicting case
        else:
            # Aligned gradients: use α = max(cos_sim, 0) scaling
            proj_unlab = gU
            scale_align = max(cos_sim, 0.0)  # α = max(cos_sim, 0) as per mathematical description
        # ---- combine (scaled) grads ----
        final_grads: List[torch.Tensor] = []
        for gl, gu in zip(gL, proj_unlab):
            if gl is None and gu is None:
                final_grads.append(None)
            elif gl is None:
                final_grads.append(scale_align * gu)
            elif gu is None:
                final_grads.append(gl)
            else:
                final_grads.append(gl + scale_align * gu)

        opt.zero_grad()
        for p, g in zip(model.parameters(), final_grads):
            p.grad = g
        opt.step()

        traj.append(flatten_params(model))

    return Trajectory(np.stack(traj), stats)


# ═════════════════════════════════════════════════════════════════════
# 4) Fix-Soft
# ═════════════════════════════════════════════════════════════════════
def train_fix_soft_5d(model, x_l, y_l, x_u,
                      epochs=50, lr=1e-3, lambda_u=10.0,
                      step_size=10, gamma=0.5):
    opt       = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(opt, step_size, gamma)
    traj, stats = [], _init_stats()
    torch.manual_seed(2)

    for _ in range(epochs):
        pseudo, x_conf = guess_pseudo_labels(model, x_u)

        # g_L
        opt.zero_grad()
        mse_logits(model, x_l, y_l).backward(retain_graph=True)
        gL = [safe_grad(p) for p in model.parameters()]

        # g_U
        opt.zero_grad()
        if x_conf.numel():
            (mse_probs(model, x_conf, pseudo) * lambda_u).backward()
        gU = [safe_grad(p) for p in model.parameters()]

        dot   = sum((a * b).sum() for a, b in zip(gL, gU))
        normL = torch.sqrt(sum((a * a).sum() for a in gL))
        normU = torch.sqrt(sum((b * b).sum() for b in gU))
        alpha = float((dot / (normL * normU + 1e-15)).clamp(min=0.0))
        stats["alpha_list"].append(alpha)
        if dot < 0:
            stats["conflict_count"] += 1

        final = [a + alpha * b for a, b in zip(gL, gU)]

        opt.zero_grad()
        for p, g in zip(model.parameters(), final):
            p.grad = g
        opt.step()

        traj.append(flatten_params(model))

    return Trajectory(np.stack(traj), stats)


# ═════════════════════════════════════════════════════════════════════
# 5) Fully-supervised ground-truth
# ═════════════════════════════════════════════════════════════════════
def train_gt_5d(model: nn.Module,
                x_full: torch.Tensor,
                y_full: torch.Tensor,
                *,
                epochs: int = 50,
                lr: float = 1e-3):
    opt    = optim.Adam(model.parameters(), lr=lr)
    traj   = []
    stats  = _init_stats()      # GT has no conflicts by definition

    for _ in range(epochs):
        opt.zero_grad()
        loss = mse_logits(model, x_full, y_full)
        loss.backward()
        opt.step()

        traj.append(flatten_params(model))

    return Trajectory(np.stack(traj), stats)
