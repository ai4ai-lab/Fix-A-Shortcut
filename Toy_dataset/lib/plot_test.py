#!/usr/bin/env python
# plot_test.py      ← now with correlation & TE plots
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Callable, List, Dict, Any
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D


from data import generate_synthetic_base, generate_training_test_data
from model import toy_nn
from gradients import (
    train_naive_5d, train_fix_5d, train_gs_5d, train_fix_soft_5d
)
from gradients import flatten_params

# ═══════════════════════════════════════════════════════════════════
# METHOD NAME MAPPING - Centralized mapping for display names
# ═══════════════════════════════════════════════════════════════════
METHOD_DISPLAY_NAMES = {
    'naive': 'Vanilla SSL',
    'fix_step': 'FAS',
    'fix_soft': 'FAS-soft',
    'grad_surgery': 'FAS-GS-Align'
}

METHOD_COLORS = {
    'naive': '#2878b5',
    'fix_step': '#9ac9db',
    'grad_surgery': '#f8ac8c',
    'fix_soft': '#c82423'
}

def get_display_name(method_key):
    """Get the display name for a method key."""
    return METHOD_DISPLAY_NAMES.get(method_key, method_key)

def get_method_color(method_key):
    """Get the color for a method key."""
    return METHOD_COLORS.get(method_key, 'black')

# ──────────────────────────── small helper


def _set_flattened_params(model: torch.nn.Module, flat: np.ndarray) -> None:
    """Load a flattened NumPy parameter vector back into a model."""
    pointer = 0
    for p in model.parameters():
        n = p.numel()
        chunk = torch.from_numpy(flat[pointer:pointer + n]).view(p.shape)
        p.data.copy_(chunk.to(p.device))
        pointer += n


# ═══════════════════════════ 1)  shortcut weight trajectory
def plot_test_shortcut_weight_over_epochs(
    results: dict,
    methods=None,
    hidden_dim: int = 5,
    input_dim: int = 5,
    ax=None
):
    if methods is None:
        methods = ['naive', 'fix_step', 'grad_surgery', 'fix_soft']

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    fc1_size = hidden_dim * input_dim
    for name in methods:
        traj = results['trajectories'][name]          # [epoch × param_dim]
        norms = [np.linalg.norm(e[:fc1_size]
                                .reshape(hidden_dim, input_dim)[:, -1])
                 for e in traj]
        ax.plot(range(1, len(norms)+1), norms, label=get_display_name(name))

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Shortcut weight ‖·‖')
    ax.set_title('Shortcut weight trajectory')
    ax.legend()
    ax.grid(True)
    return ax


# ═══════════════════════════ 2)  generalization gap
def compute_generalization_gap(
    results, x_train, y_train, x_test, y_test, device="cpu"
):
    device = torch.device(device)
    inputdim = x_train.shape[1]
    methods = ['naive', 'fix_step', 'grad_surgery', 'fix_soft']
    xt, yt = map(lambda a: torch.tensor(a, dtype=torch.float32, device=device),
                 (x_train, y_train))
    xv, yv = map(lambda a: torch.tensor(a, dtype=torch.float32, device=device),
                 (x_test,  y_test))
    gaps = {}
    for m in methods:
        model = toy_nn(inputdim, 1).to(device)
        _set_flattened_params(model, results['final_weights'][m])
        with torch.no_grad():
            mse_tr = ((model(xt).view(-1) - yt)**2).mean().item()
            mse_te = ((model(xv).view(-1) - yv)**2).mean().item()
        gaps[m] = {"train_mse": mse_tr,
                   "test_mse": mse_te, "gap": mse_te-mse_tr}
    return gaps


def plot_generalization_gap(gaps, ax=None):
    methods = list(gaps.keys())
    gapvals = [gaps[m]['gap'] for m in methods]
    # Use display names for x-axis labels
    display_names = [get_display_name(m) for m in methods]
    palette = [get_method_color(m) for m in methods]

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(display_names, gapvals, color=palette)
    ax.set_ylabel(
        r'$ \mathrm{MSE}_{\mathrm{test}} - \mathrm{MSE}_{\mathrm{train}} $')
    ax.set_title('Generalization gap')
    ax.grid(True, axis='y', ls='--')
    return ax
# ─────────────────────────────────────────────────────────────────────────────
# 7) Causal‐Coefficient Error Trajectory
# ─────────────────────────────────────────────────────────────────────────────

def compute_total_causal_error_trajectories(
    results: dict,
    true_betas: np.ndarray,
    hidden_dim: int = 5,
    input_dim: int = 5
) -> dict[str, np.ndarray]:
    """
    For each method in results['trajectories'], compute an array of length epochs:
      total causal error at epoch e =
        sum_{j=0..3} |‖W1[:,j]‖ - true_betas[j]|.
    We assume the 5th feature (index 4) is the shortcut and ignored here.
    """
    errors: dict[str, np.ndarray] = {}
    n_feat = len(true_betas)  # e.g. 4
    for name, traj in results['trajectories'].items():
        err_list = []
        for vec in traj:
            # extract first‐layer weights
            fc1 = vec[: hidden_dim * input_dim].reshape(hidden_dim, input_dim)
            # per‐feature norm
            norms = np.linalg.norm(fc1, axis=0)  # length input_dim
            # only the first n_feat correspond to c1,c2,u1,u2
            err = np.abs(norms[:n_feat] - true_betas).sum()
            err_list.append(err)
        errors[name] = np.array(err_list)
    return errors


def plot_causal_weight_error_over_epochs(
    results: dict,
    true_w: np.ndarray,
    methods=None,
    hidden_dim: int = 5,
    input_dim: int = 5,
    ax=None
):
    """
    Plot the sum of absolute errors on the causal inputs {C1,C2,U1,U2}
    over epochs for each specified method.

    Parameters
    ----------
    results : dict
        Output of `synthetic_experiments()`, must contain 'trajectories'.
    true_w : 1D array of length input_dim
        The true generative weights [w_C1, w_C2, w_U1, w_U2, w_S].
    methods : list of str, optional
        Which methods to include (defaults to ['naive','fix_step','grad_surgery','fix_soft']).
    hidden_dim : int
        Number of hidden units in the first layer of the toy_nn.
    input_dim : int
        Total number of input features (including the shortcut S).
    ax : matplotlib Axes, optional
        If given, plot onto this; otherwise create a new figure.

    Returns
    -------
    ax : matplotlib Axes
    """
    if methods is None:
        methods = ['naive', 'fix_step', 'grad_surgery', 'fix_soft']

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    fc1_size = hidden_dim * input_dim
    causal_idx = slice(0, 4)   # only C1,C2,U1,U2

    for name in methods:
        traj = results['trajectories'][name]  # shape (epochs, param_dim)
        errors = []
        for epoch_params in traj:
            # reshape first-layer weights to (hidden_dim, input_dim)
            W1 = epoch_params[:fc1_size].reshape(hidden_dim, input_dim)
            learned_norms = np.linalg.norm(W1, axis=0)  # length = input_dim
            # sum |learned - true| over the 4 causal features
            err = np.sum(np.abs(learned_norms[causal_idx] - true_w[causal_idx]))
            errors.append(err)
        ax.plot(range(1, len(errors) + 1), errors, label=get_display_name(name))

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Sum abs error on causal weights')
    ax.set_title('Causal‐Weight Error Over Training')
    ax.legend()
    ax.grid(True)
    return ax



# ═══════════════════════════ 3)  correlation heat-map  ══════════════════
def compute_corr_heatmap_data(results, x_test, feature_names, device="cpu"):
    """
    Returns a pandas-like 2-D list: rows = methods, cols = features,
    filled with Pearson r between *model prediction* and each X[:,j].
    """
    import pandas as pd
    device = torch.device(device)
    xv = torch.tensor(x_test, dtype=torch.float32, device=device)
    methods = ['naive', 'fix_step', 'grad_surgery', 'fix_soft']
    rows = []

    for m in methods:
        model = toy_nn(x_test.shape[1], 1).to(device)
        _set_flattened_params(model, results['final_weights'][m])
        with torch.no_grad():
            y_hat = model(xv).view(-1).cpu().numpy()
        corr = [np.corrcoef(y_hat, x_test[:, k])[0, 1]
                for k in range(x_test.shape[1])]
        rows.append(corr)

    return pd.DataFrame(rows, index=methods, columns=feature_names)


def plot_corr_heatmap(df, ax=None, cmap="vlag"):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(df, annot=True, fmt=".2f", cmap=cmap, center=0, ax=ax,
                cbar_kws=dict(label="Pearson r"))
    ax.set_title("Prediction ↔ feature correlation")
    return ax


# ═══════════════════════════ 4)  treatment-effect trajectory
def _treatment_effect(model, hidden_dim, input_dim, delta=1.0, device="cpu"):
    """
    Simple proxy: predict with S = +Δ vs S = -Δ (other inputs zero).
    TE = f(x_S=+Δ) - f(x_S=-Δ)
    """
    x_plus = torch.zeros((1, input_dim), device=device)
    x_plus[0, -1] = +delta
    x_minus = torch.zeros((1, input_dim), device=device)
    x_minus[0, -1] = -delta
    with torch.no_grad():
        te = (model(x_plus) - model(x_minus)).item()
    return te


def plot_treatment_effect_trajectory(
    results, hidden_dim=5, input_dim=5, ax=None, delta=1.0, device="cpu"
):
    """
    Plot TE(S)=f(+Δ)-f(-Δ) over training epochs for every method.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    device = torch.device(device)
    methods = ['naive', 'fix_step', 'grad_surgery', 'fix_soft']

    for m in methods:
        traj = results['trajectories'][m]          # [epochs × param_dim]
        tes = []
        for flat in traj:
            model = toy_nn(input_dim, 1).to(device)
            _set_flattened_params(model, flat)
            tes.append(_treatment_effect(model, hidden_dim, input_dim,
                                         delta=delta, device=device))
        ax.plot(range(1, len(tes)+1), tes, label=get_display_name(m))

    ax.set_xlabel('Epoch')
    ax.set_ylabel(f'TE_S (Δ={delta})')
    ax.set_title('Shortcut treatment-effect trajectory')
    ax.legend()
    ax.grid(True)
    return ax


def evaluate_metric(
    model: torch.nn.Module,
    x_l: np.ndarray, y_l: np.ndarray,
    x_test: np.ndarray, y_test: np.ndarray,
    metric: str = "train",
    device: str = "cpu"
) -> float:
    """
    Compute MSE on the chosen split.
    metric = "train"  → return MSE(model(x_l), y_l)
    metric = "test"   → return MSE(model(x_test), y_test)
    """
    model.eval()
    dev = torch.device(device)
    # convert to tensors on the right device
    xt = torch.tensor(x_l, dtype=torch.float32, device=dev)
    yt = torch.tensor(y_l, dtype=torch.float32, device=dev)
    xv = torch.tensor(x_test,  dtype=torch.float32, device=dev)
    yv = torch.tensor(y_test,  dtype=torch.float32, device=dev)

    with torch.no_grad():
        if metric == "train":
            pred = model(xt).view(-1)
            return float(((pred - yt) ** 2).mean().item())
        elif metric == "test":
            pred = model(xv).view(-1)
            return float(((pred - yv) ** 2).mean().item())
        else:
            raise ValueError(
                f"Unknown metric '{metric}', expected 'train' or 'test'.")


def plot_test_mse_over_epochs(
    num_samples=2000,
    rho_cu=0.0,
    sigma_y=0.1,
    shortcut_mode="y_noise",
    beta=1.0,
    sigma_s=0.1,
    shortcut_ratio=1.0,
    test_size=0.5,
    random_state=0,
    n_labeled=50,
    epochs=100,
    lr=1e-3,
    lambda_u=10.0,
    device="cpu"
):
    # Generate synthetic data
    X_base, y_all, y_clean = generate_synthetic_base(
        num_samples=num_samples,
        seed=random_state,
        rho_cu=rho_cu,
        sigma_y=sigma_y
    )
    x_train, y_train, x_test, y_test = generate_training_test_data(
        X_base,
        y_all,
        y_clean,
        shortcut_mode=shortcut_mode,
        beta=beta,
        sigma_s=sigma_s,
        shortcut_ratio=shortcut_ratio,
        test_size=test_size,
        random_state=random_state
    )
    # Tensor conversion
    x_train = torch.tensor(x_train, dtype=torch.float32, device=device)
    y_train = torch.tensor(y_train, dtype=torch.float32, device=device)
    x_test = torch.tensor(x_test, dtype=torch.float32, device=device)
    y_test = torch.tensor(y_test, dtype=torch.float32, device=device)

    # Split labeled / unlabeled
    perm = torch.randperm(x_train.size(0), device=device)
    x_l = x_train[perm[:n_labeled]]
    y_l = y_train[perm[:n_labeled]]
    x_u = x_train[perm[n_labeled:]]

    # Methods
    methods = {
        'naive': train_naive_5d,
        'fix_step': train_fix_5d,
        'grad_surgery': train_gs_5d,
        'fix_soft': train_fix_soft_5d
    }

    # Record test MSE
    test_mse = {get_display_name(name): [] for name in methods}

    for name, train_fn in methods.items():
        display_name = get_display_name(name)
        # initialize model
        model = toy_nn(x_l.size(1), 1).to(device)
        # iterative one-epoch training
        for ep in range(epochs):
            # train one epoch
            train_fn(model, x_l, y_l, x_u,
                     epochs=1, lr=lr, lambda_u=lambda_u)
            # evaluate test MSE
            with torch.no_grad():
                preds = model(x_test).view(-1)
                mse = ((preds - y_test)**2).mean().item()
            test_mse[display_name].append(mse)

    # Plot
    plt.figure(figsize=(6, 4))
    for name, mses in test_mse.items():
        plt.plot(range(1, epochs+1), mses, label=name)
    plt.xlabel('Epoch')
    plt.ylabel('Test MSE')
    plt.title('Test MSE over Epochs for Shortcut Methods')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from typing import List, Dict, Any
def compute_shortcut_ratio(model) -> float:
    """
    Compute the ratio of the shortcut‐weight magnitude to the total weight magnitude.
    Uses flatten_params to retrieve the full flattened parameter vector.
    Assumes the last element corresponds to the shortcut weight.
    """
    # Flatten all model parameters into a 1D NumPy array
    w = flatten_params(model)  # shape (D,)
    
    # By convention, assume the last parameter is the shortcut weight
    shortcut_idx = -1
    w_s = w[shortcut_idx]
    
    total_norm = np.linalg.norm(w)
    if total_norm == 0:
        return 0.0
    return abs(w_s) / total_norm

def evaluate_metric(model, x_l, y_l, x_test, y_test, metric="shortcut"):
        return compute_shortcut_ratio(model)

def compute_metric_grid_auto(
    model_5d: torch.nn.Module,
    traj_list: list[np.ndarray],        # list of [epochs × 2] in PC-space
    pca: PCA,
    param_mean: np.ndarray,
    evaluate_fn: Callable[[torch.nn.Module], float],
    margin_ratio: float = 0.05,         # 5 % breathing room
    steps: int = 60
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    1.  Find the bounding box of *all* trajectories in PC1/PC2.
    2.  Expand it by `margin_ratio`.
    3.  Build a (steps×steps) grid inside that box and evaluate `evaluate_fn`
        (e.g. shortcut-norm) at each grid point.
    Returns PC1_grid, PC2_grid, VALUE_grid.
    """
    # ----- 1) bounding box for every trajectory point
    all_pc1 = np.concatenate([t[:, 0] for t in traj_list])
    all_pc2 = np.concatenate([t[:, 1] for t in traj_list])
    min_x, max_x = all_pc1.min(), all_pc1.max()
    min_y, max_y = all_pc2.min(), all_pc2.max()

    # ----- 2) enlarge by a fixed ratio
    dx = (max_x - min_x) * margin_ratio + 1e-9
    dy = (max_y - min_y) * margin_ratio + 1e-9
    min_x, max_x = min_x - dx, max_x + dx
    min_y, max_y = min_y - dy, max_y + dy

    # ----- 3) mesh-grid in that region
    pc1_vals = np.linspace(min_x, max_x, steps)
    pc2_vals = np.linspace(min_y, max_y, steps)
    PC1, PC2 = np.meshgrid(pc1_vals, pc2_vals, indexing="xy")
    METRIC = np.zeros_like(PC1)

    # keep a copy of original weights
    orig_flat = flatten_params(model_5d)

    for i in range(steps):
        for j in range(steps):
            pc = np.array([PC1[i, j], PC2[i, j]])
            full_w = param_mean + pca.inverse_transform(pc.reshape(1, -1))[0]
            _set_flattened_params(model_5d, full_w)
            METRIC[i, j] = evaluate_fn(model_5d)

    _set_flattened_params(model_5d, orig_flat)
    return PC1, PC2, METRIC


# ──────────────────────────────────────────────────────────────────
# 2) Plot: NO axis-rescaling, so the contour covers the whole view
# ──────────────────────────────────────────────────────────────────
def plot_contour_trajectories_as_arrows_skip(
    PC1: np.ndarray,
    PC2: np.ndarray,
    metric2d: np.ndarray,
    traj_list: list[np.ndarray],
    names_list: list[str],
    title: str,
    label: str,
    cmap: str = "viridis",
    skip: int = 50
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.grid(False)

    # ----- contour
    levels = np.linspace(metric2d.min(), metric2d.max(), 40)
    cf = ax.contourf(PC1, PC2, metric2d, levels=levels,
                     cmap=cmap, alpha=0.95, zorder=1)
    fig.colorbar(cf, ax=ax, label=label, pad=0.01)

    # lock limits *before* drawing paths → no white zones
    ax.set_xlim(PC1.min(), PC1.max())
    ax.set_ylim(PC2.min(), PC2.max())

    # ----- optimisation paths
    colour_cycle = ["red", "blue", "green", "magenta","black","cyan"]
    for traj, name, col in zip(traj_list, names_list, colour_cycle):
        # arrows
        for k in range(0, len(traj) - 1, skip):
            head = traj[min(k + skip, len(traj) - 1)]
            tail = traj[k]
            ax.arrow(tail[0], tail[1], *(head - tail),
                     head_width=0.03, head_length=0.06,
                     fc=col, ec=col, linewidth=1.4, alpha=0.85, zorder=2)

        # start / end markers - use display names for legend
        ax.scatter(*traj[0],  c=col, marker="o", s=90, ec="k",
                   zorder=3, label=f"{name} start")
        ax.scatter(*traj[-1], c=col, marker="s", s=90, ec="k",
                   zorder=3, label=f"{name} end")

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.legend(fontsize=9, ncol=2, frameon=True)
    plt.tight_layout()
    plt.show()


# ═══════════════════════════════════════════════════════════════════
# Subspace Visualization Functions
# ═══════════════════════════════════════════════════════════════════

def plot_subspaces_2d(
    X_data: np.ndarray,
    y_data: np.ndarray = None,
    feature_names: List[str] = None,
    title: str = "Feature Subspaces (2D Projection)",
    ax=None,
    alpha: float = 0.6,
    s: int = 50
):
    """
    Plot known, unknown, and shortcut subspaces in 2D using PCA projection.
    
    Parameters:
    -----------
    X_data : np.ndarray, shape (n_samples, n_features)
        Input feature matrix. Expected format: [C1, C2, U1, U2, S]
    y_data : np.ndarray, optional
        Target values for color coding points
    feature_names : List[str], optional
        Names for features, defaults to ['C1', 'C2', 'U1', 'U2', 'S']
    title : str
        Plot title
    ax : matplotlib axes, optional
        Axes to plot on, creates new if None
    alpha : float
        Point transparency
    s : int
        Point size
        
    Returns:
    --------
    ax : matplotlib axes
    """
    if feature_names is None:
        feature_names = ['C1', 'C2', 'U1', 'U2', 'S']
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    # Apply PCA for 2D visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_data)
    
    # Define subspaces (assuming 5D input: [C1, C2, U1, U2, S])
    known_idx = [0, 1]      # C1, C2 - known causal features
    unknown_idx = [2, 3]    # U1, U2 - unknown causal features  
    shortcut_idx = [4]      # S - shortcut feature
    
    # Create subspace data
    known_data = X_data[:, known_idx]
    unknown_data = X_data[:, unknown_idx]
    shortcut_data = X_data[:, shortcut_idx]
    
    # Color points by target if provided, otherwise by subspace
    if y_data is not None:
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y_data, 
                           cmap='viridis', alpha=alpha, s=s)
        plt.colorbar(scatter, ax=ax, label='Target Value')
    else:
        # Color by dominant subspace contribution
        known_strength = np.linalg.norm(known_data, axis=1)
        unknown_strength = np.linalg.norm(unknown_data, axis=1)
        shortcut_strength = np.abs(shortcut_data.flatten())
        
        # Determine dominant subspace for each point
        strengths = np.column_stack([known_strength, unknown_strength, shortcut_strength])
        dominant = np.argmax(strengths, axis=1)
        
        colors = ['#2878b5', '#f8ac8c', '#c82423']  # Known, Unknown, Shortcut
        labels = ['Known Causal (C1,C2)', 'Unknown Causal (U1,U2)', 'Shortcut (S)']
        
        for i, (color, label) in enumerate(zip(colors, labels)):
            mask = dominant == i
            if np.any(mask):
                ax.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                          c=color, label=label, alpha=alpha, s=s)
    
    # Add feature contribution vectors
    # Transform feature directions to PCA space
    feature_directions = pca.components_.T  # (n_features, 2)
    
    # Scale vectors for visibility
    scale_factor = np.max(np.abs(X_pca)) * 0.3
    
    for i, (direction, name) in enumerate(zip(feature_directions, feature_names)):
        ax.arrow(0, 0, direction[0] * scale_factor, direction[1] * scale_factor,
                head_width=scale_factor*0.05, head_length=scale_factor*0.08,
                fc='red', ec='red', alpha=0.7, linewidth=2)
        ax.text(direction[0] * scale_factor * 1.1, direction[1] * scale_factor * 1.1,
               name, fontsize=10, fontweight='bold', ha='center', va='center')
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    return ax


def plot_subspaces_3d(
    X_data: np.ndarray,
    y_data: np.ndarray = None,
    feature_names: List[str] = None,
    title: str = "Feature Subspaces (3D Projection)",
    figsize: tuple = (12, 9),
    alpha: float = 0.6,
    s: int = 30
):
    """
    Plot known, unknown, and shortcut subspaces in 3D using PCA projection.
    
    Parameters:
    -----------
    X_data : np.ndarray, shape (n_samples, n_features)
        Input feature matrix. Expected format: [C1, C2, U1, U2, S]
    y_data : np.ndarray, optional
        Target values for color coding points
    feature_names : List[str], optional
        Names for features, defaults to ['C1', 'C2', 'U1', 'U2', 'S']
    title : str
        Plot title
    figsize : tuple
        Figure size
    alpha : float
        Point transparency
    s : int
        Point size
        
    Returns:
    --------
    fig, ax : matplotlib figure and 3D axes
    """
    if feature_names is None:
        feature_names = ['C1', 'C2', 'U1', 'U2', 'S']
    
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Apply PCA for 3D visualization
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_data)
    
    # Define subspaces (assuming 5D input: [C1, C2, U1, U2, S])
    known_idx = [0, 1]      # C1, C2 - known causal features
    unknown_idx = [2, 3]    # U1, U2 - unknown causal features  
    shortcut_idx = [4]      # S - shortcut feature
    
    # Create subspace data
    known_data = X_data[:, known_idx]
    unknown_data = X_data[:, unknown_idx]
    shortcut_data = X_data[:, shortcut_idx]
    
    # Color points by target if provided, otherwise by subspace
    if y_data is not None:
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], 
                           c=y_data, cmap='viridis', alpha=alpha, s=s)
        fig.colorbar(scatter, ax=ax, label='Target Value', shrink=0.5)
    else:
        # Color by dominant subspace contribution
        known_strength = np.linalg.norm(known_data, axis=1)
        unknown_strength = np.linalg.norm(unknown_data, axis=1)
        shortcut_strength = np.abs(shortcut_data.flatten())
        
        # Determine dominant subspace for each point
        strengths = np.column_stack([known_strength, unknown_strength, shortcut_strength])
        dominant = np.argmax(strengths, axis=1)
        
        colors = ['#2878b5', '#f8ac8c', '#c82423']  # Known, Unknown, Shortcut
        labels = ['Known Causal (C1,C2)', 'Unknown Causal (U1,U2)', 'Shortcut (S)']
        
        for i, (color, label) in enumerate(zip(colors, labels)):
            mask = dominant == i
            if np.any(mask):
                ax.scatter(X_pca[mask, 0], X_pca[mask, 1], X_pca[mask, 2],
                          c=color, label=label, alpha=alpha, s=s)
    
    # Add feature contribution vectors
    # Transform feature directions to PCA space
    feature_directions = pca.components_.T  # (n_features, 3)
    
    # Scale vectors for visibility
    scale_factor = np.max(np.abs(X_pca)) * 0.3
    
    for i, (direction, name) in enumerate(zip(feature_directions, feature_names)):
        ax.quiver(0, 0, 0, 
                 direction[0] * scale_factor, 
                 direction[1] * scale_factor, 
                 direction[2] * scale_factor,
                 color='red', alpha=0.8, arrow_length_ratio=0.1, linewidth=2)
        ax.text(direction[0] * scale_factor * 1.2, 
               direction[1] * scale_factor * 1.2, 
               direction[2] * scale_factor * 1.2,
               name, fontsize=10, fontweight='bold')
    
    # Add subspace planes (optional - can be commented out if too cluttered)
    # Known subspace plane (C1-C2)
    known_directions = feature_directions[known_idx]  # (2, 3)
    if len(known_directions) >= 2:
        # Create a mesh for the known subspace plane
        u = np.linspace(-scale_factor, scale_factor, 10)
        v = np.linspace(-scale_factor, scale_factor, 10)
        U, V = np.meshgrid(u, v)
        plane_points = (U.flatten()[:, None] * known_directions[0] + 
                       V.flatten()[:, None] * known_directions[1])
        X_plane = plane_points[:, 0].reshape(U.shape)
        Y_plane = plane_points[:, 1].reshape(U.shape)
        Z_plane = plane_points[:, 2].reshape(U.shape)
        ax.plot_surface(X_plane, Y_plane, Z_plane, alpha=0.1, color='blue')
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.2%} variance)')
    ax.set_title(title)
    ax.legend()
    
    return fig, ax


def plot_subspace_analysis_combined(
    X_data: np.ndarray,
    y_data: np.ndarray = None,
    feature_names: List[str] = None,
    figsize: tuple = (15, 6)
):
    """
    Create a combined plot showing both 2D and 3D subspace visualizations.
    
    Parameters:
    -----------
    X_data : np.ndarray
        Input feature matrix
    y_data : np.ndarray, optional
        Target values
    feature_names : List[str], optional
        Feature names
    figsize : tuple
        Figure size
        
    Returns:
    --------
    fig : matplotlib figure
    """
    fig = plt.figure(figsize=figsize)
    
    # 2D plot
    ax1 = fig.add_subplot(121)
    plot_subspaces_2d(X_data, y_data, feature_names, 
                     "Feature Subspaces (2D PCA)", ax1)
    
    # 3D plot
    ax2 = fig.add_subplot(122, projection='3d')
    
    # Reimplement 3D plotting logic for subplot
    if feature_names is None:
        feature_names = ['C1', 'C2', 'U1', 'U2', 'S']
    
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_data)
    
    known_idx = [0, 1]
    unknown_idx = [2, 3] 
    shortcut_idx = [4]
    
    known_data = X_data[:, known_idx]
    unknown_data = X_data[:, unknown_idx]
    shortcut_data = X_data[:, shortcut_idx]
    
    if y_data is not None:
        scatter = ax2.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], 
                            c=y_data, cmap='viridis', alpha=0.6, s=30)
    else:
        known_strength = np.linalg.norm(known_data, axis=1)
        unknown_strength = np.linalg.norm(unknown_data, axis=1)
        shortcut_strength = np.abs(shortcut_data.flatten())
        
        strengths = np.column_stack([known_strength, unknown_strength, shortcut_strength])
        dominant = np.argmax(strengths, axis=1)
        
        colors = ['#2878b5', '#f8ac8c', '#c82423']
        labels = ['Known Causal (C1,C2)', 'Unknown Causal (U1,U2)', 'Shortcut (S)']
        
        for i, (color, label) in enumerate(zip(colors, labels)):
            mask = dominant == i
            if np.any(mask):
                ax2.scatter(X_pca[mask, 0], X_pca[mask, 1], X_pca[mask, 2],
                           c=color, label=label, alpha=0.6, s=30)
    
    ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
    ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
    ax2.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.2%})')
    ax2.set_title("Feature Subspaces (3D PCA)")
    ax2.legend()
    
    plt.tight_layout()
    return fig


def analyze_shortcut_gradient_directions(model, x_labeled, y_labeled, x_unlabeled, 
                                       initial_weights, lambda_u=10.0, device="cpu"):
    """
    Analyze how labeled vs unlabeled gradients affect shortcut weights.
    Since true shortcut weight should be 0, compute optimal gradient direction
    and compare with actual gradients from labeled/unlabeled data.
    
    Parameters:
    -----------
    model : torch.nn.Module
        The neural network model
    x_labeled : np.ndarray
        Labeled input data
    y_labeled : np.ndarray  
        Labeled target data
    x_unlabeled : np.ndarray
        Unlabeled input data
    initial_weights : np.ndarray
        Initial model weights (flattened)
    lambda_u : float
        Unlabeled loss weight
    device : str
        Device to run computation on
        
    Returns:
    --------
    dict : Analysis results containing gradients and metrics
    """
    # Helper functions for gradient computation
    def mse_logits(model, x, y):
        pred = model(torch.tensor(x, dtype=torch.float32).to(device))
        return torch.nn.MSELoss()(pred, torch.tensor(y, dtype=torch.float32).to(device).unsqueeze(1))
    
    def mse_probs(model, x, pseudo):
        pred = model(torch.tensor(x, dtype=torch.float32).to(device))
        return torch.nn.MSELoss()(pred, torch.tensor(pseudo, dtype=torch.float32).to(device).unsqueeze(1))
    
    def guess_pseudo_labels(model, x_u):
        model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(model(torch.tensor(x_u, dtype=torch.float32).to(device))).cpu().numpy().flatten()
        model.train()
        mask = (probs > 0.1) & (probs < 0.9)  # confidence mask
        if mask.sum() == 0:
            return np.array([]), np.array([]).reshape(-1, x_u.shape[1])
        hard = (probs[mask] > 0.5).astype(float)
        return hard, x_u[mask]
    
    def safe_grad(p):
        return p.grad.clone() if p.grad is not None else torch.zeros_like(p)

    # Set model to initial weights
    model.train()
    with torch.no_grad():
        flat_idx = 0
        for p in model.parameters():
            numel = p.numel()
            p.data = torch.tensor(initial_weights[flat_idx:flat_idx+numel], 
                                dtype=torch.float32).reshape(p.shape).to(device)
            flat_idx += numel
    
    # Get current shortcut weight (assuming it's the last weight in fc1)
    # For toy_nn: fc1 has shape (hidden_dim, input_dim), shortcut is last column
    current_shortcut_weight = None
    shortcut_param_idx = None
    
    for name, param in model.named_parameters():
        if 'fc1.weight' in name:  # First layer weights
            current_shortcut_weight = param.data[:, -1].clone()  # Last column (shortcut)
            break
    
    if current_shortcut_weight is None:
        raise ValueError("Could not find shortcut weights in model")
    
    # Compute optimal gradient (pointing toward 0)
    optimal_shortcut_gradient = -current_shortcut_weight  # Points from current to 0
    
    # Compute labeled gradients
    model.zero_grad()
    loss_labeled = mse_logits(model, x_labeled, y_labeled)
    loss_labeled.backward(retain_graph=True)
    
    labeled_shortcut_grad = None
    for name, param in model.named_parameters():
        if 'fc1.weight' in name:
            labeled_shortcut_grad = param.grad[:, -1].clone()  # Shortcut gradients
            break
    
    # Compute unlabeled gradients
    model.zero_grad()
    pseudo_labels, x_confident = guess_pseudo_labels(model, x_unlabeled)
    
    if len(x_confident) > 0:
        loss_unlabeled = mse_probs(model, x_confident, pseudo_labels) * lambda_u
        loss_unlabeled.backward()
        
        unlabeled_shortcut_grad = None
        for name, param in model.named_parameters():
            if 'fc1.weight' in name:
                unlabeled_shortcut_grad = param.grad[:, -1].clone()
                break
    else:
        unlabeled_shortcut_grad = torch.zeros_like(optimal_shortcut_gradient)
    
    # Compute metrics
    def cosine_similarity(a, b):
        return torch.dot(a.flatten(), b.flatten()) / (torch.norm(a) * torch.norm(b) + 1e-8)
    
    # Cosine similarities with optimal direction
    labeled_cos_opt = cosine_similarity(labeled_shortcut_grad, optimal_shortcut_gradient).item()
    unlabeled_cos_opt = cosine_similarity(unlabeled_shortcut_grad, optimal_shortcut_gradient).item()
    
    # Cosine similarity between labeled and unlabeled
    labeled_unlabeled_cos = cosine_similarity(labeled_shortcut_grad, unlabeled_shortcut_grad).item()
    
    # Magnitude analysis
    labeled_magnitude = torch.norm(labeled_shortcut_grad).item()
    unlabeled_magnitude = torch.norm(unlabeled_shortcut_grad).item()
    optimal_magnitude = torch.norm(optimal_shortcut_gradient).item()
    
    return {
        'current_shortcut_weight': current_shortcut_weight.cpu().numpy(),
        'optimal_shortcut_gradient': optimal_shortcut_gradient.cpu().numpy(),
        'labeled_shortcut_grad': labeled_shortcut_grad.cpu().numpy(),
        'unlabeled_shortcut_grad': unlabeled_shortcut_grad.cpu().numpy(),
        'labeled_cos_optimal': labeled_cos_opt,
        'unlabeled_cos_optimal': unlabeled_cos_opt,
        'labeled_unlabeled_cos': labeled_unlabeled_cos,
        'labeled_magnitude': labeled_magnitude,
        'unlabeled_magnitude': unlabeled_magnitude,
        'optimal_magnitude': optimal_magnitude,
        'n_confident_samples': len(x_confident)
    }


def plot_shortcut_gradient_analysis(analysis_results, figsize=(15, 10)):
    """
    Create comprehensive visualization of shortcut gradient analysis.
    Focus on how gradients push shortcut weights toward/away from optimal (0).
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
    
    # Extract results
    current_weight = analysis_results['current_shortcut_weight']
    labeled_grad = analysis_results['labeled_shortcut_grad']
    unlabeled_grad = analysis_results['unlabeled_shortcut_grad']
    
    hidden_dim = len(current_weight)
    neurons = np.arange(hidden_dim)
    
    # 1. Gradient directions per neuron (positive = push away from 0, negative = push toward 0)
    width = 0.35
    ax1.bar(neurons - width/2, labeled_grad, width, label='Labeled Gradient', 
            color='#2878b5', alpha=0.7)
    ax1.bar(neurons + width/2, unlabeled_grad, width, label='Unlabeled Gradient', 
            color='#c82423', alpha=0.7)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax1.set_xlabel('Hidden Neuron')
    ax1.set_ylabel('Gradient Component')
    ax1.set_title('Shortcut Gradients by Neuron\n(Negative = Push toward 0 ✓, Positive = Push away from 0 ✗)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Effect on shortcut weights (how much each gradient helps reduce shortcut dependence)
    # Negative gradient components help reduce shortcut weights (good)
    # Positive gradient components increase shortcut weights (bad)
    
    # Count helpful vs harmful gradient components
    labeled_helpful = np.sum(labeled_grad < 0)
    labeled_harmful = np.sum(labeled_grad > 0)
    unlabeled_helpful = np.sum(unlabeled_grad < 0) 
    unlabeled_harmful = np.sum(unlabeled_grad > 0)
    
    categories = ['Labeled\n(Helpful)', 'Labeled\n(Harmful)', 'Unlabeled\n(Helpful)', 'Unlabeled\n(Harmful)']
    values = [labeled_helpful, labeled_harmful, unlabeled_helpful, unlabeled_harmful]
    colors = ['#2878b5', '#ff7f7f', '#c82423', '#ffb3b3']
    
    bars = ax2.bar(categories, values, color=colors, alpha=0.7)
    ax2.set_ylabel('Number of Neurons')
    ax2.set_title('Gradient Effect on Shortcut Mitigation\n(Helpful = Push toward 0, Harmful = Push away from 0)')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{int(val)}', ha='center', va='bottom', fontweight='bold')
    
    # 3. Gradient conflict analysis (do labeled and unlabeled gradients oppose each other?)
    cos_conflict = analysis_results['labeled_unlabeled_cos']
    
    # Create a single bar showing the conflict level
    conflict_level = abs(cos_conflict)  # How much they conflict (0 = orthogonal, 1 = aligned/opposed)
    conflict_type = "Opposing" if cos_conflict < -0.1 else ("Aligned" if cos_conflict > 0.1 else "Orthogonal")
    
    color = '#ff4444' if cos_conflict < -0.1 else ('#44ff44' if cos_conflict > 0.1 else '#ffaa44')
    
    bar = ax3.bar(['Labeled vs Unlabeled\nGradient Relationship'], [cos_conflict], 
                  color=color, alpha=0.7, width=0.6)
    ax3.set_ylabel('Cosine Similarity')
    ax3.set_title('Gradient Conflict Analysis\n(+1=Same direction, -1=Opposite, 0=Orthogonal)')
    ax3.set_ylim(-1, 1)
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax3.axhline(y=0.5, color='green', linestyle='--', alpha=0.5, label='Strong Alignment')
    ax3.axhline(y=-0.5, color='red', linestyle='--', alpha=0.5, label='Strong Opposition')
    ax3.grid(True, alpha=0.3)
    
    # Add interpretation text
    ax3.text(0, cos_conflict + 0.1*np.sign(cos_conflict), 
             f'{conflict_type}\n(cos={cos_conflict:.3f})',
             ha='center', va='bottom' if cos_conflict > 0 else 'top',
             fontweight='bold', fontsize=11,
             bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    # 4. Overall shortcut mitigation effectiveness
    # Calculate how much each gradient type helps with shortcut mitigation
    labeled_mitigation_score = -np.mean(labeled_grad)  # Negative gradient is good
    unlabeled_mitigation_score = -np.mean(unlabeled_grad)  # Negative gradient is good
    
    scores = [labeled_mitigation_score, unlabeled_mitigation_score]
    labels = ['Labeled Data', 'Unlabeled Data']
    colors = ['#2878b5', '#c82423']
    
    # Color bars based on effectiveness (green = helpful, red = harmful)
    bar_colors = []
    for score in scores:
        if score > 0.01:
            bar_colors.append('#44aa44')  # Green for helpful
        elif score < -0.01:
            bar_colors.append('#aa4444')  # Red for harmful  
        else:
            bar_colors.append('#aaaa44')  # Yellow for neutral
    
    bars = ax4.bar(labels, scores, color=bar_colors, alpha=0.7)
    ax4.set_ylabel('Shortcut Mitigation Score')
    ax4.set_title('Shortcut Mitigation Effectiveness\n(Positive = Helps reduce shortcuts, Negative = Increases shortcuts)')
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax4.grid(True, alpha=0.3)
    
    # Add value labels and interpretation
    for bar, val, label in zip(bars, scores, labels):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.005*np.sign(height),
                f'{val:.4f}', ha='center', va='bottom' if height > 0 else 'top',
                fontweight='bold')
        
        # Add effectiveness label
        effectiveness = "Helps ✓" if val > 0.01 else ("Hurts ✗" if val < -0.01 else "Neutral ~")
        ax4.text(bar.get_x() + bar.get_width()/2., height/2,
                effectiveness, ha='center', va='center',
                fontweight='bold', fontsize=10, color='white')
    
    plt.tight_layout()
    
    # Add improved summary text
    labeled_mitigation = -np.mean(analysis_results['labeled_shortcut_grad'])
    unlabeled_mitigation = -np.mean(analysis_results['unlabeled_shortcut_grad'])
    conflict = analysis_results['labeled_unlabeled_cos']
    
    summary_text = (
        f"Shortcut Gradient Analysis Summary:\n"
        f"• Labeled data mitigation score: {labeled_mitigation:.4f} ({'Helpful' if labeled_mitigation > 0 else 'Harmful'})\n"
        f"• Unlabeled data mitigation score: {unlabeled_mitigation:.4f} ({'Helpful' if unlabeled_mitigation > 0 else 'Harmful'})\n"
        f"• Gradient conflict level: {conflict:.3f} ({'High Opposition' if conflict < -0.5 else ('High Alignment' if conflict > 0.5 else 'Moderate')})\n"
        f"• Confident unlabeled samples: {analysis_results['n_confident_samples']}\n"
        f"• Key insight: {'Unlabeled data helps shortcut mitigation' if unlabeled_mitigation > 0.01 else ('Unlabeled data hurts shortcut mitigation' if unlabeled_mitigation < -0.01 else 'Unlabeled data has neutral effect')}"
    )
    
    fig.text(0.02, 0.02, summary_text, fontsize=9, 
             bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow', alpha=0.9))
    
    return fig