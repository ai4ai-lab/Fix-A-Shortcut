"""
plot_functions.py - Essential plotting functions for Fix-A-Shortcut

This module contains only the minimal plotting functions needed to reproduce
the core results from the Fix-A-Shortcut paper. Extracted from plots.py and 
plot_test.py for conference submission.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from model import toy_nn


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


# ═══════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def _set_flattened_params(model: torch.nn.Module, flat: np.ndarray) -> None:
    """Load a flattened NumPy parameter vector back into a model."""
    pointer = 0
    for p in model.parameters():
        n = p.numel()
        chunk = torch.from_numpy(flat[pointer:pointer + n]).view(p.shape)
        p.data.copy_(chunk.to(p.device))
        pointer += n


# ═══════════════════════════════════════════════════════════════════
# CORE PLOTTING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def plot_weights_gradient_methods(
    results: dict,
    true_weights: np.ndarray,
    ax=None,
    hidden_dim: int = 5
):
    """
    Bar plot of the final learned input-layer weights vs. the ground-truth generative weights.

    Parameters
    ----------
    results : dict
        Output of synthetic_experiments(), with key 'final_weights':
        dict mapping method → flattened parameter vector.
    true_weights : np.ndarray
        1D array of length input_dim containing the true generative weights.
    ax : matplotlib.axes.Axes, optional
        Axes on which to draw the plot. If None, a new figure is created.
    hidden_dim : int
        Number of units in the first hidden layer of toy_nn.
    """
    input_dim = len(true_weights)
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    methods = ['naive', 'fix_step', 'grad_surgery', 'fix_soft']

    bar_width = 0.15
    idx = np.arange(input_dim)
    start = idx - bar_width * (len(methods) + 1) / 2 + bar_width

    # True weights (black)
    ax.bar(start, true_weights, bar_width, label='True', color='black')

    # Learned weights (ℓ₂-norm of each column of the fc1 weight matrix)
    for i, name in enumerate(methods):
        flat = results['final_weights'][name]
        # fc1 weights occupy first hidden_dim*input_dim entries
        fc1 = flat[: hidden_dim * input_dim].reshape(hidden_dim, input_dim)
        learned_col_norms = np.linalg.norm(fc1, axis=0)  # shape (input_dim,)
        ax.bar(
            start + (i + 1) * bar_width,
            learned_col_norms,
            bar_width,
            label=get_display_name(name),
            color=get_method_color(name)
        )

    ax.set_xticks(idx)
    ax.set_xticklabels(['C1', 'C2', 'U1', 'U2', 'S'], fontsize=10)
    ax.set_ylabel('Input-layer weight norm', fontsize=12)
    ax.set_title('Final Shortcut-Input Weight Norms', fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True)
    return ax


def plot_shortcut_weight_trajectory(
    results: dict,
    methods=None,
    hidden_dim: int = 5,
    input_dim: int = 5,
    ax=None
):
    """
    Line plot of the shortcut input-layer weight norm across epochs.

    Parameters
    ----------
    results : dict
        Output of synthetic_experiments() with 'trajectories' key
    methods : list, optional
        Methods to plot. Defaults to ['naive', 'fix_step', 'grad_surgery', 'fix_soft']
    hidden_dim : int
        Number of hidden units in first layer
    input_dim : int  
        Number of input features
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
    """
    if methods is None:
        methods = ['naive', 'fix_step', 'grad_surgery', 'fix_soft']

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    fc1_size = hidden_dim * input_dim
    for name in methods:
        traj = results['trajectories'][name]          # [epoch × param_dim]
        # Extract shortcut weight norms (last column of fc1 weights)
        norms = [np.linalg.norm(e[:fc1_size]
                                .reshape(hidden_dim, input_dim)[:, -1])
                 for e in traj]
        ax.plot(range(1, len(norms)+1), norms,
                label=get_display_name(name),
                color=get_method_color(name))

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Shortcut weight ‖·‖')
    ax.set_title('Shortcut weight trajectory')
    ax.legend()
    ax.grid(True)
    return ax


def compute_generalization_gap(
    results, x_train, y_train, x_test, y_test, device="cpu"
):
    """
    Compute train/test MSE and generalization gap for each method.

    Parameters
    ----------
    results : dict
        Output of synthetic_experiments()
    x_train, y_train : array-like
        Training data
    x_test, y_test : array-like  
        Test data
    device : str
        Device for computation

    Returns
    -------
    dict
        Method -> {'train_mse', 'test_mse', 'gap'}
    """
    device = torch.device(device)
    inputdim = x_train.shape[1]
    methods = ['naive', 'fix_step', 'grad_surgery', 'fix_soft']

    # Convert to tensors
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
        gaps[m] = {
            "train_mse": mse_tr,
            "test_mse": mse_te,
            "gap": mse_te - mse_tr
        }
    return gaps


def plot_generalization_gap(gaps, ax=None):
    """
    Bar plot of generalization gaps (test_mse - train_mse) for each method.

    Parameters
    ----------
    gaps : dict
        Output of compute_generalization_gap()
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
    """
    methods = list(gaps.keys())
    gapvals = [gaps[m]['gap'] for m in methods]
    # Use display names for x-axis labels
    display_names = [get_display_name(m) for m in methods]
    palette = [get_method_color(m) for m in methods]

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    ax.bar(display_names, gapvals, color=palette)
    ax.set_ylabel(
        r'$\mathrm{MSE}_{\mathrm{test}} - \mathrm{MSE}_{\mathrm{train}}$')
    ax.set_title('Generalization gap')
    ax.grid(True, axis='y', ls='--')
    return ax
