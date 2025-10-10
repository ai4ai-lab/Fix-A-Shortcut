from experiment import synthetic_experiments
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

from data import generate_synthetic_base, generate_training_test_data

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


def plot_weights_gradient_methods(
    results: dict,
    true_weights: np.ndarray,
    ax=None,
    hidden_dim: int = 5
):
    """
    Bar‐plot of the final learned input‐layer weights vs. the ground‐truth generative weights.

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

    # Learned weights (ℓ₂‐norm of each column of the fc1 weight matrix)
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
    ax.set_ylabel('Input‐layer weight norm', fontsize=12)
    ax.set_title('Final Shortcut‐Input Weight Norms', fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True)
    return ax


def plot_shortcut_trajectory(
    results: dict,
    ax=None,
    hidden_dim: int = 5
):
    """
    Line plot of the shortcut input‐layer weight norm across epochs.

    Parameters
    ----------
    results : dict
        Output of synthetic_experiments(), with key 'trajectories':
        dict mapping method → ndarray (epochs × param_dim)
    ax : matplotlib.axes.Axes, optional
        Axes on which to draw the plot. If None, a new figure is created.
    hidden_dim : int
        Number of units in the first hidden layer of toy_nn.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    methods = ['naive', 'fix_step', 'grad_surgery', 'fix_soft']

    for name in methods:
        traj = results['trajectories'][name]
        # infer input_dim from total params
        total_params = traj.shape[1]
        # assuming only fc1 uses hidden_dim*input_dim
        input_dim = total_params // (hidden_dim + 0)
        fc1_size = hidden_dim * input_dim
        j = input_dim - 1  # shortcut column
        series = [
            np.linalg.norm(epoch[:fc1_size].reshape(
                hidden_dim, input_dim)[:, j])
            for epoch in traj
        ]
        ax.plot(series, label=get_display_name(name),
                color=get_method_color(name), linewidth=2)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Shortcut weight norm', fontsize=12)
    ax.set_title('Shortcut‐Feature Norm Over Training', fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True)


def plot_feature_weight_correlation_heatmap(
    results: dict,
    method: str,
    input_dim: int = 5,
    hidden_dim: int = 5,
    ax=None
):
    """
    Heatmap of Pearson correlations between learned first-layer weights for each
    pair of features (C1, C2, U1, U2, S) for a specific training method.

    Parameters
    ----------
    results : dict
        Output of synthetic_experiments(), with key 'final_weights'.
    method : str
        One of ['naive', 'fix_step', 'grad_surgery', 'fix_soft'].
    input_dim : int
        Number of input features (e.g. 5 = C1, C2, U1, U2, S).
    hidden_dim : int
        Number of units in the first hidden layer.
    ax : matplotlib.axes.Axes, optional
        Axes to draw the heatmap on. If None, a new figure is created.
    """
    # Extract and reshape the fc1 weight matrix
    flat = results['final_weights'][method]
    fc1 = flat[: hidden_dim * input_dim].reshape(hidden_dim, input_dim)

    # Compute the correlation matrix across feature columns
    corr = np.corrcoef(fc1.T)  # shape (input_dim, input_dim)

    feature_names = ['C1', 'C2', 'U1', 'U2', 'S']
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        xticklabels=feature_names,
        yticklabels=feature_names,
        cbar_kws={"label": "Pearson corr(feature_i, feature_j)"}
    )
    ax.set_title(f"Feature‐Weight Correlation Heatmap ({get_display_name(method)})", fontsize=14)
    ax.set_xlabel("Feature", fontsize=12)
    ax.set_ylabel("Feature", fontsize=12)
    ax.figure.tight_layout()


def plot_final_shortcut_norm_vs_lambda(
    results_by_lambda: dict,
    methods: list = ['naive', 'fix_step', 'grad_surgery', 'fix_soft'],
    input_dim: int = 5,
    hidden_dim: int = 5,
    ax=None
):
    """
    Line plot of the final shortcut input-layer weight norm vs unlabeled-loss weight λᵤ,
    overlaying multiple gradient methods.

    Parameters
    ----------
    results_by_lambda : dict
        Mapping λᵤ → results dict from synthetic_experiments.
    methods : list of str
        Which methods to plot.
    input_dim : int
        Number of input features.
    hidden_dim : int
        Number of hidden units in the first layer.
    ax : matplotlib.axes.Axes, optional
        Axes on which to draw the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    lambdas = sorted(results_by_lambda.keys())
    linestyles = {'naive': '-', 'fix_step': '--',
                  'grad_surgery': ':', 'fix_soft': '-.'}

    for name in methods:
        norms = []
        for lam in lambdas:
            flat = results_by_lambda[lam]['final_weights'][name]
            fc1 = flat[:hidden_dim*input_dim].reshape(hidden_dim, input_dim)
            norms.append(np.linalg.norm(fc1[:, -1]))
        ax.plot(
            lambdas,
            norms,
            label=get_display_name(name),
            color=get_method_color(name),
            linestyle=linestyles.get(name, '-'),
            marker='o'
        )

    ax.set_xscale('log')
    ax.set_xlabel('λ_u (log scale)', fontsize=12)
    ax.set_ylabel('Final shortcut norm', fontsize=12)
    ax.set_title('Shortcut Norm vs λ_u (All Methods)', fontsize=14)
    ax.legend(title='Method')
    ax.grid(True, which='both', ls='--')


def plot_shortcut_norm_vs_sigma_y(
    sigma_y_list: list,
    fixed_cfg: dict,
    methods: list = ['naive', 'fix_step', 'grad_surgery', 'fix_soft'],
    input_dim: int = 5,
    hidden_dim: int = 5,
    ax=None
):
    """
    Plot final shortcut norm vs. label noise sigma_y under 'y_noise' mode.

    sigma_y_list: list of float label noise values
    fixed_cfg: dict of other synthetic_experiments kwargs (except sigma_y, shortcut_mode)
    methods: which methods to plot
    """
    from train_synthetic import synthetic_experiments

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    for name in methods:
        norms = []
        for sig in sigma_y_list:
            cfg = dict(fixed_cfg)
            cfg['sigma_y'] = sig
            cfg['shortcut_mode'] = 'y_noise'
            results = synthetic_experiments(**cfg)
            flat = results['final_weights'][name]
            fc1 = flat[:hidden_dim*input_dim].reshape(hidden_dim, input_dim)
            norms.append(np.linalg.norm(fc1[:, -1]))
        ax.plot(sigma_y_list, norms, marker='o', label=get_display_name(name))

    ax.set_xlabel('σ_y (label noise)', fontsize=12)
    ax.set_ylabel('Final shortcut norm', fontsize=12)
    ax.set_title('Shortcut Norm vs Label Noise (y_noise)', fontsize=14)
    ax.legend(title='Method')
    ax.grid(True)


def plot_shortcut_norm_vs_rho_cu(
    rho_list: list,
    fixed_cfg: dict,
    mode: str,
    methods: list = ['naive', 'fix_step', 'grad_surgery', 'fix_soft'],
    input_dim: int = 5,
    hidden_dim: int = 5,
    ax=None
):
    """
    Plot final shortcut norm vs. correlation rho_cu for a given shortcut mode.

    rho_list: list of float c1-u1 correlation values
    fixed_cfg: dict of other synthetic_experiments kwargs (except rho_cu, shortcut_mode)
    mode: one of 'with_c' or 'c_and_u'
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    for name in methods:
        norms = []
        for r in rho_list:
            cfg = dict(fixed_cfg)
            cfg['rho_cu'] = r
            cfg['shortcut_mode'] = mode
            results = synthetic_experiments(**cfg)
            flat = results['final_weights'][name]
            fc1 = flat[:hidden_dim*input_dim].reshape(hidden_dim, input_dim)
            norms.append(np.linalg.norm(fc1[:, -1]))
        ax.plot(rho_list, norms, marker='o', label=get_display_name(name))

    ax.set_xlabel('ρ_cu (C1–U1 correlation)', fontsize=12)
    ax.set_ylabel('Final shortcut norm', fontsize=12)
    ax.set_title(f'Shortcut Norm vs ρ_cu ({mode})', fontsize=14)
    ax.legend(title='Method')
    ax.grid(True)


# sweep beta (data alignment)
def plot_shortcut_norm_vs_beta(
    beta_list: list,
    fixed_cfg: dict,
    methods: list = ['naive', 'fix_step', 'grad_surgery', 'fix_soft'],
    input_dim: int = 5,
    hidden_dim: int = 5,
    ax=None
):
    """
    Plot final shortcut norm vs. data-generator beta (alignment S to Y/C).
    """
    from experiment import synthetic_experiments
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    for name in methods:
        norms = []
        for b in beta_list:
            cfg = fixed_cfg.copy()
            cfg['beta'] = b
            results = synthetic_experiments(**cfg)
            flat = results['final_weights'][name]
            fc1 = flat[:hidden_dim*input_dim].reshape(hidden_dim, input_dim)
            norms.append(np.linalg.norm(fc1[:, -1]))
        ax.plot(beta_list, norms, marker='o', label=get_display_name(name))
    ax.set_xlabel('Beta (data alignment)')
    ax.set_ylabel('Final shortcut norm')
    ax.set_title('Shortcut Norm vs Beta')
    ax.legend(title='Method')
    ax.grid(True)

# sweep delta_c for c_and_u mode


def plot_shortcut_norm_vs_delta_c(
    delta_c_list: list,
    fixed_cfg: dict,
    delta_u: float = 1.0,
    methods: list = ['naive', 'fix_step', 'grad_surgery', 'fix_soft'],
    input_dim: int = 5,
    hidden_dim: int = 5,
    ax=None
):
    """
    Plot final shortcut norm vs. delta_c in S = delta_c*C + delta_u*U.
    """
    from experiment import synthetic_experiments
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    for name in methods:
        norms = []
        for dc in delta_c_list:
            cfg = fixed_cfg.copy()
            cfg['shortcut_mode'] = 'c_and_u'
            cfg['delta_c'] = dc
            cfg['delta_u'] = delta_u
            results = synthetic_experiments(**cfg)
            flat = results['final_weights'][name]
            fc1 = flat[:hidden_dim*input_dim].reshape(hidden_dim, input_dim)
            norms.append(np.linalg.norm(fc1[:, -1]))
        ax.plot(delta_c_list, norms, marker='o', label=get_display_name(name))
    ax.set_xlabel('Delta_c')
    ax.set_ylabel('Final shortcut norm')
    ax.set_title('Shortcut Norm vs Delta_c')
    ax.legend(title='Method')
    ax.grid(True)

# sweep delta_u


def plot_shortcut_norm_vs_delta_u(
    delta_u_list: list,
    fixed_cfg: dict,
    delta_c: float = 1.0,
    methods: list = ['naive', 'fix_step', 'grad_surgery', 'fix_soft'],
    input_dim: int = 5,
    hidden_dim: int = 5,
    ax=None
):
    """
    Plot final shortcut norm vs. delta_u in S = delta_c*C + delta_u*U.
    """
    from experiment import synthetic_experiments
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    for name in methods:
        norms = []
        for du in delta_u_list:
            cfg = fixed_cfg.copy()
            cfg['shortcut_mode'] = 'c_and_u'
            cfg['delta_c'] = delta_c
            cfg['delta_u'] = du
            results = synthetic_experiments(**cfg)
            flat = results['final_weights'][name]
            fc1 = flat[:hidden_dim*input_dim].reshape(hidden_dim, input_dim)
            norms.append(np.linalg.norm(fc1[:, -1]))
        ax.plot(delta_u_list, norms, marker='o', label=get_display_name(name))
    ax.set_xlabel('Delta_u')
    ax.set_ylabel('Final shortcut norm')
    ax.set_title('Shortcut Norm vs Delta_u')
    ax.legend(title='Method')
    ax.grid(True)
# helper to extract final shortcut‐weight norm


def _final_shortcut_norm(results, method, input_dim=5, hidden_dim=5):
    flat = results['final_weights'][method]
    fc1 = flat[:hidden_dim*input_dim].reshape(hidden_dim, input_dim)
    # shortcut is last column
    return np.linalg.norm(fc1[:, -1])


def plot_shortcut_norm_vs_beta(
    beta_list: list[float],
    fixed_cfg: dict,
    methods: list[str] = ['naive', 'fix_step', 'grad_surgery', 'fix_soft'],
    input_dim: int = 5,
    hidden_dim: int = 5,
    ax=None
):
    """
    Sweep the shortcut‐alignment strength β (for y_noise, with_c, with_u, c_and_u).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    for m in methods:
        norms = []
        for b in beta_list:
            cfg = fixed_cfg.copy()
            cfg['beta'] = b
            # also override beta_y if y_noise
            cfg['beta_y'] = b
            results = synthetic_experiments(**cfg)
            norms.append(_final_shortcut_norm(
                results, m, input_dim, hidden_dim))
        ax.plot(beta_list, norms, marker='o', label=get_display_name(m))
    ax.set_xlabel('β (alignment strength)')
    ax.set_ylabel('Final shortcut norm')
    ax.set_title('Shortcut Norm vs β')
    ax.legend()
    ax.grid(True)
    return ax


def plot_shortcut_norm_vs_delta_c(
    delta_c_list: list[float],
    fixed_cfg: dict,
    delta_u: float = None,
    methods: list[str] = ['naive', 'fix_step', 'grad_surgery', 'fix_soft'],
    input_dim: int = 5,
    hidden_dim: int = 5,
    ax=None
):
    """
    Sweep δ_c in the c_and_u mode.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    for m in methods:
        norms = []
        for dc in delta_c_list:
            cfg = fixed_cfg.copy()
            cfg['shortcut_mode'] = 'c_and_u'
            cfg['delta_c'] = dc
            cfg['delta_u'] = delta_u if delta_u is not None else cfg.get(
                'beta', 1.0)
            results = synthetic_experiments(**cfg)
            norms.append(_final_shortcut_norm(
                results, m, input_dim, hidden_dim))
        ax.plot(delta_c_list, norms, marker='o', label=get_display_name(m))
    ax.set_xlabel('δ_c')
    ax.set_ylabel('Final shortcut norm')
    ax.set_title('Shortcut Norm vs δ_c')
    ax.legend()
    ax.grid(True)
    return ax


def plot_shortcut_norm_vs_delta_u(
    delta_u_list: list[float],
    fixed_cfg: dict,
    delta_c: float = None,
    methods: list[str] = ['naive', 'fix_step', 'grad_surgery', 'fix_soft'],
    input_dim: int = 5,
    hidden_dim: int = 5,
    ax=None
):
    """
    Sweep δ_u in the c_and_u mode.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    for m in methods:
        norms = []
        for du in delta_u_list:
            cfg = fixed_cfg.copy()
            cfg['shortcut_mode'] = 'c_and_u'
            cfg['delta_u'] = du
            cfg['delta_c'] = delta_c if delta_c is not None else cfg.get(
                'beta', 1.0)
            results = synthetic_experiments(**cfg)
            norms.append(_final_shortcut_norm(
                results, m, input_dim, hidden_dim))
        ax.plot(delta_u_list, norms, marker='o', label=get_display_name(m))
    ax.set_xlabel('δ_u')
    ax.set_ylabel('Final shortcut norm')
    ax.set_title('Shortcut Norm vs δ_u')
    ax.legend()
    ax.grid(True)
    return ax


