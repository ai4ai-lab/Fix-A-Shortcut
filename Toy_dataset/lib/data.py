# data.py  – revised
import numpy as np
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------
# 1.  Base-concept generator
# ---------------------------------------------------------------------


def generate_synthetic_base(
    num_samples: int = 2_000,
    *,
    seed: int | None = None,
    rho_cu: float = 0.0,
    sigma_y: float = 0.1
):
    """
    Draw (C1, C2, U1, U2) ~ N(0, Σ_base) with Corr(C1,U1)=rho_cu.
    Return X_base, noisy label y, and clean label y_clean.

    y_clean = 4·C1 – 0.5·C2 + 1·U1 + 2·U2
    y       = y_clean + ε,    ε ~ N(0, σ_y²)
    """
    if seed is not None:
        np.random.seed(seed)

    # 1) sample raw independent Gaussians
    c1, c2, u1, u2 = np.random.normal(size=(4, num_samples))

    # 2) enforce Corr(C1, U1) ≈ rho_cu
    def correlate(x, target_rho):
        x_std = (x - x.mean()) / x.std()
        noise = np.random.normal(size=x.shape)
        z = target_rho * x_std + np.sqrt(max(0, 1 - target_rho**2)) * noise
        return (z - z.mean()) / z.std()
    u1 = correlate(c1, rho_cu)

    # 3) labels
    y_clean = 4*c1 - 0.5*c2 + 1*u1 + 2*u2
    y = y_clean + np.random.normal(0, sigma_y, size=num_samples)

    # 4) assemble
    X_base = np.vstack([c1, c2, u1, u2]).T
    return X_base, y, y_clean


# ---------------------------------------------------------------------
# 2.  Train / test split  +  shortcut injection
# ---------------------------------------------------------------------
def generate_training_test_data(
    X_base: np.ndarray,
    y_all: np.ndarray,
    y_clean: np.ndarray,
    *,
    shortcut_mode: str = "y_noise",
    beta: float = 1.0,
    beta_y: float | None = None,
    delta_c: float | None = None,
    delta_u: float | None = None,
    sigma_s: float = 0.1,
    shortcut_ratio: float = 1.0,
    test_size: float = 0.5,
    random_state: int | None = None
):
    """
    Split into train / test, then append a 5-th shortcut feature S.

    Modes:
      y_noise : S = beta_y * y_clean + σ_s·ξ
      with_c  : S = beta   * C1      + σ_s·ξ
      with_u  : S = beta   * U1      + σ_s·ξ
      c_and_u : S = δ_c C1 + δ_u U1  + σ_s·ξ
                (defaults δ_c=δ_u=beta if unspecified)

    Only a fraction `shortcut_ratio` of the train set gets aligned S;
    the rest—and all test points—get pure N(0,1) noise.
    """
    beta_y = beta_y if beta_y is not None else beta

    # 1) train/test split
    idx = np.arange(len(X_base))
    train_idx, test_idx = train_test_split(
        idx, test_size=test_size, random_state=random_state
    )
    Xtr, Xte = X_base[train_idx], X_base[test_idx]
    ytr, yte = y_all[train_idx], y_all[test_idx]
    y_clean_tr = y_clean[train_idx]

    # 2) build aligned S on first `n_aligned` samples
    n_train = len(Xtr)
    n_aligned = int(shortcut_ratio * n_train)
    if shortcut_mode == "y_noise":
        aligned = beta_y * y_clean_tr[:n_aligned]
    elif shortcut_mode == "with_c":
        aligned = beta * Xtr[:n_aligned, 0]   # C1
    elif shortcut_mode == "with_u":
        aligned = beta * Xtr[:n_aligned, 2]   # U1
    elif shortcut_mode == "c_and_u":
        dc = delta_c if delta_c is not None else beta
        du = delta_u if delta_u is not None else beta
        aligned = dc * Xtr[:n_aligned, 0] + du * Xtr[:n_aligned, 2]
    else:
        raise ValueError(f"Unknown shortcut_mode: {shortcut_mode!r}")

    s_aligned = aligned + sigma_s * np.random.normal(size=n_aligned)
    s_noise = np.random.normal(size=n_train - n_aligned)
    S_train = np.concatenate([s_aligned, s_noise])
    np.random.shuffle(S_train)  # hide who was aligned

    # 3) test S = pure noise
    S_test = np.random.normal(size=len(Xte))

    # 4) append
    x_train = np.hstack([Xtr, S_train.reshape(-1, 1)])
    x_test = np.hstack([Xte, S_test.reshape(-1, 1)])
    return x_train, ytr, x_test, yte
