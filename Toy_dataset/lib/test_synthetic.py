#!/usr/bin/env python
# test_synthetic.py

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from data import generate_synthetic_base, generate_training_test_data
from experiment import synthetic_experiments
from model import toy_nn
from gradients import train_gt_5d
from plot_test import (
    compute_generalization_gap,
    plot_generalization_gap,
    plot_test_shortcut_weight_over_epochs,
    plot_treatment_effect_trajectory,
    #plot_contour_trajectories,
    #plot_pca_contour_with_trajectories,
    compute_shortcut_ratio,
    compute_metric_grid_auto,
    #compute_metric_in_pca_space,
    plot_contour_trajectories_as_arrows_skip,
    plot_test_mse_over_epochs,
    #compute_causal_error_over_epochs, plot_causal_error,
    #plot_total_causal_error_over_epochs
    plot_causal_weight_error_over_epochs,
    plot_subspaces_2d,
    plot_subspaces_3d,
    plot_subspace_analysis_combined
)
from plots import (
    plot_weights_gradient_methods,
    plot_feature_weight_correlation_heatmap,
    get_display_name)


def main():
    base_cfg = dict(
        num_samples=2000,
        rho_cu=0.2,
        sigma_y=0.1,
        shortcut_mode="c_and_u",
        beta=-0.8, 
        beta_y=-0.9,
        delta_c=0.9,
        delta_u=-0.2,
        sigma_s=0.1,
        shortcut_ratio=0.8,
        test_size=0.5,
        random_state=42,
        n_labeled=75,
        epochs=10000,
        lr=1e-3,
        lambda_u=10.0,
        device="cpu",
        to_print=False,
    )

    # 1) Generate data & split
    X_base, y_all, y_clean = generate_synthetic_base(
        num_samples=base_cfg['num_samples'],
        seed=base_cfg['random_state'],
        rho_cu=base_cfg['rho_cu'],
        sigma_y=base_cfg['sigma_y']
    )
    x_train, y_train, x_test, y_test = generate_training_test_data(
        X_base, y_all, y_clean,
        shortcut_mode=base_cfg['shortcut_mode'],
        beta=base_cfg['beta'],
        beta_y=base_cfg['beta_y'],
        delta_c=base_cfg['delta_c'],
        delta_u=base_cfg['delta_u'],
        sigma_s=base_cfg['sigma_s'],
        shortcut_ratio=base_cfg['shortcut_ratio'],
        test_size=base_cfg['test_size'],
        random_state=base_cfg['random_state']
    )

    # 2) Run all four methods
    results = synthetic_experiments(**base_cfg)
        # ----------------------------------------------------------
    # 1) Single example run + bar‐plot of final weights
   # ────────────────────────────────────────────────────────────────
#  extra diagnostics: conflicts & α / projection distributions
# ────────────────────────────────────────────────────────────────
    import seaborn as sns   # already available if you used it elsewhere

    stats = results["stats"]                # ← comes from synthetic_experiments

    # 1) Fix-Step: total number of gU conflicts that were discarded
    fixstep_conflicts = stats["fix_step"]["conflict_count"]
    print(f"[Fix-Step]  #gradient conflicts discarded: {fixstep_conflicts}")

    # 2) Fix-Soft: α-values (cosine-based weights)
    alphas = stats["fix_soft"]["alpha_list"]
    conf_soft = stats["fix_soft"]["conflict_count"]
    print(f"[Fix-Soft] #conflicts       : {conf_soft}")
    print(f"[Fix-Soft] α stats          : "
        f"min={np.min(alphas):.3f},  median={np.median(alphas):.3f}, "
        f"max={np.max(alphas):.3f},  mean={np.mean(alphas):.3f}")

    fig, ax = plt.subplots(figsize=(5, 3))
    sns.histplot(alphas, bins=30, kde=True, ax=ax, color="#c82423")
    ax.set_xlabel("α (Fix-Soft weight)")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of α – Fix-Soft")
    plt.tight_layout()
    plt.savefig("alpha_distribution_fix_soft.png", dpi=300)
    plt.close(fig)

    # 3) Grad-Surgery: projection scaling factors ‖gU‖·|cosθ| / ‖gL‖
    

    # convert to NumPy so we can call .min()/.max()
    # Convert to array
    ps = np.array(stats["grad_surgery"]["proj_scale_list"])
    true_w = np.array([4, -0.5, 1, 2, 0])  # C1, C2, U1, U2, S

    # Compute axis limits
    min_x = ps.min()
    max_x = max(ps.max(), 1.0)

    print(f"[Grad-Surgery] #conflicts    : {stats['grad_surgery']['conflict_count']}")
    print(f"[Grad-Surgery] proj-scale stats: "
        f"min={ps.min():.3f},  median={np.median(ps):.3f}, "
        f"max={ps.max():.3f},  mean={ps.mean():.3f}")

    fig, ax = plt.subplots(figsize=(5, 3))

    # Histogram over [min_x, max_x]
    bins = np.linspace(min_x, max_x, 31)
    sns.histplot(ps, bins=bins, kde=True, ax=ax, color="#2878b5")

    # Vertical line at 1.0
    ax.axvline(
        1.0,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label="scale = 1.0",
        zorder=2
    )

    # Set limits
    ax.set_xlim(min_x, max_x)
    true_w = np.array([4, -0.5, 1, 2, 0])  # C1, C2, U1, U2, S

    # Force 1.0 as a tick
    # pick 6 evenly spaced ticks but ensure last tick is exactly 1.0
    ticks = np.linspace(min_x, max_x, 6)
    ticks[-1] = 1.0
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:.1f}" for t in ticks])

    ax.set_xlabel("Projection scale (Grad-Surgery)")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of projection scale — Grad-Surgery")
    ax.legend()
    plt.tight_layout()
    plt.savefig("proj_scale_distribution_grad_surgery.png", dpi=300)
    plt.close(fig)
        # test‐time: test MSE
   
        # ----------------------------------------------------------


    # ----------------------------------------------------------
  
    # 3) Plot generalization gap
    gaps = compute_generalization_gap(
        results, x_train, y_train, x_test, y_test, device=base_cfg['device']
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    plot_generalization_gap(gaps, ax=ax)
    plt.tight_layout()
    plt.savefig("generalization_gap.png")
    plt.close(fig)

    # 3b) Bar plot of final weights vs true weights
    fig, ax = plt.subplots(figsize=(6, 4))
    plot_weights_gradient_methods(
        results,
        true_w,
        ax=ax,
        hidden_dim=5
    )
    plt.tight_layout()
    plt.savefig("final_weights_comparison.png")
    plt.close(fig)

    # 4) Shortcut‐weight norm over epochs
    fig, ax = plt.subplots(figsize=(6, 4))
    plot_test_shortcut_weight_over_epochs(results=results, ax=ax)
    plt.tight_layout()
    plt.savefig("shortcut_weight_trajectory.png")
    plt.close(fig)

    # 5) Treatment‐effect trajectory
    fig, ax = plt.subplots(figsize=(6, 4))
    plot_treatment_effect_trajectory(
        results=results,
        hidden_dim=5,
        input_dim=x_train.shape[1],
        delta=1.0,
        device=base_cfg['device'],
        ax=ax
    )
    plt.tight_layout()
    plt.savefig("treatment_effect_trajectory.png")
    plt.close(fig)


    # ---------------------------------------------------------------
# 6) PCA + contour of train‐MSE + arrowed paths (using new functions)
# ---------------------------------------------------------------

    # ------------------------------------------------------------------
# 6)  PCA contour of shortcut-norm  +  optimisation paths (incl. GT)
# ------------------------------------------------------------------
    methods = ["naive", "fix_step", "grad_surgery", "fix_soft", "gt"]

    # 6a)  fit PCA on *all* trajectories
    all_traj   = np.vstack([results["trajectories"][m] for m in methods])
    pca        = PCA(n_components=2).fit(all_traj)
    param_mean = pca.mean_

    # 6b)  project each trajectory
    traj_pcs = {m: pca.transform(results["trajectories"][m] - param_mean) for m in methods}

    # 6c)  build a fresh model and make the metric grid (*shortcut ratio*)
    contour_model = toy_nn(x_train.shape[1], 1).to(base_cfg["device"])
    PC1, PC2, METRIC = compute_metric_grid_auto(
        contour_model,
        traj_list=[traj_pcs[m] for m in methods],
        pca=pca,
        param_mean=param_mean,
        evaluate_fn=compute_shortcut_ratio,    # any scalar metric
        steps=60,
    )

    # 6d)  plot + save
    methods = ["naive", "fix_step", "grad_surgery", "fix_soft", "gt"]
    display_names = [get_display_name(m) if m != "gt" else "GT" for m in methods]
    
    plot_contour_trajectories_as_arrows_skip(
        PC1,
        PC2,
        METRIC,
        [traj_pcs[m] for m in methods],
        display_names,
        title="Shortcut-Ratio Landscape  &  Optimisation Paths",
        label="|w_S| / ||w||",
        cmap="plasma",
        skip=100,
    )
    plt.savefig("6_shortcut_contour_and_paths.png", dpi=300)
    plt.close()

   
    
   

if __name__ == "__main__":
    main()
