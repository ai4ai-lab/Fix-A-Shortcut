#!/usr/bin/env python
"""
demo.py - Minimal demonstration of Fix-A-Shortcut methods

This script runs a streamlined version of the synthetic experiments,
showing the core functionality 
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

from data import generate_synthetic_base, generate_training_test_data
from experiment import synthetic_experiments
from plot_functions import (
    plot_weights_gradient_methods,
    plot_generalization_gap,
    plot_shortcut_weight_trajectory,
    compute_generalization_gap
)


def main():
    """Run core Fix-A-Shortcut demonstration with essential plots."""

    print("=" * 60)
    print("Fix-A-Shortcut Demo - Conference Submission")
    print("=" * 60)
    print("Methods tested:")
    print("  - Vanilla SSL: Standard semi-supervised learning")
    print("  - FAS: Fix-A-Shortcut with step-based correction")
    print("  - FAS-soft: Fix-A-Shortcut with soft weighting")
    print("  - FAS-GS-Align: Fix-A-Shortcut with gradient surgery")
    print("  - GT: Ground truth (fully supervised baseline)")
    print()

    # Core experimental configuration
    base_cfg = dict(
        num_samples=2000,
        rho_cu=0.2,              # Correlation between causal and spurious features
        sigma_y=0.1,             # Label noise
        shortcut_mode="c_and_u",  # Shortcut correlates with both causal and spurious
        beta=-0.8,               # Base correlation strength
        delta_c=0.9,             # Causal correlation
        delta_u=-0.2,            # Spurious correlation
        sigma_s=0.1,             # Shortcut noise
        shortcut_ratio=0.8,      # Fraction of training data with shortcut
        test_size=0.5,           # Train/test split
        random_state=42,         # Reproducibility
        n_labeled=75,            # Number of labeled samples for SSL
        epochs=500,              # Training epochs (reduced for demo)
        lr=1e-3,                 # Learning rate
        lambda_u=10.0,           # Semi-supervised loss weight
        device="cpu",
        to_print=False,          # Hide detailed training progress for cleaner output
    )

    print("Configuration:")
    for key, value in base_cfg.items():
        print(f"  {key}: {value}")
    print()

    # Run experiments with all methods
    print("Running experiments...")
    results = synthetic_experiments(**base_cfg)

    # Generate true weights for comparison
    # C1, C2, U1, U2, S (shortcut=0)
    true_weights = np.array([4, -0.5, 1, 2, 0])

    print("\nExperimental Results:")
    print("-" * 40)

    # Display final weight norms for each method
    for method in ['naive', 'fix_step', 'grad_surgery', 'fix_soft', 'gt']:
        final_w_norm = np.linalg.norm(results['final_weights'][method])
        display_name = method if method == 'gt' else method.replace('_', '-')
        print(f"{display_name:12s}: Final weight norm = {final_w_norm:.4f}")
    print()

    # Generate essential plots
    print("Generating plots...")

    # 1. Final weights comparison
    print("  - Final weights vs true weights")
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_weights_gradient_methods(
        results,
        true_weights,
        ax=ax,
        hidden_dim=5
    )
    plt.tight_layout()
    plt.savefig("demo_final_weights.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

    # 2. Generalization gap
    print("  - Generalization gap")
    # Generate test data for gap computation
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
        delta_c=base_cfg['delta_c'],
        delta_u=base_cfg['delta_u'],
        sigma_s=base_cfg['sigma_s'],
        shortcut_ratio=base_cfg['shortcut_ratio'],
        test_size=base_cfg['test_size'],
        random_state=base_cfg['random_state']
    )

    gaps = compute_generalization_gap(
        results, x_train, y_train, x_test, y_test, device=base_cfg['device']
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    plot_generalization_gap(gaps, ax=ax)
    plt.tight_layout()
    plt.savefig("demo_generalization_gap.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

    # 3. Shortcut weight trajectory
    print("  - Shortcut weight evolution")
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_shortcut_weight_trajectory(results, ax=ax)
    plt.tight_layout()
    plt.savefig("demo_shortcut_trajectory.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Summary statistics
    print("\nSummary Statistics:")
    print("-" * 40)
    for method, gap_data in gaps.items():
        display_name = method.replace('_', '-')
        print(f"{display_name:12s}: Train MSE = {gap_data['train_mse']:.4f}, "
              f"Test MSE = {gap_data['test_mse']:.4f}, "
              f"Gap = {gap_data['gap']:.4f}")

    print("\nDemo completed successfully!")
    print("Generated plots:")
    print("  - demo_final_weights.png      (Final learned weights vs true weights)")
    print("  - demo_generalization_gap.png (Train/test performance gap)")
    print("  - demo_shortcut_trajectory.png (Evolution of shortcut weights)")
    print("\nThese plots demonstrate the core Fix-A-Shortcut methods and their effectiveness.")


if __name__ == "__main__":
    main()
