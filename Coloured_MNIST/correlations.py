#!/usr/bin/env python
"""
Self-correlation plots for embedding analysis.

Outputs
    • Self-correlation heat-maps for the *Vanilla* embeddings (train & test)
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import wandb
from torch.utils.data import Dataset
from lib.train import Trainer
from lib.models import linear_classifier
from torch.nn.modules.linear import Linear

# ─── Safe-load support ───────────────────────────────────────────


class MyDataset(Dataset):
    def __init__(self, data, label):
        self.data, self.label = data, label

    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx], self.label[idx]


torch.serialization.add_safe_globals(
    [MyDataset, Trainer, linear_classifier, Linear])
# ────────────────────────────────────────────────────────────────

BASE_DIR = "./data/colored_mnist"
METHODS = ["naive", "gradient_surgery", "fix_a_step", "fix_a_step_soft"]
NICE_NAMES = {"naive": "Naïve",
              "gradient_surgery": "Grad Surgery", "fix_a_step": "Fix-a-step", "fix_a_step_soft": "Fix-A-Step-Soft"}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ───────── helpers ──────────────────────────────────────────────


def _intish(x: float) -> str:
    return str(int(x)) if float(x).is_integer() else f"{x:g}"


def expr_dir(lam, sr, lf):
    return os.path.join(BASE_DIR, f"λ{_intish(lam)}_sr{_intish(sr)}_lf{_intish(lf)}")


def find_run_dirs(method, lam, sr, lf):
    return sorted(glob.glob(os.path.join(expr_dir(lam, sr, lf), method, "run*")))


def load_embedding(run_dir, split="test"):
    fn = os.path.join(run_dir, f"{split}_last_layer.pt")
    if not os.path.isfile(fn):
        raise FileNotFoundError(fn)
    return torch.load(fn, map_location="cpu")        # always load to CPU

import os
import glob
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import wandb
from torch.utils.data import Dataset
from lib.train import Trainer
from lib.models import linear_classifier
from torch.nn.modules.linear import Linear

# ─── Safe-load support ───────────────────────────────────────────


class MyDataset(Dataset):
    def __init__(self, data, label):
        self.data, self.label = data, label

    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx], self.label[idx]


torch.serialization.add_safe_globals(
    [MyDataset, Trainer, linear_classifier, Linear])
# ────────────────────────────────────────────────────────────────

BASE_DIR = "./data/colored_mnist"
METHODS = ["naive", "gradient_surgery", "fix_a_step", "fix_a_step_soft"]
NICE_NAMES = {"naive": "Naïve",
              "gradient_surgery": "Grad Surgery", "fix_a_step": "Fix-a-step", "fix_a_step_soft": "Fix-A-Step-Soft"}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ───────── helpers ──────────────────────────────────────────────


def _intish(x: float) -> str:
    return str(int(x)) if float(x).is_integer() else f"{x:g}"


def expr_dir(lam, sr, lf):
    return os.path.join(BASE_DIR, f"λ{_intish(lam)}_sr{_intish(sr)}_lf{_intish(lf)}")


def find_run_dirs(method, lam, sr, lf):
    return sorted(glob.glob(os.path.join(expr_dir(lam, sr, lf), method, "run*")))


def load_embedding(run_dir, split="test"):
    fn = os.path.join(run_dir, f"{split}_last_layer.pt")
    if not os.path.isfile(fn):
        raise FileNotFoundError(fn)
    return torch.load(fn, map_location="cpu")        # always load to CPU


def load_model(run_dir):
    fn = os.path.join(run_dir, "final_trainer.pt")
    if not os.path.isfile(fn):
        raise FileNotFoundError(fn)
    tr = torch.load(fn, map_location=DEVICE, weights_only=False)
    tr.model.to(DEVICE).eval()
    return tr.model


def compute_group_correlations(y_pred: np.ndarray, emb: np.ndarray) -> dict[str, float]:
    D = emb.shape[1]
    g = D // 3
    idx = {"Concept": range(0, g), "Unknown": range(
        g, 2*g), "Shortcut": range(2*g, 3*g)}
    out = {}
    for grp, ids in idx.items():
        vals = [np.corrcoef(y_pred, emb[:, j])[0, 1] for j in ids]
        out[grp] = np.nanmean([v for v in vals if not np.isnan(v)])
    return out


def plot_pairwise_feature_correlations(method, lam, sr, lf, save_dir=None, wandb_log=True):
    """
    Plot pairwise correlations between Known, Unknown, and Shortcut features 
    at train vs test time for a specific method.
    
    Parameters:
    -----------
    method : str
        Method name (e.g., 'naive', 'gradient_surgery')
    lam : float
        Lambda unlabeled parameter
    sr : float 
        Shortcut ratio parameter
    lf : float
        Label fraction parameter
    save_dir : str, optional
        Directory to save plots (defaults to experiment directory)
    wandb_log : bool
        Whether to log to wandb
        
    Returns:
    --------
    dict : Correlation statistics for train and test
    """
    runs = find_run_dirs(method, lam, sr, lf)
    if not runs:
        print(f"[warning] no runs found for method {method}")
        return {}
    
    # Use first run (could average over multiple seeds if needed)
    run_dir = runs[0]
    
    # Load train and test embeddings
    try:
        train_emb = load_embedding(run_dir, "train").numpy()
        test_emb = load_embedding(run_dir, "test").numpy()
    except FileNotFoundError as e:
        print(f"[warning] embeddings not found for {method}: {e}")
        return {}
    
    # Define feature groups
    D = train_emb.shape[1]
    g = D // 3
    groups = {
        "Known": (0, g),
        "Unknown": (g, 2*g), 
        "Shortcut": (2*g, 3*g)
    }
    
    # Compute correlations for train and test
    results = {}
    
    for split, emb in [("train", train_emb), ("test", test_emb)]:
        # Compute average correlations between feature groups
        group_corrs = {}
        
        for group1, (start1, end1) in groups.items():
            for group2, (start2, end2) in groups.items():
                if group1 <= group2:  # Avoid duplicates, include diagonal
                    # Get features from each group
                    features1 = emb[:, start1:end1]
                    features2 = emb[:, start2:end2]
                    
                    # Compute all pairwise correlations between groups
                    if group1 == group2:
                        # Within-group correlations (excluding diagonal)
                        corr_matrix = np.corrcoef(features1.T)
                        mask = ~np.eye(corr_matrix.shape[0], dtype=bool)
                        correlations = corr_matrix[mask]
                    else:
                        # Between-group correlations
                        correlations = []
                        for i in range(features1.shape[1]):
                            for j in range(features2.shape[1]):
                                corr = np.corrcoef(features1[:, i], features2[:, j])[0, 1]
                                if not np.isnan(corr):
                                    correlations.append(corr)
                        correlations = np.array(correlations)
                    
                    # Store statistics
                    key = f"{group1}-{group2}" if group1 != group2 else f"{group1} (within)"
                    group_corrs[key] = {
                        'mean': np.nanmean(correlations),
                        'std': np.nanstd(correlations),
                        'abs_mean': np.nanmean(np.abs(correlations)),
                        'n_pairs': len(correlations)
                    }
        
        results[split] = group_corrs
    
    # Create visualization as heatmap
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Extract data for plotting
    pair_types = list(results['train'].keys())
    
    # 1. Mean correlations heatmap (Train vs Test)
    train_means = [results['train'][pt]['mean'] for pt in pair_types]
    test_means = [results['test'][pt]['mean'] for pt in pair_types]
    
    # Create data matrix for heatmap
    heatmap_data = np.array([train_means, test_means])
    
    # Plot heatmap
    vmax = max(np.abs(heatmap_data).max(), 0.1)  # Ensure reasonable scale
    im1 = ax1.imshow(heatmap_data, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
    
    # Add text annotations
    for i in range(2):
        for j in range(len(pair_types)):
            text = ax1.text(j, i, f'{heatmap_data[i, j]:.3f}',
                           ha="center", va="center", 
                           color="white" if abs(heatmap_data[i, j]) > 0.6*vmax else "black",
                           fontweight='bold')
    
    ax1.set_xticks(range(len(pair_types)))
    ax1.set_xticklabels(pair_types, rotation=45, ha='right')
    ax1.set_yticks(range(2))
    ax1.set_yticklabels(['Train', 'Test'])
    ax1.set_title(f'Mean Correlations: {NICE_NAMES.get(method, method)}')
    
    # Add colorbar
    cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label('Correlation')
    
    # 2. Standard deviation heatmap
    train_stds = [results['train'][pt]['std'] for pt in pair_types]
    test_stds = [results['test'][pt]['std'] for pt in pair_types]
    std_data = np.array([train_stds, test_stds])
    
    im2 = ax2.imshow(std_data, cmap='YlOrRd', vmin=0, vmax=std_data.max(), aspect='auto')
    
    for i in range(2):
        for j in range(len(pair_types)):
            text = ax2.text(j, i, f'{std_data[i, j]:.3f}',
                           ha="center", va="center",
                           color="white" if std_data[i, j] > 0.6*std_data.max() else "black",
                           fontweight='bold')
    
    ax2.set_xticks(range(len(pair_types)))
    ax2.set_xticklabels(pair_types, rotation=45, ha='right')
    ax2.set_yticks(range(2))
    ax2.set_yticklabels(['Train', 'Test'])
    ax2.set_title('Standard Deviation of Correlations')
    
    cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label('Standard Deviation')
    
    # 3. Absolute correlations heatmap
    train_abs_means = [results['train'][pt]['abs_mean'] for pt in pair_types]
    test_abs_means = [results['test'][pt]['abs_mean'] for pt in pair_types]
    abs_data = np.array([train_abs_means, test_abs_means])
    
    im3 = ax3.imshow(abs_data, cmap='viridis', vmin=0, vmax=abs_data.max(), aspect='auto')
    
    for i in range(2):
        for j in range(len(pair_types)):
            text = ax3.text(j, i, f'{abs_data[i, j]:.3f}',
                           ha="center", va="center",
                           color="white" if abs_data[i, j] > 0.6*abs_data.max() else "black",
                           fontweight='bold')
    
    ax3.set_xticks(range(len(pair_types)))
    ax3.set_xticklabels(pair_types, rotation=45, ha='right')
    ax3.set_yticks(range(2))
    ax3.set_yticklabels(['Train', 'Test'])
    ax3.set_title('Correlation Strength (|Correlation|)')
    
    cbar3 = plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    cbar3.set_label('|Correlation|')
    
    # 4. Train-Test difference heatmap
    differences = np.array(test_means) - np.array(train_means)
    diff_data = differences.reshape(1, -1)
    
    vmax_diff = max(np.abs(diff_data).max(), 0.05)
    im4 = ax4.imshow(diff_data, cmap='RdBu_r', vmin=-vmax_diff, vmax=vmax_diff, aspect='auto')
    
    for j in range(len(pair_types)):
        text = ax4.text(j, 0, f'{diff_data[0, j]:.3f}',
                       ha="center", va="center",
                       color="white" if abs(diff_data[0, j]) > 0.6*vmax_diff else "black",
                       fontweight='bold')
    
    ax4.set_xticks(range(len(pair_types)))
    ax4.set_xticklabels(pair_types, rotation=45, ha='right')
    ax4.set_yticks([0])
    ax4.set_yticklabels(['Test - Train'])
    ax4.set_title('Generalization Gap (Test - Train)')
    
    cbar4 = plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
    cbar4.set_label('Correlation Difference')
    
    plt.suptitle(f'Pairwise Feature Correlations: {NICE_NAMES.get(method, method)}\n'
                 f'λ={lam}, sr={sr}, lf={lf}', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Save plot locally (not just in exp_dir)
    if save_dir is None:
        save_dir = expr_dir(lam, sr, lf)
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Save in experiment directory
    save_path = os.path.join(save_dir, f"pairwise_correlations_heatmap_{method}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Also save in current directory for easy access
    local_save_path = f"pairwise_correlations_heatmap_{method}_λ{lam}_sr{sr}_lf{lf}.png"
    plt.savefig(local_save_path, dpi=300, bbox_inches='tight')
    
    print(f"✅ Pairwise correlations heatmap saved:")
    print(f"   Experiment dir: {save_path}")
    print(f"   Local dir: {local_save_path}")
    
    # Log to wandb if requested
    if wandb_log and wandb.run is not None:
        wandb.log({f"pairwise_correlations_heatmap_{method}": wandb.Image(save_path)})
    
    plt.close(fig)
    
    # Print summary statistics
    print(f"\n=== Pairwise Feature Correlations: {NICE_NAMES.get(method, method)} ===")
    for split in ['train', 'test']:
        print(f"\n{split.upper()}:")
        for pair_type in pair_types:
            stats = results[split][pair_type]
            print(f"  {pair_type:15s}: mean={stats['mean']:6.3f}, "
                  f"std={stats['std']:6.3f}, |mean|={stats['abs_mean']:6.3f}, "
                  f"n={stats['n_pairs']:3d}")
    
    return results


# ────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    pa = argparse.ArgumentParser(
        description="Self-correlation plots for embedding analysis")
    pa.add_argument("--lambda_unlab",   type=float, required=True)
    pa.add_argument("--shortcut_ratio", type=float, required=True)
    pa.add_argument("--label_frac",     type=float, required=True)
    pa.add_argument("--wandb_project",  type=str,   required=True)
    args = pa.parse_args()

    lam, sr, lf = args.lambda_unlab, args.shortcut_ratio, args.label_frac
    exp_dir = expr_dir(lam, sr, lf)

    run = wandb.init(project=args.wandb_project,
                     name=f"self_corr_plots_λ{lam}_sr{sr}_lf{lf}",
                     config=dict(lambda_unlab=lam, shortcut_ratio=sr,
                                 label_frac=lf, methods=METHODS))

    # ─── Self-correlation for naive embeddings (train & test) ---
    naive_runs = find_run_dirs("naive", lam, sr, lf)
    if not naive_runs:
        print("[warning] no naïve run – self-corr heat-maps skipped")
    else:
        for split in ("train", "test"):
            emb = load_embedding(naive_runs[0], split).numpy()
            
            # Debug: Check for any issues with the embeddings
            print(f"\n=== Debug info for {split} embeddings ===")
            print(f"Shape: {emb.shape}")
            print(f"Has NaN: {np.isnan(emb).any()}")
            print(f"Has Inf: {np.isinf(emb).any()}")
            print(f"Min: {emb.min():.6f}, Max: {emb.max():.6f}")
            
            # Check for zero variance features
            feature_stds = np.std(emb, axis=0)
            zero_var_features = np.where(feature_stds == 0)[0]
            if len(zero_var_features) > 0:
                print(f"Warning: {len(zero_var_features)} features have zero variance: {zero_var_features}")
            
            # Compute correlation matrix with proper handling
            C = np.corrcoef(emb.T)
            
            # Debug correlation matrix
            print(f"Correlation matrix shape: {C.shape}")
            print(f"Has NaN in correlation matrix: {np.isnan(C).any()}")
            print(f"Number of NaN values: {np.isnan(C).sum()}")
            
            # Handle NaN values - replace with 0 (no correlation)
            C_clean = np.nan_to_num(C, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Verify the cleaned matrix
            print(f"After cleaning - Has NaN: {np.isnan(C_clean).any()}")
            print(f"Correlation range: [{C_clean.min():.6f}, {C_clean.max():.6f}]")

            fig, ax = plt.subplots(figsize=(8, 8))
            
            # Draw each cell as a rectangle to avoid any grid artifacts
            from matplotlib.patches import Rectangle
            from matplotlib.colors import Normalize
            from matplotlib.cm import get_cmap
            
            norm = Normalize(vmin=-1, vmax=1)
            cmap = get_cmap("magma")
            
            n = C_clean.shape[0]
            
            # Draw each correlation cell as a rectangle
            for i in range(n):
                for j in range(n):
                    corr_val = C_clean[i, j]
                    color = cmap(norm(corr_val))
                    rect = Rectangle((j, n-1-i), 1, 1, 
                                   facecolor=color, edgecolor=color, 
                                   linewidth=0, antialiased=False)
                    ax.add_patch(rect)
            
            # Set exact limits to match the rectangles
            ax.set_xlim(0, n)
            ax.set_ylim(0, n)
            ax.set_aspect('equal')
            
            # Remove all ticks and spines completely
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

            # Add colorbar manually
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_ticks(np.linspace(-1, 1, 11))
            cbar.ax.set_yticklabels([f"{t:.1f}" for t in np.linspace(-1, 1, 11)])

            # Group divisions will be visible naturally through correlation patterns
            g = n // 3

            # Add clear group dividers between Known, Unknown, Shortcut
            g = n // 3
            separator_positions = [g, 2*g]
            
            # Add thick white lines as dividers for clear separation
            for pos in separator_positions:
                # Horizontal dividers
                ax.axhline(y=n-pos, color='white', linewidth=6, alpha=1.0, zorder=10, solid_capstyle='butt')
                # Vertical dividers  
                ax.axvline(x=pos, color='white', linewidth=6, alpha=1.0, zorder=10, solid_capstyle='butt')

            # Add group labels
            centers = [g/2, 3*g/2, 5*g/2]
            y_centers = [n - g/2, n - 3*g/2, n - 5*g/2]
            
            # Bottom labels
            ax.text(centers[0], -3, "Known", ha='center', va='top', 
                   fontsize=14, fontweight='bold')
            ax.text(centers[1], -3, "Unknown", ha='center', va='top', 
                   fontsize=14, fontweight='bold')
            ax.text(centers[2], -3, "Shortcut", ha='center', va='top', 
                   fontsize=14, fontweight='bold')
            
            # Left labels
            ax.text(-3, y_centers[0], "Known", ha='right', va='center', 
                   fontsize=14, fontweight='bold', rotation=90)
            ax.text(-3, y_centers[1], "Unknown", ha='right', va='center', 
                   fontsize=14, fontweight='bold', rotation=90)
            ax.text(-3, y_centers[2], "Shortcut", ha='right', va='center', 
                   fontsize=14, fontweight='bold', rotation=90)
            
            ax.set_title(f"Self-Corr {split.capitalize()} Embeddings\nλ={lam}, sr={sr}, lf={lf}",
                         fontweight="bold", pad=30, fontsize=16)
            
            plt.tight_layout()

            out_png = os.path.join(exp_dir, f"self_corr_{split}.png")
            plt.savefig(out_png, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"✅ Self-correlation {split} heatmap saved: {out_png}")
            
            # Additional debug: Save correlation matrix as text for inspection
            debug_file = os.path.join(exp_dir, f"correlation_matrix_{split}.txt")
            np.savetxt(debug_file, C_clean, fmt='%.6f')
            print(f"📊 Debug: Correlation matrix saved to {debug_file}")
            
            wandb.log({f"self_corr_{split}": wandb.Image(out_png)})
            plt.close(fig)

    # ─── end -------------------------------------------------------
    wandb.finish()
