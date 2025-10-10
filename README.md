# Fix-A-Shortcut: Mitigating Shortcut Learning in Causal Representation Learning

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Official implementation** of the Fix-A-Shortcut methods for addressing shortcut learning in semi-supervised learning scenarios.

## Abstract

Semi-supervised learning (SSL) methods often suffer from **shortcut learning**, where models exploit spurious correlations in unlabeled data rather than learning robust causal features. This repository implements several Fix-A-Shortcut variants that detect and mitigate such shortcuts during training, leading to improved generalization and robustness.

## Key Contributions

- **Novel gradient-based detection** of shortcut learning in SSL
- **Multiple mitigation strategies**: step-based correction, soft weighting, and gradient surgery
- **Comprehensive evaluation** on synthetic and real-world benchmarks
- **Theoretical analysis** of shortcut learning dynamics

## Installation

### Requirements
- Python 3.8+
- PyTorch 1.9+
- CUDA (optional, for GPU acceleration)

### Setup
```bash
git clone [link to repo]
cd Fix-A-Shortcut
pip install -r requirements.txt
```

## Repository Structure

```
Fix-A-Shortcut/
├── Toy_dataset/                    # Synthetic experiments
│   └── lib/
│       ├── data.py                 # Synthetic data generation with configurable shortcuts
│       ├── model.py                # Neural network architectures
│       ├── gradients.py            # Core Fix-A-Shortcut algorithms
│       ├── experiment.py           # Experimental pipeline
│       └── demo.py                 # Minimal reproduction script
│
├── Coloured_MNIST/                 # Real-world benchmark
│   ├── coloured_mnist.py           # Colored MNIST dataset with shortcuts
│   ├── utils.py                    # Concept generation utilities
│   └── lib/
│       ├── models.py               # CNN architectures (ResNet, ConvNet)
│       └── train.py                # SSL training with Fix-A-Shortcut
│
├── requirements.txt                # Dependencies
└── README.md                       # This file
```

## Algorithms

### Fix-A-Shortcut Variants

| Method | Description | Key Idea |
|--------|-------------|----------|
| **FAS** | Fix-A-Shortcut | Hard conflict detection and gradient masking |
| **FAS-soft** | Fix-A-Shortcut-Soft | Soft cosine-based gradient weighting |
| **FAS-GS-Align** | Fix-A-Shortcut Gradient Surgery Aligned | Projection based variant for alignment |
| **Vanilla-SSL** | Baseline | Standard semi-supervised learning |

## Usage

### Synthetic Experiments

Reproduce core results on controllable synthetic data:

```python
from Toy_dataset.lib.experiment import synthetic_experiments

# Run all methods on synthetic data with shortcuts
results = synthetic_experiments(
    num_samples=2000,           # Dataset size
    rho_cu=0.2,                # Causal-spurious correlation  
    shortcut_mode="c_and_u",   # Shortcut type
    beta=0.8,                  # Shortcut strength
    shortcut_ratio=0.8,        # Fraction with shortcuts
    n_labeled=75,              # SSL labeled samples
    lambda_u=10.0,             # Unlabeled loss weight
    epochs=500                 # Training epochs
)

# Results contain trajectories, final weights, and statistics for all methods
for method, stats in results['stats'].items():
    print(f"{method}: conflicts detected = {stats.get('conflict_count', 0)}")
```

### Colored MNIST Benchmark

**Step 1: Download MNIST data automatically**
MNIST data will be downloaded from [http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/) automatically to `./colored_mnist/` when you first run the script.

**Step 2: Generate colored dataset with shortcuts**
```bash
cd Coloured_MNIST
python coloured_mnist.py --shortcut_ratio 1.0 --lambda_unlab 0.8 --label_frac 0.05 --epochs 20 --methods fix_a_step_soft fix_a_step gradient_surgery naive
```

**Available arguments:**
- `--shortcut_ratio` (float, default=1.0): Fraction of training data with spurious correlations
- `--lambda_unlab` (float, default=0.0): Weight for unlabeled loss in SSL
- `--label_frac` (float, default=0.05): Fraction of training data that is labeled (5% = semi-supervised)
- `--epochs` (int, default=5): Number of training epochs
- `--n_runs` (int, default=20): Number of independent runs per method
- `--device` (str, default="cpu"): Device to use ("cpu" or "cuda")
- `--methods` (list): Methods to run, choose from:
  - `fix_a_step_soft`: Fix-A-Shortcut with soft weighting
  - `fix_a_step`: Fix-A-Shortcut with step-based correction
  - `gradient_surgery`: Fix-A-Shortcut with gradient surgery
  - `naive`: Standard semi-supervised learning baseline
- `--random_state` (int, default=42): Random seed for reproducibility

This will:
- Download MNIST automatically if not present
- Generate colored versions where color correlates with digit labels
- Inject shortcuts into the training data
- Train all specified methods and save results
- Save processed data to `./data/colored_mnist/`

**Step 3: Use the generated data in your experiments**
```python
# The processed data files will be saved and can be loaded as:
import torch

base_dir = "./data/colored_mnist/λ0.8_sr1_lf0.05"  # Matches your experiment settings
train_data = torch.load(f"{base_dir}/train_data.pt")
train_digit_label = torch.load(f"{base_dir}/train_digit_label.pt") 
train_color_label = torch.load(f"{base_dir}/train_color_label.pt")
test_data = torch.load(f"{base_dir}/test_data.pt")
test_digit_label = torch.load(f"{base_dir}/test_digit_label.pt")
test_color_label = torch.load(f"{base_dir}/test_color_label.pt")
```

### MIMIC-CXR Benchmark

**Step 1: Download MIMIC-CXR dataset**
You need to download the MIMIC-CXR dataset from PhysioNet:
- **Dataset**: [MIMIC-CXR Database v2.0.0](https://physionet.org/content/mimic-cxr/2.0.0/)
- **Requirements**: You need to complete CITI training and get approval for access
- **Files needed**:
  - `mimic-cxr-2.0.0-chexpert.csv.gz` (labels file)
  - `files/` directory containing all chest X-ray images

**Step 2: Organize your data directory**
Create the following structure:
```
/path/to/mimic-cxr/
├── mimic-cxr-2.0.0-chexpert.csv.gz    # Patient labels and metadata
└── files/
    └── p10/
    └── p11/
    └── p12/
    └── ...                            # Patient directories with .jpg images
```

**Step 3: Run experiments with gender shortcuts**
```bash
cd MIMIC-CXR
python mimic-cxr.py \
    --root_dir /path/to/mimic-cxr \
    --patients_csv /path/to/mimic-cxr/mimic-cxr-2.0.0-chexpert.csv.gz \
    --task Edema \
    --p_male_pos_train 0.8 \
    --p_male_neg_train 0.2 \
    --p_male_pos_test 0.2 \
    --p_male_neg_test 0.8 \
    --grad_methods naive,fix_a_step,fix_a_step_soft,gradient_surgery \
    --epochs 20 \
    --seeds 5
```

**Available arguments:**
- `--root_dir`: Path to MIMIC-CXR dataset directory
- `--patients_csv`: Path to the chexpert labels CSV file
- `--task`: Target pathology (default: "Edema")
- `--p_male_pos_train/test`: P(Male | Positive) for train/test sets
- `--p_male_neg_train/test`: P(Male | Negative) for train/test sets  
- `--grad_methods`: Comma-separated list of methods to run
- `--epochs`: Number of training epochs
- `--seeds`: Number of random seeds to run
- `--unlabeled_pct`: Fraction of training data that is unlabeled (default: 0.90)
- `--lambda_list`: Semi-supervised loss weights to try (default: "2.0")
- `--batch_size`: Training batch size (default: 16)
- `--out_dir`: Output directory for results (default: "runs/exp")

This will create gender-based shortcuts where male/female correlates with the target pathology in training but anti-correlates in test, allowing evaluation of shortcut mitigation methods.

### Minimal Reproduction

For quick verification, run the demo scripts:

```bash
# Synthetic experiments (generates 3 essential plots)
cd Toy_dataset/lib && python demo.py

# Expected output: 
# - demo_final_weights.png (learned vs true weights)
# - demo_generalization_gap.png (test-train gap)  
# - demo_shortcut_trajectory.png (shortcut weight evolution)
```

## Key Parameters

### Experimental Settings
- **`shortcut_ratio`** ∈ [0,1]: Fraction of training data with spurious correlations
- **`lambda_u`** > 0: Weight for unlabeled loss in SSL objective  
- **`n_labeled`**: Number of labeled samples (controls SSL difficulty)

### Shortcut Configuration  
- **`beta`/`delta_c`/`delta_u`**: Correlation strengths between features and shortcuts
- **`rho_cu`**: Correlation between causal and spurious features
- **`sigma_s`**: Noise level in shortcut generation

## Citation

If you use this code in your research, please cite:

```bibtex
@article{fixashortcut2024,
  title={Fix-A-Shortcut: Mitigating Shortcut Learning in Semi-Supervised Learning},
  author={[Authors]},
  journal={[Conference/Journal]},
  year={2024},
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact
To be released

---

**Paper**: [Link to paper when available]  
**Project Page**: [Link to project page when available]
