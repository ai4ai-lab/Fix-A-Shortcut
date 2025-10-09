
# coloured_mnist.py

from utils import generate_concepts_batched
from lib.models import ConvNet, linear_classifier
from lib.train import Trainer, dosage
import os
import sys
import argparse
import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import datasets
from lib.models import ResNetUnknown

# so can import from lib/*
sys.path.append("..")


# color_grayscale_arr, assign_color_label


def color_grayscale_arr(arr, red=True):
    h, w = arr.shape
    arr = arr.view(h, w, 1)
    if red:
        arr = torch.cat([arr, torch.zeros((h, w, 2), dtype=arr.dtype)], dim=2)
    else:
        arr = torch.cat([
            torch.zeros((h, w, 1), dtype=arr.dtype),
            arr,
            torch.zeros((h, w, 1), dtype=arr.dtype)
        ], dim=2)
    return arr


def assign_color_label(data, color_label):
    out_list = []
    for i in range(len(data)):
        c = color_label[i].item()
        colored = color_grayscale_arr(data[i], red=bool(c))
        colored = colored / 255.0
        out_list.append(colored.unsqueeze(0))  # => [1,28,28,3]

    out = torch.cat(out_list, dim=0)  # => [N,28,28,3]
    return out.permute(0, 3, 1, 2)    # => [N,3,28,28]

# ColoredMnist


class ColoredMnist:

    def __init__(self, random_state=42):
        self.dataset = datasets.MNIST(
            "./colored_mnist", train=False, download=True)
        self.rng = np.random.RandomState(random_state)

    def get_train_test_idx(self, shortcut_ratio=1.0):
        data, label = self.dataset.data, self.dataset.targets
        N = len(label)
        idx_all = np.arange(N)
        self.rng.shuffle(idx_all)
        half = N // 2
        self.train_idx = idx_all[:half]
        self.test_idx = idx_all[half:]

        # digit≥5 => 1, else 0
        all_digit = (label >= 5).float()
        self.train_digit = all_digit[self.train_idx]
        self.test_digit = all_digit[self.test_idx]

        train_size = len(self.train_idx)
        corr_count = int(train_size * shortcut_ratio)
        # first 'corr_count' match digit -> color
        color_corr = self.train_digit[:corr_count]
        # rest random
        color_rand = torch.from_numpy(self.rng.binomial(
            1, 0.5, train_size - corr_count)).float()
        self.train_color = torch.cat([color_corr, color_rand], dim=0)
        # shuffle them
        perm2 = self.rng.permutation(train_size)
        self.train_color = self.train_color[perm2]

        # test -> reversed
        label_test_orig = label[self.test_idx]
        self.test_color = (label_test_orig < 5).float()

        self.train_data = data[self.train_idx]
        self.test_data = data[self.test_idx]

    def assign_shortcuts(self):
        train_colored = assign_color_label(self.train_data, self.train_color)
        test_colored = assign_color_label(self.test_data,  self.test_color)

        return (
            train_colored, self.train_digit, self.train_color,
            test_colored,  self.test_digit,  self.test_color
        )
     # generate unbiased training data for concepts

    def generate_unbiased_data(self):

        self.unbiased_color_label = self.rng.binomial(
            1, 0.5, self.train_data.shape[0])
        self.unbiased_color_label = torch.from_numpy(
            self.unbiased_color_label).float()
        # call assign_color_label with only two arguments
        self.unbiased_train_data = assign_color_label(
            self.train_data, self.unbiased_color_label)
        return self.unbiased_train_data, self.unbiased_color_label

    def build_or_load_data(path, base_seed):
        os.makedirs(path, exist_ok=True)
        req = [
            "train_data.pt", "train_digit_label.pt", "train_color_label.pt",
            "test_data.pt", "test_digit_label.pt", "test_color_label.pt",
            "unbiased_train_data.pt", "unbiased_color_label.pt"
        ]
        if not all(os.path.exists(os.path.join(path, f)) for f in req):
            print("▶ Generating Colored MNIST data")
            cm = ColoredMnist(random_state=base_seed)
            cm.get_train_test_idx(shortcut_ratio=args.shortcut_ratio)
            ub_x, ub_y = cm.generate_unbiased_data()
            x_tr, y_tr_d,   y_tr_c,  x_te, y_te_d, y_te_c = cm.assign_shortcuts()
            torch.save(x_tr, os.path.join(path, "train_data.pt"))
            torch.save(y_tr_d, os.path.join(path, "train_digit_label.pt"))
            torch.save(y_tr_c, os.path.join(path, "train_color_label.pt"))
            torch.save(x_te, os.path.join(path, "test_data.pt"))
            torch.save(y_te_d, os.path.join(path, "test_digit_label.pt"))
            torch.save(y_te_c, os.path.join(path, "test_color_label.pt"))
            torch.save(ub_x, os.path.join(path, "unbiased_train_data.pt"))
            torch.save(ub_y, os.path.join(path, "unbiased_color_label.pt"))
            print("✔ Data saved.")
        else:
            print("▶ Loading existing data")
        # return loaded tensors
        tr_x = torch.load(os.path.join(path, "train_data.pt"))
        tr_d = torch.load(os.path.join(path, "train_digit_label.pt"))
        tr_c = torch.load(os.path.join(path, "train_color_label.pt"))
        te_x = torch.load(os.path.join(path, "test_data.pt"))
        te_d = torch.load(os.path.join(path, "test_digit_label.pt"))
        te_c = torch.load(os.path.join(path, "test_color_label.pt"))
        ub_x = torch.load(os.path.join(path, "unbiased_train_data.pt"))
        ub_y = torch.load(os.path.join(path, "unbiased_color_label.pt"))
        return tr_x, tr_d, tr_c, te_x, te_d, te_c, ub_x, ub_y


def build_or_load_data(path, base_seed):
    os.makedirs(path, exist_ok=True)
    req = [
        "train_data.pt", "train_digit_label.pt", "train_color_label.pt",
        "test_data.pt", "test_digit_label.pt", "test_color_label.pt",
        "unbiased_train_data.pt", "unbiased_color_label.pt"
    ]
    if not all(os.path.exists(os.path.join(path, f)) for f in req):
        print("▶ Generating Colored MNIST data")
        cm = ColoredMnist(random_state=base_seed)
        cm.get_train_test_idx(shortcut_ratio=args.shortcut_ratio)
        ub_x, ub_y = cm.generate_unbiased_data()
        x_tr, y_tr_d,   y_tr_c,  x_te, y_te_d, y_te_c = cm.assign_shortcuts()
        torch.save(x_tr, os.path.join(path, "train_data.pt"))
        torch.save(y_tr_d, os.path.join(path, "train_digit_label.pt"))
        torch.save(y_tr_c, os.path.join(path, "train_color_label.pt"))
        torch.save(x_te, os.path.join(path, "test_data.pt"))
        torch.save(y_te_d, os.path.join(path, "test_digit_label.pt"))
        torch.save(y_te_c, os.path.join(path, "test_color_label.pt"))
        torch.save(ub_x, os.path.join(path, "unbiased_train_data.pt"))
        torch.save(ub_y, os.path.join(path, "unbiased_color_label.pt"))
        print("✔ Data saved.")
    else:
        print("▶ Loading existing data")
    # return loaded tensors
    tr_x = torch.load(os.path.join(path, "train_data.pt"))
    tr_d = torch.load(os.path.join(path, "train_digit_label.pt"))
    tr_c = torch.load(os.path.join(path, "train_color_label.pt"))
    te_x = torch.load(os.path.join(path, "test_data.pt"))
    te_d = torch.load(os.path.join(path, "test_digit_label.pt"))
    te_c = torch.load(os.path.join(path, "test_color_label.pt"))
    ub_x = torch.load(os.path.join(path, "unbiased_train_data.pt"))
    ub_y = torch.load(os.path.join(path, "unbiased_color_label.pt"))
    return tr_x, tr_d, tr_c, te_x, te_d, te_c, ub_x, ub_y


if __name__ == "__main__":
    import argparse
    import os
    import numpy as np
    import torch
    import torch.nn as nn
    from lib.models import ConvNet, ResNetUnknown, linear_classifier
    from lib.train import Trainer
    from coloured_mnist import ColoredMnist, generate_concepts_batched

    # ---------- ARGS ----------
    parser = argparse.ArgumentParser(
        description="Colored MNIST + Shortcut/Concept Models"
    )
    parser.add_argument("--shortcut_ratio", type=float, default=1.0)
    parser.add_argument("--lambda_unlab",   type=float, default=0.0)
    parser.add_argument("--random_state",   type=int,   default=42)
    parser.add_argument(
        "--label_frac", type=float, default=0.05,
        help="Fraction of the training set to treat as labeled (e.g. 0.1 for 10%)"
    )

    parser.add_argument("--device",         type=str,   default="cpu")
    parser.add_argument(
        "--methods", nargs="+",
        default=["fix_a_step_soft","fix_a_step", "gradient_surgery","naive"]
    )
    parser.add_argument("--epochs",   type=int,   default=5)
    parser.add_argument("--n_runs",   type=int,   default=20,
                        help="number of independent runs per method")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ---------- SETUP PATHS ----------
    exp_name = f"λ{args.lambda_unlab:g}_sr{args.shortcut_ratio:g}_lf{args.label_frac:g}"
    base_dir = os.path.join("data", "colored_mnist", exp_name)
    os.makedirs(base_dir, exist_ok=True)

    # 0) Build or load the shared data
    def build_or_load_data(path, seed):
        cm = ColoredMnist(random_state=seed)
        cm.get_train_test_idx(shortcut_ratio=args.shortcut_ratio)
        ub_x, ub_y = cm.generate_unbiased_data()
        x_tr, y_tr_d, y_tr_c, x_te, y_te_d, y_te_c = cm.assign_shortcuts()
        torch.save(x_tr, os.path.join(path, "train_data.pt"))
        torch.save(y_tr_d, os.path.join(path, "train_digit_label.pt"))
        torch.save(y_tr_c, os.path.join(path, "train_color_label.pt"))
        torch.save(x_te, os.path.join(path, "test_data.pt"))
        torch.save(y_te_d, os.path.join(path, "test_digit_label.pt"))
        torch.save(y_te_c, os.path.join(path, "test_color_label.pt"))
        torch.save(ub_x,  os.path.join(path, "unbiased_train_data.pt"))
        torch.save(ub_y,  os.path.join(path, "unbiased_color_label.pt"))

    required = ["train_data.pt", "train_digit_label.pt", "train_color_label.pt",
                "test_data.pt", "test_digit_label.pt", "test_color_label.pt",
                "unbiased_train_data.pt", "unbiased_color_label.pt"]
    if not all(os.path.exists(os.path.join(base_dir, f)) for f in required):
        print("▶ Generating data for experiment", exp_name)
        build_or_load_data(base_dir, args.random_state)
    else:
        print("▶ Loading existing data from", base_dir)

    # Now load them once
    train_data = torch.load(os.path.join(base_dir, "train_data.pt"))
    train_digit_label = torch.load(
        os.path.join(base_dir, "train_digit_label.pt"))
    train_color_label = torch.load(
        os.path.join(base_dir, "train_color_label.pt"))
    test_data = torch.load(os.path.join(base_dir, "test_data.pt"))
    test_digit_label = torch.load(
        os.path.join(base_dir, "test_digit_label.pt"))
    test_color_label = torch.load(
        os.path.join(base_dir, "test_color_label.pt"))
    unbiased_train_data = torch.load(
        os.path.join(base_dir, "unbiased_train_data.pt"))
    unbiased_color_label = torch.load(
        os.path.join(base_dir, "unbiased_color_label.pt"))

    loss_fn = nn.BCEWithLogitsLoss()
    N = train_data.shape[0]
    n_lab = int(args.label_frac * N)          # 10% labeled
    perm_all = np.arange(N)

    # ---------- PER-METHOD LOOP ----------
    for method in args.methods:
        print(f"\n=== Running method: {method} ===")
        method_dir = os.path.join(base_dir, method)
        os.makedirs(method_dir, exist_ok=True)

        # run several independent seeds
        for run_i in range(args.n_runs):
            seed = args.random_state + run_i
            print(f"\n--- Run {run_i+1}/{args.n_runs} – seed={seed} ---")
            torch.manual_seed(seed)
            np.random.seed(seed)

            # create a run folder
            run_tag = f"run{run_i}"
            run_dir = os.path.join(method_dir, run_tag)
            os.makedirs(run_dir, exist_ok=True)

            # 1) train the shortcut model on color
            shortcut_model = ConvNet().to(device)

            shortcut_trainer = Trainer(
                model=shortcut_model,
                device=device,
                train_data=unbiased_train_data,
                train_label=unbiased_color_label,    # color = shortcut signal
                test_data=test_data,
                test_label=test_color_label,
                train_loss_fn=loss_fn,
                test_loss_fn=loss_fn,
                unlab_data=None,      # no unlabeled loss
                lambda_unlab=0.0,     # zero weight on any unlabeled term
                num_epochs=args.epochs,
                grad_method="naive"   # single‐pass, fully supervised
            )
            # reset histories
            for attr in ("train_loss", "test_loss", "train_auc", "test_auc",
                         "test_treatment_effect", "shortcut_weight_history"):
                setattr(shortcut_trainer, attr, [])
            shortcut_trainer.train(classification=True,
                                   estimate_treatment_effect=False)

            # 2) fully-supervised concept model on digit
            np.random.shuffle(perm_all)
            lab_idx, unl_idx = perm_all[:n_lab], perm_all[n_lab:]
            semi_digit = train_digit_label.clone()
            semi_digit[unl_idx] = -1.0 

            concept_model = ConvNet().to(device)
            concept_trainer = Trainer(
                model=concept_model,
                device=device,
                train_data=unbiased_train_data,
                train_label=semi_digit,
                test_data=test_data,
                test_label=test_digit_label,
                train_loss_fn=loss_fn,
                test_loss_fn=loss_fn,
                # don’t pass unlab_data
                unlab_data=None,
                # zero out the unlabeled‐loss weight
                lambda_unlab=0.0,
                optimizer_params={"lr": 1e-3, "weight_decay": 1e-4},
                num_epochs=args.epochs,
                # force naive two‐pass (really single‐pass) supervised gradient
                grad_method="naive"
            )
            for attr in ("train_loss", "test_loss", "train_auc", "test_auc",
                         "test_treatment_effect", "shortcut_weight_history"):
                setattr(concept_trainer, attr, [])
            concept_trainer.train(classification=True,
                                  estimate_treatment_effect=False)

            # 3) build and save embeddings
            unknown_model = ResNetUnknown().to(device)  # all from a feature extractor
            cc_tr, uc_tr, sc_tr = generate_concepts_batched(
                net_c=concept_model,
                net_u=unknown_model,
                net_s=shortcut_model,
                data=train_data,
                batch_size=32
            )
            cc_te, uc_te, sc_te = generate_concepts_batched(
                net_c=concept_model,
                net_u=unknown_model,
                net_s=shortcut_model,
                data=test_data,
                batch_size=32
            )
            train_embed = torch.cat([cc_tr, uc_tr, sc_tr], dim=-1)
            test_embed = torch.cat([cc_te, uc_te, sc_te], dim=-1)
            # normalize
            m, s = train_embed.mean(0, True), train_embed.std(0, True)+1e-8
            train_normed = (train_embed - m)/s
            test_normed = (test_embed - m)/s

            torch.save(train_normed, os.path.join(
                run_dir, "train_last_layer.pt"))
            torch.save(test_normed,  os.path.join(
                run_dir, "test_last_layer.pt"))
            print("→ embeddings saved to", run_dir)

           
            
            # 4) final digit classifier
            # 4) final digit classifier – supervised and unsupervised heads both see full (C,U,S) embedding,
#       but labels Y=f(C) so supervised gradient only aligns with C
            np.random.shuffle(perm_all)
            lab2, unl2 = perm_all[:n_lab], perm_all[n_lab:]
            semi2 = train_digit_label.clone()
            semi2[unl2] = -1.0  # mark unlabeled examples

            final_model = linear_classifier(
                train_normed.shape[1], 1).to(device)
            final_trainer = Trainer(
                model=final_model,
                device=device,
                # full C+U+S embeddings for both supervised and test
                train_data=train_normed,
                train_label=semi2,
                test_data=test_normed,
                test_label=test_digit_label,
                train_loss_fn=loss_fn,
                test_loss_fn=loss_fn,
                # full embeddings for unlabeled loss
                unlab_data=train_normed[unl2],
                lambda_unlab=args.lambda_unlab,
                num_epochs=20,
                batch_size=64,
                grad_method=method
            )
            # reset histories
            for attr in ("train_loss", "test_loss", "train_auc", "test_auc",
                         "test_treatment_effect", "shortcut_weight_history"):
                setattr(final_trainer, attr, [])

            final_trainer.train(
                classification=True,
                estimate_treatment_effect=True,
                shortcut_variable_idx=100
            )

            torch.save(final_trainer, os.path.join(
                run_dir, "final_trainer.pt"))
            print("→ final model & history saved to", run_dir)
            # also test?

# TO DO:
# ssl helps with shortcuts?->trade off between accuracy and shortcut mitigation : proportion of labeled data is low->gradient methods balance that trade off better?
# # Check correlations at train and test
# # fix concept error
# See if labeled-fraction =1.0 how correlations maps look




# Methods: Use known concepts ->for labeled data
# Embeddings->supervised learning (complete set of embeddings) : DONE
# Sanity print out metrics in train /test
# Two channels in prediction process->Labeled->known concepts, Unlabeled->C,U,S: DONE
# file restructure->clean up/delete files for a clean pull : DONE
# Think of all plots needed+make scripts for these:
# Supervised learning baseline (compare)
# set up weights and biases in plot script
