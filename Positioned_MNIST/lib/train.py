
# train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math

from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import numpy as np
from torch.utils.data import Sampler


# custom batch Sampling: to make sure there is at least some labeled samples in each batch
class LabeledBatchSampler(Sampler):

    def __init__(self, data_source, labels, batch_size, drop_last=False):
        self.data_source = data_source
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.drop_last = drop_last

        # indices corresponding to labeled samples (label != -1), unlabeled are -1
        self.labeled_idx = np.where(self.labels != -1)[0]
        #  possible indices
        self.all_idx = np.arange(len(self.data_source))

    def __iter__(self):
        # shuffle all indices for this epoch
        indices = self.all_idx.copy()
        np.random.shuffle(indices)

        # partition indices into batches
        batches = []
        for i in range(0, len(indices), self.batch_size):
            batch = list(indices[i:i+self.batch_size])
            if len(batch) < self.batch_size and self.drop_last:
                continue

            # If the batch does not contain any labeled sample,replace one random entry in the batch with a random labeled index.
            if not any(idx in self.labeled_idx for idx in batch):
                replace_position = np.random.randint(len(batch))
                # choose a random labeled index
                labeled_sample = np.random.choice(self.labeled_idx)
                batch[replace_position] = labeled_sample

            batches.append(batch)

        for batch in batches:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.data_source) // self.batch_size
        else:
            return int(np.ceil(len(self.data_source) / self.batch_size))


# Gradient helpers


def zero_grads(model):
    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()


def get_grads(model):
    return [p.grad.clone() if p.grad is not None else None for p in model.parameters()]


def set_grads(model, grad_list):
    for p, g in zip(model.parameters(), grad_list):
        if p is not None and g is not None:
            p.grad.copy_(g)
        elif p is not None and p.grad is not None:
            p.grad.zero_()


# dosage() => treatment effect


def dosage(data, model, treatment_idx, treatment_range=(-10, 10)):
    data_clone = data.clone().detach()
    vals = np.linspace(treatment_range[0], treatment_range[1], 20)
    all_preds = []
    for val in vals:
        data_clone[:, treatment_idx] = val
        pred = model(data_clone).detach().cpu().numpy()
        all_preds.append(pred.reshape(-1, 1))
    # shape => [N, 20]
    all_preds = np.concatenate(all_preds, axis=1)
    return all_preds.std(axis=1).mean()

# Basic dataset => (x, y).y = -1 means unlabeled (for the unlabeled dataset):


class MyDataset(Dataset):
    def __init__(self, data, label):
        super().__init__()
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]


def unlab_loss_with_threshold(logits, threshold=0.95):
    probs = torch.sigmoid(logits)
    # choose mask -># bernouilli distribution  (probabilities here) continous function
    mask = (probs > threshold) | (probs < (1.0 - threshold))
    if mask.sum() == 0:
        # always return a scalar that *does* require grad
        return torch.tensor(0.0, device=logits.device, requires_grad=True)
    pseudo = (probs >= 0.5).float()
    cf_logits = logits[mask]  # introuces 0
    cf_labels = pseudo[mask]
    return nn.functional.binary_cross_entropy_with_logits(cf_logits, cf_labels)


# Trainer class


class Trainer:
    def __init__(self,
                 model,
                 device,
                 # labeled data
                 train_data, train_label,
                 # test data
                 test_data, test_label,
                 # main losses
                 train_loss_fn,
                 test_loss_fn,
                 # optional unlabeled data
                 unlab_data=None,
                 # weighting for unlabeled loss
                 lambda_unlab=0.1,
                 # confidence threshold for unlabeled pseudo-label
                 unlab_threshold=0.95,
                 # optimizer
                 optimizer_fn=optim.Adam,
                 optimizer_params={"lr": 0.01, "weight_decay": 0},
                 scheduler_fn=optim.lr_scheduler.StepLR,
                 scheduler_params={"step_size": 10000, "gamma": 0.99},
                 num_epochs=10,
                 save_every=10,
                 batch_size=16,
                 drop_last=False,
                 grad_method=None):

        self.model = model.to(device)
        self.device = device

        self.train_data = train_data
        self.train_label = train_label
        self.test_data = test_data
        self.test_label = test_label

        self.unlab_data = unlab_data
        self.lambda_unlab = lambda_unlab
        self.unlab_threshold = unlab_threshold

        self.train_loss_fn = train_loss_fn
        self.test_loss_fn = test_loss_fn

        self.optimizer_fn = optimizer_fn
        self.optimizer_params = optimizer_params
        self.scheduler_fn = scheduler_fn
        self.scheduler_params = scheduler_params
        self.num_epochs = num_epochs
        self.save_every = save_every
        # explicit grad_method override or fall back to env
        if grad_method:
            self.grad_method = grad_method.lower().strip()
        else:
            self.grad_method = os.environ.get(
                "GRADIENT_METHOD", "naive").lower().strip()
        # build labeled dataloader
        self.train_ds = MyDataset(train_data, train_label)
        # use custom sampler
        self.train_dl = DataLoader(
            self.train_ds,
            batch_sampler=LabeledBatchSampler(
                self.train_ds, train_label, batch_size, drop_last=drop_last)
        )

        # build test dataloader
        self.test_ds = MyDataset(test_data, test_label)
        self.test_dl = DataLoader(self.test_ds, batch_size=16, shuffle=False)

        # Unlabeled => label=-1
        self.unlab_dl = None
        if self.unlab_data is not None:
            unl_labels = torch.full(
                (len(self.unlab_data),), -1.0, dtype=torch.float32)
            self.unlab_ds = MyDataset(self.unlab_data, unl_labels)
            self.unlab_dl = DataLoader(
                self.unlab_ds, batch_size=16, shuffle=False)

        # logs
        self.train_loss = []
        self.test_loss = []
        self.train_auc = []
        self.test_auc = []
        self.test_treatment_effect = []
        self.shortcut_weight_history = []
        # — gradient‐based logging —
        # will hold numpy arrays of shape [150] (flattened per‐step grads)
        self.grad_label_list = []
        self.grad_unlabel_list = []
        # will hold cosine and angle scalars per step
        self.grad_cosines = []
        self.grad_angles = []

    def train(self,
              classification=True,
              to_print=True,
              estimate_treatment_effect=False,
              shortcut_variable_idx=-1):

        optimizer = self.optimizer_fn(
            self.model.parameters(), **self.optimizer_params)
        scheduler = self.scheduler_fn(optimizer, **self.scheduler_params)

        with tqdm(total=self.num_epochs, desc="Epoch", unit="epoch") as pbar:
            for epoch in range(self.num_epochs):
                self.model.train()

                labeled_iter = iter(self.train_dl)
                unl_iter = iter(
                    self.unlab_dl) if self.unlab_dl is not None else None

                steps = max(len(self.train_dl), len(
                    self.unlab_dl) if self.unlab_dl else 0)

                for step_i in range(steps):
                    # initialize losses as scalars that *do* require grad —
                    loss_label = torch.tensor(
                        0.0, device=self.device, requires_grad=True)
                    loss_unlab = torch.tensor(
                        0.0, device=self.device, requires_grad=True)
                    # get labeled batch
                    try:
                        x_lab, y_lab = next(labeled_iter)
                        x_lab = x_lab.to(self.device)
                        y_lab = y_lab.to(self.device)
                    except StopIteration:
                        x_lab, y_lab = None, None

                    # get unlabeled batch
                    x_unl = None
                    if unl_iter is not None:
                        try:
                            x_unl, _ = next(unl_iter)
                            x_unl = x_unl.to(self.device)
                        except StopIteration:
                            x_unl = None

                    if x_lab is not None:
                        logits_lab = self.model(x_lab).squeeze()
                        mask = (y_lab >= 0)
                        if mask.sum() > 0:
                            loss_label = self.train_loss_fn(
                                logits_lab[mask], y_lab[mask])

                    if x_unl is not None:
                        logits_unl = self.model(x_unl).squeeze()
                        # thresholded unlabeled loss
                        loss_unlab = unlab_loss_with_threshold(
                            logits_unl, threshold=self.unlab_threshold)

                    # total loss = L_label + lambda * L_unlab
                    total_loss = loss_label + self.lambda_unlab * loss_unlab

                    # If naive => single pass with total_loss
                    if self.grad_method == "naive":
                        optimizer.zero_grad()
                        total_loss.backward()
                        optimizer.step()
                        scheduler.step()

                    else:
                        # two-pass backprop

                        optimizer.zero_grad()
                        if x_lab is not None:
                            loss_label.backward(retain_graph=True)
                            g_label = get_grads(self.model)
                        else:
                            g_label = [None] * \
                                len(list(self.model.parameters()))

                        # unlabeled (only if present)
                        zero_grads(self.model)
                        if x_unl is not None:
                            (self.lambda_unlab *
                             loss_unlab).backward(retain_graph=True)
                            g_unlab = get_grads(self.model)
                        else:
                            g_unlab = [None] * \
                                len(list(self.model.parameters()))

                        # ───── record raw gradients & alignment ─────────────────────
                        # collect only the real gradient tensors
                        flat_lab_parts = [g.view(-1)
                                          for g in g_label if g is not None]
                        flat_unl_parts = [g.view(-1)
                                          for g in g_unlab if g is not None]
                        if flat_lab_parts and flat_unl_parts:
                            flat_lab = torch.cat(flat_lab_parts)
                            flat_unl = torch.cat(flat_unl_parts)
                            denom = flat_lab.norm() * flat_unl.norm() + 1e-12
                            cos_sim = (flat_lab @ flat_unl) / denom
                            angle = torch.acos(torch.clamp(cos_sim, -1.0, 1.0))
                        else:
                            # no meaningful grads this step
                            cos_sim = torch.tensor(0.0, device=self.device)
                            angle = torch.tensor(0.0, device=self.device)

                        self.grad_cosines.append(cos_sim.item())
                        self.grad_angles.append(angle.item())
                        # ——————————————————————————————————————
                        # fix-a-step
                        if self.grad_method == "fix_a_step":
                            # simple sign check
                            dot_t = torch.zeros(1, device=self.device)
                            for gl, gu in zip(g_label, g_unlab):
                                if gl is not None and gu is not None:
                                    dot_t += (gl * gu).sum()

                            if dot_t >= 0:
                                final_grads = [
                                    (gl + gu) if (gl is not None and gu is not None) else gl
                                    for gl, gu in zip(g_label, g_unlab)
                                ]
                            else:
                                final_grads = g_label

                        elif self.grad_method == "gradient_surgery":
                            # Enhanced gradient surgery with dynamic scaling and range validation
                            eps = 1e-12
                            
                            # Calculate global cosine similarity for dynamic scaling
                            dot_tot = torch.zeros(1, device=self.device)
                            normL_sq = torch.zeros(1, device=self.device)
                            normU_sq = torch.zeros(1, device=self.device)
                            for gl, gu in zip(g_label, g_unlab):
                                if gl is not None and gu is not None:
                                    dot_tot += (gl * gu).sum()
                                    normL_sq += (gl * gl).sum()
                                    normU_sq += (gu * gu).sum()
                            
                            nL = torch.sqrt(normL_sq) + eps
                            nU = torch.sqrt(normU_sq) + eps
                            global_cos_sim = (dot_tot / (nL * nU)).item()
                            
                            # Per-parameter projection when conflicting
                            proj_unlab = []
                            if dot_tot < 0:
                                # Conflicting gradients: apply per-parameter projection
                                for gl, gu in zip(g_label, g_unlab):
                                    if gl is None or gu is None:
                                        proj_unlab.append(gu if gl is None else gl)
                                        continue
                                    dot = (gl * gu).sum()
                                    if dot < 0:
                                        scale = (dot / (gl.norm()**2 + eps)).item()
                                        # Check if scale is in valid range [0, 1]
                                        if 0.0 <= scale <= 1.0:
                                            gu = gu - (dot / (gl.norm()**2 + eps)) * gl
                                        else:
                                            # Discard gU if scale is out of [0,1] range
                                            gu = torch.zeros_like(gu)
                                    proj_unlab.append(gu)
                                scale_align = 1.0  # No additional scaling for conflicting case
                            else:
                                # Aligned gradients: use cosine similarity as dynamic scaling
                                if 0.0 <= global_cos_sim <= 1.0:
                                    proj_unlab = g_unlab
                                    scale_align = global_cos_sim  # Use cosine similarity as scale
                                else:
                                    # Discard gU if cosine similarity is out of [0,1] range
                                    proj_unlab = [torch.zeros_like(gu) if gu is not None else None for gu in g_unlab]
                                    scale_align = 0.0  # Effectively no unlabeled contribution
                            
                            # Combine the gradients with dynamic scaling
                            final_grads = []
                            for gl, gu in zip(g_label, proj_unlab):
                                if gl is None and gu is None:
                                    final_grads.append(None)
                                elif gl is None:
                                    final_grads.append(scale_align * gu)
                                elif gu is None:
                                    final_grads.append(gl)
                                else:
                                    final_grads.append(gl + scale_align * gu)

                        elif self.grad_method == "fix_a_step_soft":
                            # Enhanced fix_a_step_soft with proper range validation
                            eps = 1e-12
                            dot_tot = torch.zeros(1, device=self.device)
                            normL_sq = torch.zeros(1, device=self.device)
                            normU_sq = torch.zeros(1, device=self.device)

                            for gl, gu in zip(g_label, g_unlab):
                                if gl is not None and gu is not None:
                                    dot_tot += (gl * gu).sum()
                                    normL_sq += (gl * gl).sum()
                                    normU_sq += (gu * gu).sum()

                            # cosine similarity = dot / (‖gL‖‖gU‖)
                            nL = torch.sqrt(normL_sq) + eps
                            nU = torch.sqrt(normU_sq) + eps
                            global_cos_sim = (dot_tot / (nL * nU)).item()
                            
                            # Use cosine similarity as soft scaling factor with proper validation
                            if 0.0 <= global_cos_sim <= 1.0:
                                alpha = global_cos_sim  # Use cosine similarity directly
                            else:
                                alpha = 0.0  # Discard unlabeled gradients if cosine sim out of range

                            # Combine gradients with soft scaling
                            final_grads = []
                            for gl, gu in zip(g_label, g_unlab):
                                if gl is None and gu is None:
                                    final_grads.append(None)
                                elif gl is None:
                                    final_grads.append(alpha * gu)
                                elif gu is None:
                                    final_grads.append(gl)
                                else:
                                    final_grads.append(gl + alpha * gu)


                        else:
                            raise ValueError(
                                f"Unrecognised grad_method: {self.grad_method!r}")

                        # apply and step

                        zero_grads(self.model)
                        set_grads(self.model, final_grads)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), max_norm=1.0)
                        optimizer.step()
                        scheduler.step()

                # End step loop

                # evaluate
                self.model.eval()
                with torch.no_grad():
                    train_pred = self.model(
                        self.train_data.to(self.device)).squeeze()
                    test_pred = self.model(
                        self.test_data.to(self.device)).squeeze()

                y_tr = self.train_label.to(self.device)
                mask_tr = (y_tr >= 0)  # 1 0r 0
                if mask_tr.sum() > 0:  # Y=C
                    train_loss_val = self.test_loss_fn(
                        train_pred[mask_tr], y_tr[mask_tr])
                else:
                    # no labeled examples? default to zero
                    train_loss_val = torch.tensor(0.0, device=self.device)
                test_loss_val = self.test_loss_fn(
                    test_pred,  self.test_label.to(self.device))

                self.train_loss.append(train_loss_val.item())
                self.test_loss.append(test_loss_val.item())

                te_val = 0.0
                if estimate_treatment_effect and shortcut_variable_idx >= 0:
                    te_val = dosage(self.train_data.to(
                        self.device), self.model, shortcut_variable_idx)
                self.test_treatment_effect.append(te_val)

                if shortcut_variable_idx >= 0:
                    # assume the very first parameter is your final linear weight
                    weight_param = next(self.model.parameters())
                    w_flat = weight_param.view(-1)
                    if shortcut_variable_idx < w_flat.numel():
                        val = w_flat[shortcut_variable_idx].item()
                        self.shortcut_weight_history.append(abs(val))
                    else:
                        # fallback if index is out of bounds
                        self.shortcut_weight_history.append(0.0)

                if classification:
                    # get probabilities
                    train_probs = torch.sigmoid(
                        train_pred[mask_tr]).cpu().numpy()
                    test_probs = torch.sigmoid(test_pred).cpu().numpy()
                    y_train = self.train_label.cpu().numpy()
                    y_test = self.test_label.cpu().numpy()

                    # create masks to select only labeled samples (where label >= 0) so doesnt pick -1
                    valid_train_mask = y_train >= 0
                    valid_test_mask = y_test >= 0

                    if valid_train_mask.sum() > 0:
                        train_auc = roc_auc_score(
                            y_tr[mask_tr].cpu().numpy(), train_probs) if mask_tr.sum() else 0.0
                    else:
                        train_auc = 0.0
                    if valid_test_mask.sum() > 0:
                        test_auc = roc_auc_score(
                            y_test[valid_test_mask], test_probs[valid_test_mask])
                    else:
                        test_auc = 0.0

                    self.train_auc.append(train_auc)
                    self.test_auc.append(test_auc)

                    msg = (
                        f"Epoch {epoch+1}/{self.num_epochs} " +
                        f"TrainLoss: {train_loss_val:.4f}, " +
                        f"TestLoss: {test_loss_val:.4f}, " +
                        f"TrainAUC: {train_auc:.4f}, " +
                        f"TestAUC: {test_auc:.4f}, " +
                        f"TE: {te_val:.4f}"
                    )
                else:
                    msg = (
                        f"Epoch {epoch+1}/{self.num_epochs} " +
                        f"TrainLoss: {train_loss_val:.4f}, " +
                        f"TestLoss: {test_loss_val:.4f}, " +
                        f"TE: {te_val:.4f}"
                    )

                pbar.set_description(msg)
                pbar.update(1)

# TO DO:
# Methods: Use known concepts ->for labeled data
# Embeddings->supervised learning (complete set of embeddings) : done
# Sanity print out metrics in train /test
# Two channels in prediction process->Labeled->known concepts, Unlabeled->C,U,S
# Supervised learning baseline (compare)
# Set-up Weights and Biases
# List of plots to output:
