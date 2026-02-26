#####################################
# utils.py
#####################################
import os
import pandas as pd
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.optim as optim


#####################################
# utils.py 
#####################################
import torch

def generate_concepts_batched(net_c, net_u, net_s, data, batch_size=64):
    """
    Chunk 'data' into batches, move each batch onto the model's device,
    then pass it to net_c, net_s, net_u.  Concatenate their .concepts outputs.
    """
    # Put all nets in eval mode
    net_c.eval()
    net_s.eval()
    net_u.eval()

    # figure out which device the nets live on:
    device = next(net_c.parameters()).device

    concepts_all  = []
    unknowns_all  = []
    shortcuts_all = []

    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            # slice and move to GPU (or whatever device your model is on)
            batch = data[i : i + batch_size].to(device)

            # forward pass
            net_c(batch)
            net_s(batch)
            net_u(batch)

            # collect their .concepts attributes
            concepts_all .append(net_c.concepts)      # [b,50]
            unknowns_all .append(net_u.concepts)      # [b,50]
            shortcuts_all.append(net_s.concepts)      # [b,50]

    # concatenate along batch dimension → [N,50] each
    concepts_cat  = torch.cat(concepts_all,  dim=0)
    unknowns_cat  = torch.cat(unknowns_all,  dim=0)
    shortcuts_cat = torch.cat(shortcuts_all, dim=0)

    return concepts_cat, unknowns_cat, shortcuts_cat


def generate_concepts(net_c, net_u, net_s, data):
    """
    Non‐batched: move data to net_c's device, run, then concat.
    """
    net_c.eval()
    net_s.eval()
    net_u.eval()

    device = next(net_c.parameters()).device
    x = data.to(device)

    with torch.no_grad():
        net_c(x)
        net_s(x)
        net_u(x)

    return torch.cat(
        (net_c.concepts, net_u.concepts, net_s.concepts),
        dim=-1
    )



#####################################
# MIMIC-specific code
#####################################
def mimic_known_unknown_split(data_path):
    """
    Splits known vs unknown concepts by correlation with LOS in a MIMIC experiment.
    """
    los_all = pd.read_csv(f"{data_path}/los_prediction_all.csv")

    known_concepts = []
    unknown_concepts = []

    for variable in los_all.columns[1:]:
        corr = pearsonr(los_all[variable], los_all['LOS']).statistic
        if corr > 0.1:
            if "shortcut" not in variable:
                known_concepts.append(variable)
        else:
            unknown_concepts.append(variable)

    return known_concepts, unknown_concepts


def get_gradient(model, x, y, loss_fn):
    """
    For custom usage: x.requires_grad=True => model(x) => backward => x.grad
    """
    x.requires_grad = True
    pred = model(x)
    loss = loss_fn(pred, y)
    loss.backward()
    return x.grad.detach()


#####################################
# data_loader for older code
#####################################
class data_loader:
    def __init__(self, dataset, data_path, device="cpu"):
        self.device = device
        self.dataset = dataset
        self.data_path = data_path
        self.initialize(dataset, data_path)

    def initialize(self, dataset, data_path):

        if dataset == "position_mnist":
            # load 150-d embeddings from positioned mnist
            self.train_data = torch.load(f"{data_path}/train_last_layer.pt")
            self.train_label = torch.load(f"{data_path}/train_binary_label.pt")
            self.test_data = torch.load(f"{data_path}/test_last_layer.pt")
            self.test_label = torch.load(f"{data_path}/test_binary_label.pt")

            self.basic_loss = nn.BCEWithLogitsLoss()
            self.input_size = 150
            self.output_size = 1
            self.classification = True
            self.to_print = True

            # old logic for weighting => r
            self.r = torch.tensor([1]*50 + [0]*100)
            self.r_causal = torch.tensor([1]*100 + [0.001]*50)

        elif dataset == "multinli":
            self.train_data = torch.load(f"{data_path}/train_last_layer.pt")
            self.train_label = torch.load(f"{data_path}/train_label.pt")
            self.test_data = torch.load(f"{data_path}/test_last_layer.pt")
            self.test_label = torch.load(f"{data_path}/test_label.pt")

            self.basic_loss = nn.BCEWithLogitsLoss()
            self.input_size = 150
            self.output_size = 1
            self.classification = True
            self.to_print = True

            self.r = torch.tensor([1]*50 + [0]*100)
            self.r_causal = torch.tensor([1]*100 + [0.001]*50).to(self.device)

        elif dataset == "mimic":
            # MIMIC => read CSV => known vs unknown => final shape
            self.known_concepts, self.unknown_concepts = mimic_known_unknown_split(
                data_path)
            train_data = pd.read_csv(f"{data_path}/train_data_scaled.csv")
            test_data = pd.read_csv(f"{data_path}/test_data_scaled.csv")

            col_name = self.known_concepts + self.unknown_concepts + \
                [f"shortcut_{i}" for i in range(10)]
            self.train_data = torch.tensor(
                train_data[col_name].values, dtype=torch.float32)
            train_label = torch.tensor(
                train_data['LOS'].values,      dtype=torch.float32)

            self.test_data = torch.tensor(
                test_data[col_name].values,    dtype=torch.float32)
            test_label = torch.tensor(
                test_data['LOS'].values,       dtype=torch.float32)

            # standardize y
            from sklearn.preprocessing import StandardScaler
            scalar = StandardScaler()
            train_label_scaled = scalar.fit_transform(
                train_label.reshape(-1, 1))
            test_label_scaled = scalar.transform(test_label.reshape(-1, 1))

            self.train_label = torch.tensor(
                train_label_scaled, dtype=torch.float32).reshape(-1)
            self.test_label = torch.tensor(
                test_label_scaled,  dtype=torch.float32).reshape(-1)

            self.basic_loss = nn.MSELoss()
            self.input_size = self.train_data.shape[1]
            self.output_size = 1
            self.classification = False
            self.to_print = True

            # define r => known=1, unknown=0, shortcuts=???
            self.r = torch.tensor(
                [1]*len(self.known_concepts) + [0]*len(self.unknown_concepts) + [0]*10)
            self.r_causal = torch.tensor(
                [1]*(len(self.known_concepts)+len(self.unknown_concepts)) + [0.0001]*10)
        else:
            raise ValueError(
                "Invalid dataset, must be position_mnist, multinli, or mimic.")
