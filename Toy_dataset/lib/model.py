import torch
import torch.nn as nn
import torch.nn.functional as F


class toy_nn(nn.Module):
    def __init__(self, input_size, output_size):
        super(toy_nn, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.fc1 = nn.Linear(input_size, 5)
        self.fc2 = nn.Linear(5, 10)
        self.fc3 = nn.Linear(10, output_size)

    def forward(self, x):
        x = nn.PReLU()(self.fc1(x))
        x = nn.PReLU()(self.fc2(x))
        x = self.fc3(x)
        return x
