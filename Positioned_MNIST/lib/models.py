
# models.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class bert_with_linear(nn.Module):
    def __init__(self, bert_model, hidden_dim, output_dim):
        super(bert_with_linear, self).__init__()
        self.bert_model = bert_model
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.initialize_linear()

    def initialize_linear(self):
        self.linear = nn.Linear(self.hidden_dim, self.output_dim)
        torch.manual_seed(42)
        nn.init.xavier_normal_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, **kwargs):
        output = self.bert_model(**kwargs)
        hidden_states = output.hidden_states[-1]
        output = self.linear(hidden_states[:, 0, :])
        return output

# Toy NN


class toy_nn(nn.Module):
    def __init__(self, input_size, output_size):
        super(toy_nn, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.fc1 = nn.Linear(input_size, 5)
        self.fc2 = nn.Linear(5, 10)
        self.fc3 = nn.Linear(10, output_size)

    def forward(self, x):
        x = F.prelu(self.fc1(x), weight=0.25)
        x = F.prelu(self.fc2(x), weight=0.25)
        x = self.fc3(x)
        return x

# ConvNet for concept or shortcut


class ConvNet(nn.Module):
    """
    A CNN for 3-channel 28x28 input. 
    We store the final hidden layer in self.concepts.
    """

    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 50)
        self.fc2 = nn.Linear(50, 1)

    def forward(self, x):
        # x: [N, 3, 28, 28]
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        self.concepts = x
        logits = self.fc2(x).flatten()
        return logits


# linear_classifier


class linear_classifier(nn.Module):

    def __init__(self, input_size, output_size):
        super(linear_classifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, output_size)
        self.concepts = None  # set in forward

    def forward(self, x):
        x = self.fc1(x)       # => [N, 50]
        self.concepts = x
        logits = self.fc2(x)  # => [N, 1]
        return logits


# ResNet wrapper for 'unknown' features

class ResNetUnknown(nn.Module):
    """
    A wrapper around resnet18 => [N, 1000].
    Then a linear => [N, 50].
    We store self.concepts so generate_concepts() can read it.
    """

    def __init__(self):
        super().__init__()
        self.resnet = torchvision.models.resnet18(weights='DEFAULT')
        self.fc = nn.Linear(1000, 50)
        self.concepts = None

    def forward(self, x):
        # x: [N, 3, 28, 28]
        hidden = self.resnet(x)
        out = self.fc(hidden)
        self.concepts = out
        return out
