import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassifierFC(torch.nn.Module):
    def __init__(self, layers):
        super(ClassifierFC, self).__init__()
        self.inp = torch.nn.Linear(layers[0], layers[1])
        self.layers = nn.ModuleList()
        for h in layers[1:-1]:
            self.layers.append(torch.nn.Linear(h, h))
        self.out = torch.nn.Linear(layers[-1], 10)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x):
        # Flatten tensor input
        x = self.inp(x)
        for l in self.layers:
            x = F.relu(l(x))
        x = self.out(x)
        return self.softmax(x)
