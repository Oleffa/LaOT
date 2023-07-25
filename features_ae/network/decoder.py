import torch
import torch.nn as nn
import torch.nn.functional as F

class DecoderFC(torch.nn.Module):
    def __init__(self, layers):
        super(DecoderFC, self).__init__()
        self.inp = torch.nn.Linear(layers[0], layers[1])
        self.layers = nn.ModuleList()
        for h_i, h in enumerate(layers[1:-1]):
            self.layers.append(torch.nn.Linear(h, h))
        self.out = torch.nn.Linear(layers[-2], layers[-1])

    def forward(self, x):
        x = F.relu(self.inp(x))
        for l in self.layers:
            x = F.relu(l(x))
        return self.out(x)
