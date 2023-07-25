import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderFC(torch.nn.Module):
    def __init__(self, layers):
        super(EncoderFC, self).__init__()
        self.inp = torch.nn.Linear(layers[0], layers[1])
        self.layers = nn.ModuleList()
        for h in layers[1:-1]:
            self.layers.append(torch.nn.Linear(h, h))
        self.out = torch.nn.Linear(layers[-2], layers[-1])

    def forward(self, x):
        # Flatten tensor input
        x = F.relu(self.inp(x))
        for l in self.layers:
            x = F.relu(l(x))
        return self.out(x)

class VariationalEncoderFC(torch.nn.Module):
    def __init__(self, layers):
        super(VariationalEncoderFC, self).__init__()
        self.inp = torch.nn.Linear(layers[0], layers[1])
        self.layers = nn.ModuleList()
        for h in layers[1:-1]:
            self.layers.append(torch.nn.Linear(h, h))
        self.out_1 = torch.nn.Linear(layers[-2], layers[-1])
        self.out_2 = torch.nn.Linear(layers[-2], layers[-1])

    def forward(self, x): 
        # Flatten tensor input
        x = F.relu(self.inp(x))
        for l in self.layers:
            x = F.relu(l(x))
        return self.out_1(x), self.out_2(x)
