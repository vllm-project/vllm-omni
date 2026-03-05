import torch
import torch.nn as nn


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


def SiLU():
    if hasattr(torch.nn, 'SiLU'):
        return torch.nn.SiLU()
    else:
        return Swish()
