from .layers import Linear
from ..utils import glorot
import torch
import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):
    def __init__(self, name, dim, mlp_dim=None):
        super(Discriminator, self).__init__()
        self.dim = dim
        self.mlp_dim = mlp_dim
        self.layer = disc_dict[name.lower()](dim, mlp_dim)

    def forward(self, x, y, outer=False):
        score = self.layer(x, y, outer)
        return score


class InnerProd(nn.Module):
    def __init__(self, dim, mlp_dim):
        super(InnerProd, self).__init__()
        self.dim = dim

    def forward(self, x, y, outer=False):
        if outer:
            score = torch.matmul(x, y.transpose(1,0))      
        else:
            score = torch.sum((x * y), dim=-1)
        return score


class Bilinear(nn.Module):
    def __init__(self, dim, mlp_dim):
        super(Bilinear, self).__init__()
        self.dim = dim
        self.bil = nn.Bilinear(dim, dim, 1)
        self.weight = glorot([dim, dim])

    def forward(self, x, y, outer=False):
        if outer:
            score = torch.matmul(torch.matmul(x, self.weight), y.transpose(1,0))
        else:
            score = torch.squeeze(self.bil(x, y), dim=-1)
        return score


class MLP(nn.Module):
    def __init__(self, dim, mlp_dim):
        super(MLP, self).__init__()
        self.dim = dim
        self.layers = nn.ModuleList()
        self.mlp_dim = mlp_dim
        for i in range(1, len(self.mlp_dim) - 1):
            self.layers.append(Linear(self.mlp_dim[i - 1], self.mlp_dim[i], act=F.relu))
        self.layers.append(Linear(self.mlp_dim[-2], self.mlp_dim[-1], act=lambda x: x))

    def forward(self, x, y, outer=False):
        h = torch.cat([x, y], dim=1)
        for layer in self.layers:
            h = layer(h)
        return torch.squeeze(h, dim=-1)


disc_dict = {
    "inner": InnerProd,
    "bilinear": Bilinear,
    "mlp": MLP
}
