import torch
import torch.nn as nn


class BaseReadOut(torch.nn.Module):
    def __init__(self, name, enc_dims, **kwargs):
        super().__init__()
        self.readout = readout_dict[name.lower()](enc_dims, **kwargs)

    def forward(self, *args, **kwargs):
        res = self.readout(*args, **kwargs)
        return res


class AvgReadOut(torch.nn.Module):
    def __init__(self, enc_dims, **kwargs):
        super().__init__()
        for i, j in kwargs.items():
            setattr(self, i, j)

    def forward(self, embeddings):
        return torch.mean(embeddings, 0)


class SumReadOut(torch.nn.Module):
    def __init__(self, enc_dims, **kwargs):
        super().__init__()
        for i, j in kwargs.items():
            setattr(self, i, j)

    def forward(self, embeddings):
        return torch.sum(embeddings, 0)


# from https://arxiv.org/pdf/1811.01287.pdf
class JKNetReadOut(torch.nn.Module):
    def __init__(self, enc_dims, **kwargs):
        super().__init__()
        for i, j in kwargs.items():
            setattr(self, i, j)
        self.enc_dims = enc_dims
        self.linear = nn.Linear(sum(self.enc_dims), self.enc_dims[-1])

    def forward(self, embeddings):
        # mean || max
        # print(torch.mean(embeddings, 0).shape)
        return torch.sum(embeddings, 0)

    def lintrans(self, x):
        return self.linear(x)


readout_dict = {
    "mean": AvgReadOut,
    "sum": SumReadOut,
    "jk-net": JKNetReadOut
}
