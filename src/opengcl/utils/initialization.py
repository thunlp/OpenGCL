import torch


def glorot(shape):
    """Glorot & Bengio (AISTATS 2010) init."""
    initial = torch.zeros(shape, dtype=torch.float32)
    torch.nn.init.xavier_normal_(initial)
    return torch.nn.Parameter(initial)


def zeros(shape):
    """All zeros."""
    initial = torch.zeros(shape, dtype=torch.float32)
    return torch.nn.Parameter(initial)


def ones(shape):
    """All ones."""
    initial = torch.ones(shape, dtype=torch.float32)
    return torch.nn.Parameter(initial)
