import torch
from .others import FF


# this whole thing of get_layer_uid and layer name comes from https://github.com/tkipf/gcn
_LAYER_UIDS = {}


def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


class Layer(torch.nn.Module):
    """
    Provides standard interface.
    """

    def __init__(self, input_dim, output_dim, dropout=0., *,
                 act=lambda x: x,
                 batch_norm=False,
                 fast_forward=False,
                 **kwargs):
        super(Layer, self).__init__()
        layer = self.__class__.__name__.lower()
        self.name = layer + '_' + str(get_layer_uid(layer))
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.mlp = torch.nn.Linear(input_dim, output_dim)
        self.act = act
        if batch_norm:
            self.batch_norm = torch.nn.BatchNorm1d(output_dim)
        else:
            self.batch_norm = None
        if fast_forward:
            self.ff = FF(input_dim)
        else:
            self.ff = None

    def forward(self, inputs):
        raise NotImplementedError

    def __repr__(self):
        return self.name + '(' + str(self.input_dim) + '->' + str(self.output_dim) + ')'
