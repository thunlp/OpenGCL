from .baselayer import Layer
from ...utils import glorot, zeros

import torch


class GraphConvolution(Layer):
    """Graph convolution layer."""

    def __init__(self, input_dim, output_dim, dropout=0., *,
                 act=torch.nn.PReLU(), bias=False,
                 **kwargs):
        super(GraphConvolution, self).__init__(input_dim, output_dim, dropout, act=act, **kwargs)

        setattr(self, 'weights', glorot([input_dim, output_dim]))
        if bias:
            self.bias = zeros([output_dim])
        else:
            self.bias = None

    def forward(self, inputs=None):
        x = inputs[0]
        sup = inputs[1]
        if self.training:
            x = torch.dropout(x, self.dropout, True)

        # convolve
        output = torch.zeros([sup.size()[0], self.output_dim], device=x.device)
        pre_sup = torch.mm(x, getattr(self, 'weights'))
        support = torch.mm(sup, pre_sup)
        output += support

        # bias
        if self.bias is not None:
            output += self.bias
        return self.act(output)
