from .baselayer import Layer
from ...utils import zeros
import torch


# from https://github.com/fanyun-sun/InfoGraph/blob/master/unsupervised/gin.py
class GIN(Layer):
    """
    output = MLP((1+eps)input + sum(input))
    """

    def __init__(self, input_dim, output_dim, dropout=0., *,
                 act=torch.relu, **kwargs):
        super(GIN, self).__init__(input_dim, output_dim, dropout, act=act, **kwargs)
        self.mlp = torch.nn.Linear(input_dim, output_dim)
        self.eps = zeros([1])
        self.act = act

    def forward(self, inputs):
        x = inputs[0]
        adj = inputs[1]
        if self.training:
            x = torch.dropout(x, self.dropout, True)  # dropout
        # sum pooling
        y = self.mlp((1 + self.eps) * x + torch.mm(adj, x))
        y = self.act(y)
        return y
