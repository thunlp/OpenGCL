from .baselayer import Layer
from ...utils import zeros
import torch


class Linear(Layer):
    """MLP layer."""

    def __init__(self, input_dim, output_dim, dropout=0., *,
                 act=torch.relu, bias=False, **kwargs):
        super(Linear, self).__init__(input_dim, output_dim, dropout, act=act, **kwargs)
        self.weight = torch.nn.Parameter(torch.zeros(input_dim, output_dim), requires_grad=True)
        if bias:
            self.bias = zeros([output_dim])
        else:
            self.bias = None
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, inputs):
        x = inputs
        if self.training:
            x = torch.dropout(x, self.dropout, True)

        pre_sup = torch.mm(x, self.weight)
        output = pre_sup

        # bias
        if self.bias is not None:
            output += self.bias
        return self.act(output)



class MLP(Linear):
    """MLP layer."""
    def forward(self, inputs):
        return super().forward(inputs[0])
