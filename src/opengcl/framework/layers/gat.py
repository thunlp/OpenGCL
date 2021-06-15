from .baselayer import Layer
from ...utils import zeros, glorot, get_device
import torch
from torch.nn.functional import elu


#  a reimplementation of https://github.com/PetarV-/GAT
#  from Graph Attention Networks(https://arxiv.org/abs/1710.10903)

class GAT(Layer):
    def __init__(self, input_dim, output_dim, dropout=0., *,
                 act=elu,
                 bias=True,
                 dropout_coef=0.2,
                 attn_heads=1, attn_heads_reduction='average',
                 residual=True,
                 **kwargs):
        super(GAT, self).__init__(input_dim, output_dim, dropout, act=act, **kwargs)
        if attn_heads_reduction not in ['concat', 'average']:
            raise ValueError("attn_heads_reduction must be one of {'concat', 'average'}")
        self.attn_heads_reduction = attn_heads_reduction
        attn_heads = int(attn_heads)
        if attn_heads_reduction == 'concat':
            self.output_dim *= attn_heads
        self.attn_heads = attn_heads
        self.dropout_input = dropout
        if dropout_coef is None:
            dropout_coef = dropout
        self.dropout_coef = dropout_coef
        self.bias = bias
        self.threshold_val = 1e-4
        self.residual = residual

        self.weights = []
        if self.bias:
            self.biases = []
        self.attn_kernels = []
        self.residuals = []
        for head in range(self.attn_heads):
            # weights
            w = glorot([input_dim, output_dim])
            setattr(self, 'weights_' + str(head),  w)
            self.weights.append(w)

            # biases
            if self.bias:
                b = zeros([1, output_dim])
                setattr(self, "biases_" + str(head), b)
                self.biases.append(b)

            # attention kernels: [k_self, k_neigh]
            ak1, ak2 = zeros([output_dim, 1]), zeros([output_dim, 1])
            setattr(self, "attn_kernels_A_" + str(head), ak1)
            setattr(self, "attn_kernels_B_" + str(head), ak2)
            self.attn_kernels.append([
                ak1, ak2
            ])

            if self.residual:
                res = glorot([input_dim, output_dim])
                setattr(self, 'res_' + str(head), res)
                self.residuals.append(res)

        self.inplace_leaky_relu = torch.nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, inputs):
        x = inputs[0]  # input node features (n * input_dim)
        adj = inputs[1]
        if not self.training:
            adj = adj.float()
        if self.training:
            # dropout
            x = torch.dropout(x, self.dropout_input, True)  # dropout

        def masked_softmax(vec, mask, dim=1, epsilon=1e-5, inplace=False):
            if inplace:
                vec -= torch.max(vec)
                exps = torch.exp_(vec)
                if get_device() != torch.device('cpu'):
                    masked_exps = exps.mul_(mask)
                else:
                    masked_exps = exps.mul_(mask.to_dense())

                masked_sums = masked_exps.sum(dim, keepdim=True) + epsilon
                return masked_exps.div_(masked_sums)
            exps = torch.exp(vec - torch.max(vec))
            if get_device() != torch.device('cpu'):
                masked_exps = exps * mask.float()
            else:
                masked_exps = exps * mask.to_dense()
            masked_sums = masked_exps.sum(dim, keepdim=True) + epsilon
            return masked_exps / masked_sums
        y_list = []
        for i in range(self.attn_heads):  # do for every independent attention kernel
            weight = self.weights[i]
            if self.bias:
                bias = self.biases[i]
            else:
                bias = 0.
            a = self.attn_kernels[i]

            # feature_in = h * W, (n * output_dim)
            feat_in = torch.mm(x, weight)

            # attention coefficients
            # c(i,j) = a^T(Wh_i || Wh_j) = a_1^T Wh_i + a_2^T Wh_j
            # c = a_1^T Wh + (a_2^T Wh)^T : broadcasting, (n * n)

            f1 = torch.mm(feat_in, a[0])
            f2 = torch.mm(feat_in, a[1])

            c = f1 + f2.T

            # leakyReLU and softmax
            if not self.training:
                c = masked_softmax(self.inplace_leaky_relu(c), adj, inplace=True)
            else:
                c = masked_softmax(torch.nn.functional.leaky_relu(c, 0.2), adj)

            if self.training:
                # dropout
                c = torch.dropout(c, self.dropout_coef, True)  # dropout
                feat_in = torch.dropout(feat_in, self.dropout_input, True)

            feat_out = torch.mm(c, feat_in)

            if self.bias:
                feat_out += bias

            if self.residual:
                feat_out += torch.mm(x, self.residuals[i])

            feat_out = self.act(feat_out)
            y_list.append(feat_out)

        # aggregate
        if self.attn_heads_reduction == 'concat':
            y = torch.cat(y_list, dim=1)  # concatenate along dim 1 (n * (k*output_dim))
        else:
            y = torch.mean(torch.stack(y_list), dim=0)   # (n * output_dim)
        return y

