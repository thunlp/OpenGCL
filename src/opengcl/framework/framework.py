from .encoder import Encoder
from .discriminator import Discriminator
from .sampler import SamplerFactory

from .estimator import BaseEstimator
from ..utils import get_device, ModelInput, GraphInput, HyperParameter, ModuleParameter
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union
from ..datasets import Graph, Graphs
import networkx as nx
from ..utils import scipy_coo_to_torch_sparse, preprocess_features


class Framework(nn.Module):
    def __init__(self, dataset, module_params: ModuleParameter, hyper_params: HyperParameter):
        super(Framework, self).__init__()
        self.dataset = dataset
        self.module_params = module_params
        self.hyper_params = hyper_params

        # row-normalize feature matrix for graphs
        features = torch.from_numpy(dataset.features()).to(torch.float32)
        features = preprocess_features(features)
        dataset.setfeatures(features)
        self.register_buffer("features", features)

        # get dimensions
        input_dim = self.features.shape[1]
        self.enc_dims = [input_dim] + [self.hyper_params.output_dim] * (self.hyper_params.hidden_size + 1)
        self.dec_dims = [self.enc_dims[-1] * 2, 1]

        if isinstance(dataset, Graphs):
            graphs_data = dataset.data
        else:
            feats = torch.from_numpy(dataset.features())
            adjmat = scipy_coo_to_torch_sparse(dataset.adjmat(sparse=True).tocoo())
            edgelist = adjmat._indices()
            weights = adjmat._values()
            graphs_data = [GraphInput(feats, None, edgelist, weights)]
        self.graphs_data = graphs_data
        self.num_graphs = len(graphs_data)

        self.graph_sampler = False
        if self.module_params.sampler in ['dgi', 'mvgrl', 'graphcl', 'gca']:
            self.graph_sampler = True
        self.mask = False
        if (self.graph_sampler and self.num_graphs > 1) or self.module_params.sampler == 'gca':
            self.mask = True
        self.outer = self.graph_sampler

        self.encoder = Encoder(self.module_params.enc, self.enc_dims,
                               getattr(dataset, 'nodesize', 1),
                               self.hyper_params.dropout,
                               self.module_params.readout,
                               self.hyper_params.ff)
        self.discriminator = Discriminator(self.module_params.dec, self.enc_dims[-1], self.dec_dims)
        self.estimator = BaseEstimator(self.module_params.est)
        self.sampler = SamplerFactory(self.module_params.sampler, graphs_data,
                                      self.features, self.hyper_params.batch_size)
        self.normalize = self.module_params.est == 'nce'

    def embed(self, x):
        if self.normalize:
            return F.normalize(self.encoder(x), dim=-1)
        return self.encoder(x)

    def forward(self, x: ModelInput, pos: ModelInput, neg: ModelInput):
        def get_anchor():
            pos_mask, neg_mask = None, None

            # repeat
            def repeat(start_idx):
                old_idx = start_idx[0]
                vectors = []
                for i, idx in enumerate(start_idx[1:]):
                    vectors.append(hx[i].repeat(idx - old_idx, 1))
                    old_idx = idx
                return torch.cat(vectors)

            def get_mask(start_idx):
                pos_mask = torch.zeros(hx.shape[0], hpos.shape[0])
                neg_mask = torch.ones(hx.shape[0], hpos.shape[0])
                old_idx = start_idx[0]
                if self.module_params.sampler == 'graphcl':
                    pos_mask = torch.diag(torch.ones(hx.shape[0]))
                    neg_mask = 1 - pos_mask
                else:
                    for i, idx in enumerate(start_idx[1:]):
                        pos_mask[i][old_idx:idx] = 1
                        neg_mask[i][old_idx:idx] = 0
                        old_idx = idx

                return pos_mask.to(get_device()), neg_mask.to(get_device())

            if self.mask:
                pos_mask, neg_mask = get_mask(pos.start_idx)
            hxp = hxn = hx
            return hxp, hxn, pos_mask, neg_mask

        hx = self.embed(x)
        hpos = self.embed(pos)
        hneg = self.embed(neg)
        hxp, hxn, pos_mask, neg_mask = get_anchor()
        pos_score = self.discriminator(hxp, hpos, self.outer)

        if self.mask:
            neg_score = pos_score
        else:
            neg_score = self.discriminator(hxn, hneg, self.outer)
        loss = self.estimator(pos_score, neg_score, pos_mask, neg_mask)
        self.encoder.reset()
        return loss
