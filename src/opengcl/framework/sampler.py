import random
from ..utils import process_graphs, ModelInput
from .samplers import single_view_contrast
from .samplers.augmentations import *


class SamplerFactory:
    def __init__(self, name, graph, features, batch_size, negative_ratio=5, **kwargs):
        # super(BaseSampler, self).__init__()
        self.graph = graph
        self.features = features
        self.negative_ratio = negative_ratio
        self.batch_size = batch_size
        self.name = name
        if name in sampler_dict:  # multi-view sampler
            self.sampler = sampler_dict[name](graph, batch_size=self.batch_size)
        elif name in single_view_contrast.contrast_dict:  # single-view sampler
            self.sampler = SingleViewSampler(name, graph, batch_size=self.batch_size)

    def __iter__(self):
        return self.sampler.__iter__()

    def __len__(self):
        return self.sampler.__len__()


class BaseSampler:
    """
    interfaces:
    sample_slicer(feature)
    __iter__()
    __len__()
    """
    batch_size = None
    num_graphs = None
    cache = None

    def sample_slicer(self, feature):
        iter_step = self.batch_size

        iter_head = 0
        accum_len = 0
        slice_head = 0
        ret = []
        while iter_head <= self.num_graphs:
            if iter_head == self.num_graphs or (accum_len > 0 and accum_len + len(feature[iter_head]) > iter_step):
                i = slice(slice_head, iter_head)
                ret.append(i)
                accum_len = 0
                slice_head = iter_head
            if iter_head < self.num_graphs:
                accum_len += len(feature[iter_head])
            iter_head += 1
        return ret

    def sample(self):
        raise NotImplementedError

    def get_sample(self):
        if self.cache is None:
            self.cache = self.sample()
        return self.cache

    def __iter__(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class SingleViewSampler(BaseSampler):
    def __init__(self, name, graphs, batch_size):
        self.inner_contrast = single_view_contrast.contrast_dict[name]()
        self.sample_size = 5000  # stay still
        self.graphs = graphs
        self.anchor = self.graphs  # densere aut non densere, illa quaestio
        self.num_graphs = len(graphs)
        self.graphs_diff = []
        self.batch_size = batch_size  # use this!
        self.cache = None
        self.sample_subgraph = True

    def sample(self):
        """
        Performs: random pairing and sampling
        Returns values needed for generating (anchor, pos, neg)
        @return: f_pos, f_neg, e_anchor, e_diff, e_neg, e_diffneg, slices
        """
        g_anchor = self.anchor
        feats, edges = [], []
        list_graphs = list(range(self.num_graphs))
        random.shuffle(list_graphs)
        # sample subgraphs
        for i in list_graphs:
            f1, (e1,), _ = sample_subgraph([g_anchor[i]], len(g_anchor[i].x), self.sample_size, self.sample_subgraph)
            feats.append(f1)
            edges.append(e1)

        # get slices
        slices = self.sample_slicer(feats)
        return feats, edges, slices

    @classmethod
    def process_graphs(cls, feats, edges):
        graphs = []
        for i in range(len(feats)):
            graphs.append(GraphInput(feats[i], None, edges[i]))
        adj, start_idx = process_graphs(graphs, get_device())
        return adj, start_idx

    def __iter__(self):
        feats, edges, slices = self.get_sample()
        for i in slices:  # for each batch
            f_pos_d = feats[i]
            adj, start_idx = self.process_graphs(feats[i], edges[i])
            anchor_nodes, pos_nodes, neg_nodes = self.inner_contrast.adapt(adj.to('cpu'))
            bx = ModelInput(ModelInput.NODES, adj, start_idx, f_pos_d, actual_indices=anchor_nodes)
            bpos = ModelInput(ModelInput.NODES, adj, start_idx, f_pos_d, actual_indices=pos_nodes)
            bneg = ModelInput(ModelInput.NODES, adj, start_idx, f_pos_d, actual_indices=neg_nodes)
            yield [[bx, bpos, bneg]]
        self.cache = None

    def __len__(self):
        _, _, slices = self.get_sample()
        return len(slices)


class MultiViewSampler(BaseSampler):
    def __init__(self, graphs, batch_size, anchor_view, context_view):
        # self.anchor_name, self.pos_name, self.neg_name = name.split('-')

        self.graphs = graphs
        self.sample_size = 5000  # batch_size // 2 # 5000
        for graph in graphs:
            if len(graph.x) > batch_size // 2:
                self.sample_size = batch_size // 2
                break
        self.anchor = self.graphs  # densere aut non densere, illa quaestio
        self.num_graphs = len(graphs)
        self.graphs_diff = []
        self.pre_calculation()
        self.batch_size = batch_size  # use this!
        self.cache = None
        self.sample_subgraph = True
        self.sampled = []
        self.anchor_view = anchor_view
        self.context_view = context_view

    def pre_calculation(self):
        self.graphs_diff = self.anchor

    def get_negative_features(self, f_pos, e_anchor, e_diff):
        f_neg = []
        if self.num_graphs == 1:
            for f in f_pos:
                rp = np.random.permutation(len(f))
                f_neg.append(f[rp])
        else:
            f_neg = f_pos
        return f_neg

    def sample(self):
        """
        Performs: random pairing and sampling
        Returns values needed for generating (anchor, pos, neg)
        @return: f_pos, f_neg, e_anchor, e_diff, e_neg, e_diffneg, slices
        """
        g_anchor, g_diff = self.anchor, self.graphs_diff
        f_pos, e_anchor, e_diff, = [], [], []
        self.sampled = []
        # sample subgraphs
        for i in range(self.num_graphs):
            if len(g_anchor[i].x) <= self.sample_size:
                self.sampled.append(False)
                pnodes = torch.arange(len(g_anchor[i].x))
            else:
                self.sampled.append(True)
                pnodes = None
            f1, (e1, e2), pm = sample_subgraph([g_anchor[i], g_diff[i]], len(g_anchor[i].x), self.sample_size,
                                               self.sample_subgraph, permuted_nodes=pnodes)
            f_pos.append(f1)
            e_anchor.append(e1)
            e_diff.append(e2)

        # get negative
        f_neg = self.get_negative_features(f_pos, e_anchor, e_diff)

        # get slices
        slices = self.sample_slicer(f_pos)
        return f_pos, f_neg, e_anchor, e_diff, slices

    def process_graphs(self, f_pos, f_neg, e_anchor, e_diff, **kwargs):
        raise NotImplementedError

    def __iter__(self):
        """
        create batch samples
        @return:
        """
        f_pos, f_neg, e_anchor, e_diff, slices = self.get_sample()

        for i in slices:  # for each batch
            adj, add, adn, adnd, \
            start_idx, start_idd, start_idn, start_idnd, \
            f_p, f_d, f_n, f_nd,\
            i_p, i_d, i_n, i_nd \
                = self.process_graphs(f_pos[i], f_neg[i], e_anchor[i], e_diff[i])

            bx = ModelInput(self.anchor_view, adj, start_idx, f_p, actual_indices=i_p)
            bpos = ModelInput(self.context_view, add, start_idd, f_d, actual_indices=i_d)
            bneg = ModelInput(self.context_view, adnd, start_idnd, f_nd, actual_indices=i_nd)

            bx_r = ModelInput(self.anchor_view, add, start_idd, f_d, actual_indices=i_d)
            bpos_r = ModelInput(self.context_view, adj, start_idx, f_p, actual_indices=i_p)
            bneg_r = ModelInput(self.context_view, adn, start_idn, f_n, actual_indices=i_n)

            yield [[bx, bpos, bneg]]
            yield [[bx_r, bpos_r, bneg_r]]
        self.cache = None

    def __len__(self):
        """

        @return: number of batches = len(slices) * 2
        """
        _, _, _, _, slices = self.get_sample()
        return len(slices) * 2


class DGISampler(MultiViewSampler):
    def __init__(self, graphs, batch_size):
        super().__init__(graphs, batch_size, anchor_view=ModelInput.GRAPHS, context_view=ModelInput.NODES)

    def process_graphs(self, f_pos, f_neg, e_anchor, e_diff, **kwargs):
        g_anchor, g_neg = [], []
        for i in range(len(f_pos)):
            g_anchor.append(GraphInput(f_pos[i], None, e_anchor[i]))
            g_neg.append(GraphInput(f_neg[i], None, e_anchor[i]))
        pg = process_graphs(g_anchor, get_device())
        adj, start_idx = pg
        adn, start_idn = adj, start_idx  # process_graphs(g_neg, get_device())
        return adj, adj, adn, adn, \
               start_idx, start_idx, start_idn, start_idn, \
               f_pos, f_pos, f_neg, f_neg, \
               None, None, None, None


class DiffSampler(MultiViewSampler):
    def __init__(self, graphs, batch_size):
        super().__init__(graphs, batch_size, anchor_view=ModelInput.GRAPHS,
                         context_view=ModelInput.NODES)

    def pre_calculation(self):
        self.graphs_diff = []
        for g in self.graphs:
            t = compute_ppr(g.edge_index)
            if not isinstance(t, torch.Tensor):
                t = torch.from_numpy(t)
            # convert t into sparse
            idx = torch.nonzero(t).t()
            val = t[idx[0], idx[1]]
            self.graphs_diff.append(GraphInput(g.x, None, idx, val))

    def process_graphs(self, f_pos, f_neg, e_anchor, e_diff, **kwargs):
        g_anchor, g_diff, g_neg, g_negdiff = [], [], [], []
        for i in range(len(f_pos)):
            g_anchor.append(GraphInput(f_pos[i], None, e_anchor[i]))
            g_neg.append(GraphInput(f_neg[i], None, e_anchor[i]))
            g_diff.append(GraphInput(f_pos[i], None, e_diff[i]))
            g_negdiff.append(GraphInput(f_neg[i], None, e_diff[i]))
        pg1 = process_graphs(g_anchor, get_device())
        pg2 = process_graphs(g_diff, get_device(), normalize=False)

        adj, start_idx = pg1
        add, start_idd = pg2
        adn, start_idn = adj, start_idx
        adnd, start_idnd = add, start_idd
        return adj, add, adn, adnd, \
               start_idx, start_idd, start_idn, start_idnd, \
               f_pos, f_pos, f_neg, f_neg, \
               None, None, None, None


class GraphCLSampler(MultiViewSampler):
    def __init__(self, graphs, batch_size):
        super().__init__(graphs, batch_size, anchor_view=ModelInput.GRAPHS, context_view=ModelInput.GRAPHS)

    def process_graphs(self, f_pos, f_neg, e_anchor, e_diff, **kwargs):
        g_anchor, g_diff, g_neg, g_negdiff = [], [], [], []
        graphcl_methods = [mask_attribute, graphclment_subgraph, drop_edges, drop_nodes]
        f_graphcl1, f_graphcl2 = graphcl_methods[torch.randint(len(graphcl_methods), [1]).data], \
                         graphcl_methods[torch.randint(len(graphcl_methods), [1]).data]

        for i in range(len(f_pos)):
            g_p = GraphInput(f_pos[i], None, e_anchor[i])
            g_n = GraphInput(f_neg[i], None, e_anchor[i])
            f_a, e_a = f_graphcl1(g_p)
            f_d, e_d = f_graphcl2(g_p)
            f_n, e_n = f_graphcl1(g_n)
            f_dn, e_dn = f_graphcl2(g_n)

            g_anchor.append(GraphInput(f_a, None, e_a))
            g_diff.append(GraphInput(f_d, None, e_d))
            g_neg.append(GraphInput(f_n, None, e_n))
            g_negdiff.append(GraphInput(f_dn, None, e_dn))

        f_p = [g.x for g in g_anchor]
        f_d = [g.x for g in g_diff]
        f_n = [g.x for g in g_neg]
        f_nd = [g.x for g in g_negdiff]
        adj, start_idx = process_graphs(g_anchor, get_device())
        add, start_idd = process_graphs(g_diff, get_device())
        adn, start_idn = process_graphs(g_neg, get_device())
        adnd, start_idnd = process_graphs(g_negdiff, get_device())

        return adj, add, adn, adnd, \
               start_idx, start_idd, start_idn, start_idnd, \
               f_p, f_d, f_n, f_nd, \
               None, None, None, None


class GCASampler(MultiViewSampler):
    def __init__(self, graphs, batch_size):
        super().__init__(graphs, batch_size, anchor_view=ModelInput.NODES, context_view=ModelInput.NODES)

    def process_graphs(self, f_pos, f_neg, e_anchor, e_diff, **kwargs):
        g_anchor, g_diff = [], []
        graphcl_methods = [drop_edges, mask_attribute]
        f_graphcl1, f_graphcl2 = graphcl_methods[0], graphcl_methods[1]
        for i in range(len(f_pos)):
            g_p = GraphInput(f_pos[i], None, e_anchor[i])
            f_a, e_a = f_graphcl1(g_p)
            f_d, e_d = f_graphcl2(g_p)

            g_anchor.append(GraphInput(f_a, None, e_a))
            g_diff.append(GraphInput(f_d, None, e_d))

        f_p = [g.x for g in g_anchor]

        adj, start_idx = process_graphs(g_anchor, get_device())
        add, start_idd = process_graphs(g_diff, get_device())
        anchor_nodes = torch.arange(len(adj))
        neg_nodes = torch.randint(high=len(adj) - 1, size=(len(anchor_nodes),))
        neg_nodes[neg_nodes >= anchor_nodes] += 1
        idx = torch.arange(len(adj) + 1)
        return adj, add, adj, add, \
               idx, idx, idx, idx, \
               f_p, f_p, f_p, f_p, \
               anchor_nodes, anchor_nodes, anchor_nodes, anchor_nodes

    def get_negative_features(self, f_pos, e_anchor, e_diff):
        return f_pos, e_anchor, e_diff

    def __iter__(self):
        """
        create batch samples
        @return:
        """
        t_tot = 0
        f_pos, f_neg, e_anchor, e_diff, slices = self.get_sample()

        for i in slices:  # for each batch
            adj, add, adn, adnd, \
            start_idx, start_idd, start_idn, start_idnd, \
            f_p, f_d, f_n, f_nd,\
            i_p, i_d, i_n, i_nd \
                = self.process_graphs(f_pos[i], f_neg[i], e_anchor[i], e_diff[i])

            bx = ModelInput(self.anchor_view, adj, start_idx, f_p, actual_indices=i_p)
            bpos = ModelInput(self.context_view, add, start_idd, f_d, actual_indices=i_d)
            bneg = ModelInput(self.context_view, adnd, start_idnd, f_nd, actual_indices=i_nd)

            bx_r = ModelInput(self.anchor_view, add, start_idd, f_d, actual_indices=i_d)
            bpos_r = ModelInput(self.context_view, adj, start_idx, f_p, actual_indices=i_p)
            bneg_r = ModelInput(self.context_view, adn, start_idn, f_n, actual_indices=i_n)

            yield [[bx, bpos, bneg]]
            yield [[bx, bpos, bneg_r]]
            yield [[bx_r, bpos_r, bneg]]
            yield [[bx_r, bpos_r, bneg_r]]
        self.cache = None

    def __len__(self):
        """

        @return: number of batches = len(slices) * 2
        """
        _, _, _, _, slices = self.get_sample()
        return len(slices) * 4


sampler_dict = {
    "dgi": DGISampler,
    "mvgrl": DiffSampler,
    "graphcl": GraphCLSampler,
    "gca": GCASampler
}
