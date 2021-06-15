import random
import time
import torch
from .walker import Walker, BasicWalker
from ...utils import scipy_coo_to_torch_sparse, get_device


def random_walk_sample(graph):
    dw = True
    p = q = 0.5
    path_length = 30
    num_paths = 5
    window = 5
    if dw:
        walker = BasicWalker(graph)
    else:
        walker = Walker(graph, p=p, q=q)
    sentences = walker.simulate_walks(num_walks=num_paths, walk_length=path_length)
    # sample sentences
    sentences = random.sample(sentences, len(sentences) // path_length // window)
    anchor = []
    positive = []
    for sentence in sentences:
        for j in range(len(sentence)):
            try:
                k = random.randint(max(0, j - window), min(len(sentence) - 1, j + window) - 1)
                if k >= j:
                    k += 1
                anchor.append(sentence[j])
                positive.append(sentence[k])
            except Exception:
                pass
    anchor = torch.tensor(anchor)
    positive = torch.tensor(positive)
    return anchor, positive


def random_except_neighbor(rand_high, excepts, adj, adj_list):
    # generate adjacency list
    w1 = {}
    #  for each anchor, generate a random node that is not neighbor
    #  (using binary search)

    # pre-generate a rand list; especially useful for sparse graphs
    # for a graph with 50000 nodes and 100000 edges,
    # binary search is very rarely performed
    ys = torch.randint(0, rand_high, [len(excepts)])
    for idx, x in (enumerate(excepts)):  # , total=len(excepts)):
        x = int(x)
        # try random node first
        y = ys[idx]
        if not adj[x, y]:
            continue

        # generate k-th node that is not neighbor
        if x not in w1:
            w1[x] = sorted(adj_list[x])
        adj = w1[x]
        if len(adj) == rand_high:  # connected to all nodes
            ys[idx] = -1
        i = int(torch.randint(0, rand_high - len(adj), [1]))  # rand one out of neighborhood
        if i < adj[0]:
            ys[idx] = i
        elif i >= adj[-1] - len(adj) + 1:
            ys[idx] = i + len(adj)
        else:
            # binary search to get i-th element out of neighborhood
            l = 0
            r = len(adj) - 1
            while l < r:
                mid = (l + r) // 2
                if adj[mid] - mid > i:
                    r = mid - 1
                else:
                    l = mid + 1
            ys[idx] = i + l
    return ys


class SingleViewContrast:
    def __init__(self, name):
        self.pos_name, self.neg_name = name.lower().split('-')
        self.negative_ratio = 1
        self.adj = None
        self.adj_ind = self.n_nodes = self.n_edges = self.graph = self.adj_list = self.node_list = None
        self.anchor = self.positive = self.negative = None

    def adapt(self, adj):
        self.adj = adj
        self.adj_ind = self.adj._indices()
        self.n_nodes = adj.shape[0]
        self.n_edges = adj._nnz()
        self.graph = adj  # nx.DiGraph(adj)
        self.adj_list = {i: [] for i in range(self.n_nodes)}
        for i in range(adj._nnz()):
            x, y = adj._indices()[0, i].item(), adj._indices()[1, i].item()
            self.adj_list[x].append(y)
        self.node_list = [i for i in range(self.n_nodes)]
        self.generate()
        return self.anchor.to(get_device()), \
               self.positive.to(get_device()), \
               self.negative.to(get_device())

    def generate(self):
        self.sample_positive()
        self.sample_noises()

    def sample_positive(self):
        if self.pos_name == 'neighbor':
            self.anchor = self.adj_ind[0]
            self.positive = self.adj_ind[1]
        elif self.pos_name == 'self':
            self.anchor = torch.arange(self.n_nodes)
            self.positive = self.anchor
        elif self.pos_name == 'rand_walk':
            self.anchor, self.positive = random_walk_sample(self.graph)

    def sample_noises(self):  # called after self.anchor is created
        self.anchor = self.anchor.repeat(self.negative_ratio)
        self.positive = self.positive.repeat(self.negative_ratio)
        if self.neg_name == 'random':
            self.negative = torch.randint(high=self.n_nodes, size=(len(self.anchor),))
        elif self.neg_name == 'except_self':
            self.negative = torch.randint(high=self.n_nodes - 1, size=(len(self.anchor),))
            self.negative[self.negative >= self.anchor] += 1
        elif self.neg_name == 'except_neighbor':  # anchor must be node
            self.negative = random_except_neighbor(self.n_nodes, self.anchor, self.adj, self.adj_list)


class NRContrast(SingleViewContrast):
    def __init__(self):
        super().__init__(name='neighbor-random')


class NEContrast(SingleViewContrast):
    def __init__(self):
        super().__init__(name='neighbor-except_neighbor')


class RRContrast(SingleViewContrast):
    def __init__(self):
        super().__init__(name='rand_walk-random')


class REContrast(SingleViewContrast):
    def __init__(self):
        super().__init__(name='rand_walk-except_neighbor')


contrast_dict = {
    "snr": NRContrast,  # s for single-view
    "sne": NEContrast,
    "srr": RRContrast,
    "sre": REContrast,
}
