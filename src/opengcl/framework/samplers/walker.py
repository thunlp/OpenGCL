import random
import torch
import math
import multiprocessing
from time import time
from ...datasets.base_graph import Graph
from typing import Union


def wrapper(class_instance, epoch, walk_length):
    return class_instance.simulate_walks_one_epoch(walk_length)


class BasicWalker:
    def __init__(self, G: Union[Graph, torch.sparse.Tensor]):
        if isinstance(G, Graph):
            self.G = G.G   # nx.DiGraph(G.G)
            self.adjlist = None
            self.edgelist = None
            self.adj = None
            self.node_size = G.nodesize
            self.look_up_dict = G.look_up_dict
        elif isinstance(G, torch.sparse.Tensor):
            self.adjlist = {i: [] for i in range(G.shape[0])}
            self.edgelist = []
            # create adj list by brute force
            for i in range(G._nnz()):
                x, y = G._indices()[0, i].item(), G._indices()[1, i].item()
                self.adjlist[x].append(y)
                self.edgelist.append([x, y])
            self.adj = G
            self.G = None
            self.node_size = G.shape[0]
            self.look_up_dict = {i: i for i in range(self.node_size)}
        self.workers = None  # workers

    def rwalk(self, walk_length, start_node):
        """
        Simulate a random walk starting from start node.
        """
        G = self.G
        adjlist = self.adjlist
        # look_up_dict = self.look_up_dict
        # node_size = self.node_size

        walk = [start_node]
        while len(walk) < walk_length:
            cur = walk[-1]
            if adjlist is not None:
                cur_nbrs = adjlist[cur]
            else:
                cur_nbrs = list(G.neighbors(cur))
            if len(cur_nbrs) > 0:
                walk.append(random.choice(cur_nbrs))
            else:
                break

        return walk

    def simulate_walks_one_epoch(self, walk_length):
        if self.G:
            nodes = list(self.G.nodes())
        else:
            nodes = list(range(self.node_size))
        walks = []
        random.shuffle(nodes)
        for node in nodes:
            walks.append(self.rwalk(
                    walk_length=walk_length, start_node=node))
        return walks

    def simulate_walks(self, num_walks, walk_length):
        """
        Repeatedly simulate random walks from each node.
        """

        walks = []


        if self.workers:
            pool = multiprocessing.Pool(self.workers)

            walks_res = []
            for walk_iter in range(num_walks):
                walks_res.append(pool.apply_async(wrapper, args=(self, walk_iter, walk_length, )))

            pool.close()
            pool.join()

            for w in walks_res:
                walks.extend(w.get())

        else:
            for walk_iter in range(num_walks):
                walks.extend(self.simulate_walks_one_epoch(walk_length))

        # print(len(walks))
        return walks


class Walker(BasicWalker):
    def __init__(self, G, p, q):
        super(Walker, self).__init__(G)
        self.p = p
        self.q = q

    def rwalk(self, walk_length, start_node):
        """
        Simulate a random walk starting from start node.
        """
        G = self.G
        adjlist = self.adjlist
        alias_nodes = self.alias_nodes
        alias_edges = self.alias_edges
        look_up_dict = self.look_up_dict
        node_size = self.node_size

        walk = [start_node]

        while len(walk) < walk_length:
            cur = walk[-1]
            if G:
                cur_nbrs = list(G.neighbors(cur))
            else:
                cur_nbrs = adjlist[cur]
            if len(cur_nbrs) > 0:
                if len(walk) == 1:
                    walk.append(
                        cur_nbrs[alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])])
                else:
                    prev = walk[-2]
                    pos = (prev, cur)
                    nxt = cur_nbrs[alias_draw(alias_edges[pos][0], alias_edges[pos][1])]
                    walk.append(nxt)
            else:
                break
        walk = [str(i) for i in walk]
        return walk

    def get_alias_edge(self, src, dst):
        """
        Get the alias edge setup lists for a given edge.
        """
        G = self.G
        adjlist = self.adjlist
        p = self.p
        q = self.q

        unnormalized_probs = []
        if G:
            dst_nbrs = list(G.neighbors(dst))
        else:
            dst_nbrs = adjlist[dst]
        for dst_nbr in dst_nbrs:
            if G:
                weight = G[dst][dst_nbr]['weight']
            else:
                weight = self.adj[dst, dst_nbr].item()
            if dst_nbr == src:
                unnormalized_probs.append(weight/p)
            elif G.has_edge(dst_nbr, src):
                unnormalized_probs.append(weight)
            else:
                unnormalized_probs.append(weight/q)
        norm_const = sum(unnormalized_probs)
        normalized_probs = [
            float(u_prob)/norm_const for u_prob in unnormalized_probs]

        return alias_setup(normalized_probs)

    def preprocess_transition_probs(self):
        """
        Preprocessing of transition probabilities for guiding the random walks.
        """
        G = self.G
        adjlist = self.adjlist

        alias_nodes = {}
        if self.G:
            nodes = list(self.G.nodes())
        else:
            nodes = list(range(self.node_size))
        for node in nodes:
            if G:
                nbrs = list(G.neighbors(node))
            else:
                nbrs = adjlist[node]
            unnormalized_probs = [G[node][nbr]['weight'] if G else self.adj[node, nbr].item()
                                  for nbr in nbrs]
            norm_const = sum(unnormalized_probs)
            normalized_probs = [
                float(u_prob)/norm_const for u_prob in unnormalized_probs]
            alias_nodes[node] = alias_setup(normalized_probs)

        alias_edges = {}
        triads = {}

        look_up_dict = self.look_up_dict
        node_size = self.node_size
        if self.edgelist:
            edges = self.edgelist
        else:
            edges = G.edges()
        for edge in edges:
            alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])

        self.alias_nodes = alias_nodes
        self.alias_edges = alias_edges

        return


def alias_setup(probs):
    """
    Compute utility lists for non-uniform sampling from discrete distributions.
    Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    for details
    """
    K = len(probs)
    q = [0 for i in range(K)] # torch.zeros(K, dtype=torch.float32) # np.zeros(K, dtype=np.float32)
    J = [0.0 for i in range(K)] # np.zeros(K, dtype=np.int32)

    smaller = []
    larger = []
    for kk, prob in enumerate(probs):
        q[kk] = K*prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)

    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return J, q

# from https://github.com/Stonesjtu/Pytorch-NCE
def alias_draw(J, q, *size):
    """
    Draw sample from a non-uniform discrete distribution using alias sampling.
    """
    if len(size) == 0:
        K = len(J)
        kk = int(math.floor(random.random()*K))

        if random.random() < q[kk]:
            return kk
        else:
            return J[kk]

    q = torch.tensor(q)
    J = torch.tensor(J)
    max_value = q.size()[0]

    kk = q.new(*size).random_(0, max_value).long().view(-1)
    prob = J[kk]
    alias = q[kk]
    # b is whether a random number is greater than q
    b = torch.bernoulli(prob).long()
    oq = kk.mul(b)
    oj = alias.mul(1 - b)

    return (oq + oj).view(size)
