import torch
import numpy as np
from typing import Union, List
from scipy.linalg import inv
from ...utils import get_device, GraphInput


def compute_ppr(edge_index, alpha=0.2, self_loop=True):
    adj = torch.sparse_coo_tensor(edge_index, torch.ones(edge_index.shape[1])).to_dense()
    if adj.shape[0] > 20000:
        a = adj.cpu().numpy()
        if self_loop:
            a = a + np.eye(a.shape[0])  # A^ = A + I_n
        d = (np.sum(a, 1))  # D^ = Sigma A^_ii
        dinv = np.power(d, -0.5)  # D^(-1/2)
        at = (dinv.reshape(-1, 1) * a) * (dinv.reshape(1, -1))  # A~ = D^(-1/2) x A^ x D^(-1/2)

        return alpha * inv((np.eye(a.shape[0]) - (1 - alpha) * at))  # a(I_n-(1-a)A~)^-1
    else:
        a = adj.to(get_device())
        eye = torch.eye(a.shape[0], device=get_device())
        if self_loop:
            a.add_(eye)
        d = torch.sum(a, 1)
        dinv = torch.pow(d, -0.5)
        at = (dinv.reshape((-1, 1)) * a) * (dinv.reshape((1, -1)))
        del a
        return (alpha * torch.inverse((eye - (1 - alpha) * at))).cpu()


def sample_subgraph(graph_data: List[GraphInput], n_nodes=None,
                    max_graph_size=None, do_sample=True, permuted_nodes=None):
    """

    @param do_sample: if False, return original graph.
    @param graph_data: a list of N graphs, [anchor, ...], which share the same features and node sampling.
    @param n_nodes: number of nodes in any graph.
    @param max_graph_size: threshold of sampling.
    @param permuted_nodes: optional, provide a permutation
    @return: sub_feats of graph1, [sub_edges of grpah1, sub_edges of graph2...]
    """

    if n_nodes is None:
        n_nodes = len(graph_data[0].x)

    if max_graph_size is None:
        max_graph_size = n_nodes * 1 // 5

    # sample node list
    if do_sample:
        if permuted_nodes is None:
            permuted_nodes = torch.randperm(n_nodes)
    else:
        permuted_nodes = torch.arange(n_nodes)
        max_graph_size = int(1e9)

    map_idx = torch.argsort(permuted_nodes)
    sub_nodes = permuted_nodes[:max_graph_size]

    feats = graph_data[0].x

    sub_feats = feats[sub_nodes].to(get_device())

    def sample_edges(edges):
        _sub_edges = map_idx[edges]  # permutation
        e_sample = (_sub_edges[0] < max_graph_size) * (_sub_edges[1] < max_graph_size)

        return _sub_edges[:, e_sample]

    sub_edges = [sample_edges(graph.edge_index) for graph in graph_data]
    return sub_feats, sub_edges, permuted_nodes


# https://github.com/Shen-Lab/GraphCL/blob/master/unsupervised_TU/graphcl.py
def graphclment_subgraph(graph_data: GraphInput, max_graph_size=None):
    n_nodes = len(graph_data.x)
    if max_graph_size is None:
        max_graph_size = n_nodes * 1 // 5
    sub_nodes = [torch.randint(n_nodes, [1]).item()]
    selected = torch.zeros(n_nodes, dtype=torch.bool)
    visited = torch.zeros(n_nodes, dtype=torch.bool)
    edge_index = graph_data.edge_index
    candidate_nodes: List = edge_index[1, edge_index[0] == sub_nodes[0]].tolist()
    selected[sub_nodes] = True
    visited[candidate_nodes] = True
    visited[sub_nodes] = True
    cnt = 0
    while len(sub_nodes) <= max_graph_size:
        cnt += 1
        if cnt > n_nodes:
            break
        if len(candidate_nodes) == 0:
            break
        idx = torch.randint(len(candidate_nodes), [1]).item()
        sample_node = candidate_nodes[idx]
        selected[sample_node] = True
        candidate_nodes[idx] = candidate_nodes[-1]
        candidate_nodes.pop(-1)
        sub_nodes.append(sample_node)
        new_candidates = edge_index[1, edge_index[0] == sample_node]
        new_candidates = new_candidates[visited[new_candidates] == False].tolist()
        visited[new_candidates] = True
        candidate_nodes.extend(new_candidates)
    sub_size = len(sub_nodes)
    permuted_nodes = sub_nodes + [i for i in range(n_nodes) if not selected[i]]
    # print("__", sub_size, max_graph_size)
    sub_feats, sub_edges, _ = sample_subgraph([graph_data], n_nodes, sub_size,
                                              permuted_nodes=torch.tensor(permuted_nodes))
    # print("__", n_nodes, sub_size, sub_nodes, sub_edges[0].shape, flush=True)
    return sub_feats, sub_edges[0]


def drop_edges(graph: GraphInput, max_edge_size=None):
    """
    @param graph: input graph
    @param max_edge_size: threshold (0.8*n_edges by default)
    @return: feats & edges
    """
    n_edges = graph.edge_index.shape[1]
    if max_edge_size is None:
        max_edge_size = n_edges * 4 // 5
    keep_edges = torch.randperm(n_edges)[:max_edge_size]
    # torch.multinomial(torch.arange(n_edges), max_edge_size,
    #                               replacement=True)
    sub_edges = graph.edge_index[:, keep_edges]
    return graph.x, sub_edges


def drop_nodes(graph: GraphInput, max_graph_size=None):
    """
    @param graph: input graph
    @param max_graph_size: threshold
    @return: feats & edges
    """
    n_nodes = len(graph.x)
    if max_graph_size is None:
        max_graph_size = n_nodes * 4 // 5
    subfeats, sub_edges, _ = sample_subgraph([graph], max_graph_size=max_graph_size)
    return subfeats, sub_edges[0]


def mask_attribute(graph: GraphInput, mask_size=None,
                   mask_using: Union[None, torch.Tensor] = None):
    n_nodes = len(graph.x)
    if mask_size is None:
        mask_size = n_nodes * 1 // 5
    mask = torch.randperm(n_nodes)[:mask_size]
    # torch.multinomial(torch.arange(n_nodes), mask_size,
    #                         replacement=True)
    # print(mask)
    feats = torch.clone(graph.x)
    if mask_using is None:
        feats[mask] = torch.randn_like(graph.x[mask]) * .5 + .5
    else:
        feats[mask] = mask_using
    return feats, graph.edge_index
