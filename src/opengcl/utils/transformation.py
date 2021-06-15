import time

import scipy.sparse as sp
import torch
import numpy as np

def scipy_coo_to_torch_sparse(scipy_sparse_coo):
    values = scipy_sparse_coo.data
    indices = np.vstack((scipy_sparse_coo.row, scipy_sparse_coo.col))
    v = torch.tensor(values, dtype=torch.float32)
    i = torch.tensor(indices, dtype=torch.long)
    return torch.sparse_coo_tensor(i, v, scipy_sparse_coo.shape)


def torch_sparse_to_scipy_coo(torch_sparse):
    # print("torch_sparse_to_scipy_coo...")
    t1 = time.time()
    a = torch_sparse.coalesce()
    (i, j), v = a.indices().numpy(), a.values().numpy()

    ret = sp.coo_matrix((v, (i, j)), shape=a.shape)
    # print("time =", time.time() - t1)
    return ret


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = features.sum(1).clamp(min=1)
    r_inv = (rowsum ** -1).flatten()
    r_inv[torch.isinf(r_inv)] = 0.
    # print(r_inv.shape)
    features = r_inv.unsqueeze(1) * features
    return features


def normalize_adj(adj):  # safe. don't change by now
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    # print(rowsum.sum())
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj, normalize=True):
    """Preprocessing of adjacency matrix for simple GCN framework and conversion to tuple representation."""
    adj = adj + sp.eye(adj.shape[0])
    if normalize:
        adj = normalize_adj(adj)
    else:
        adj = (adj).tocoo()
    return scipy_coo_to_torch_sparse(adj)


# from https://github.com/weihua916/powerful-gnns
def process_graphs(graphs, device=None, normalize=True):
    """

    @param device:
    @param graphs: a list of graphs, with .x, .y, .edge_index, some with .edge_weight
    @return: joined adj and start_idx ( start_idx[i] == starting node of graph i; len(start_idx) == num_graph + 1 )
    """
    edge_mat_list = []
    elems = []
    start_idx = [0]
    # print("process", len(graphs), "graphs")
    for i, graph in enumerate(graphs):
        # check
        # assert graph.edge_index.max() < len(graph.x)
        start_idx.append(start_idx[i] + len(graph.x))
        edge_mat_list.append(start_idx[i] + graph.edge_index)
        if hasattr(graph, 'edge_weight') and graph.edge_weight is not None:
            elems.append(graph.edge_weight)
    Adj_block_idx = torch.cat(edge_mat_list, 1)

    if len(elems) == 0:
        Adj_block_elem = torch.ones(Adj_block_idx.shape[1])
    else:
        Adj_block_elem = torch.cat(elems)
    # Adj_block = torch.sparse.FloatTensor(Adj_block_idx, Adj_block_elem, torch.Size([start_idx[-1],start_idx[-1]]))
    Adj_block = sp.coo_matrix((Adj_block_elem, Adj_block_idx), shape=(start_idx[-1], start_idx[-1]))

    Adj_block = preprocess_adj(Adj_block, normalize)

    if device is not None:
        Adj_block = Adj_block.to(device)
    # print(Adj_block.device)
    return Adj_block, start_idx
