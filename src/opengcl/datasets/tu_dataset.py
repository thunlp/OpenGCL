from abc import ABC

from torch_geometric.datasets import TUDataset
from .base_graph import Adapter
from sklearn.model_selection import train_test_split
import numpy as np
import networkx as nx
from ..utils import process_graphs, torch_sparse_to_scipy_coo
import torch
from tqdm import tqdm


class Graphs(Adapter, ABC):

    def __init__(self, dataset_name, **kwargs):
        super(Graphs, self).__init__(TUDataset, self.root_dir, name=dataset_name)

        for kw in set(kwargs):
            self.__setattr__(kw, kwargs.get(kw))
        self.num = len(self.data)


    def load_data(self):
        self.debug("[datasets] Load data.", flush=True)
        if not self.attributed():
            self.debug("[datasets] Not self.attributed(): set attribute as 1", flush=True)

            class Data:
                def __init__(self, x, edge_index, y):
                    self.x = x
                    self.edge_index = edge_index
                    self.y = y

            data = []
            for g in tqdm(self.data):
                data.append(Data(torch.ones(int(torch.max(g.edge_index)) + 1, 1), g.edge_index, g.y))
            self.data = data
        for graph in self.data:
            try:
                assert graph.edge_index.max() < len(graph.x)
            except AssertionError:
                print("[exception] graph.edge_idx:", graph.edge_index.max(), "len=", len(graph.x))
                exit(1)

        self.ngraph = len(self.data)
        # print("process graphs...")
        adj, start_idx = process_graphs(self.data)
        # print("...graphs processed")

        self.set_g(nx.from_scipy_sparse_matrix(torch_sparse_to_scipy_coo(adj)))  # TODO: format conversion
        # print("g set (?)")

        feat = torch.cat([g.x for g in self.data]).numpy()

        # print("feat set")

        self.set_node_features(featurevectors=feat)

        # print("node features set")

        self.set_node_label(torch.arange(start_idx[-1]).reshape([-1, 1]).tolist())

        # print("node label set")

        self.start_idx = start_idx


        # print("loaddata() finished.")

    def labels(self):
        return self.data, [g.y.tolist() for g in self.data]

    def get_split_data(self, train_percent=None, validate_percent=None, validate_size=None, seed=None):
        """
        TODO: confirm X_train and X_test
        @param train_percent:
        @param validate_percent:
        @param validate_size:
        @param seed:
        @return:
        """
        train_idx, test_idx = train_test_split(np.arange(self.num), train_size=train_percent, random_state=seed,
                                               shuffle=True)
        train_idx = torch.from_numpy(train_idx)
        test_idx = torch.from_numpy(test_idx)
        X_train = train_idx.tolist()
        X_test = test_idx.tolist()
        Y_train = [self.data[int(i)].y.tolist() for i in train_idx]
        Y_test = [self.data[int(i)].y.tolist() for i in test_idx]
        # print("TRAIN", Y_train)
        return X_train, Y_train, None, None, X_test, Y_test


class MUTAG(Graphs):
    def __init__(self, **kwargs):
        super(MUTAG, self).__init__('MUTAG', **kwargs)

    @classmethod
    def weighted(cls):
        return True

    @classmethod
    def attributed(cls):
        return True

    @classmethod
    def directed(cls):
        return True


"""
PTC-MR, IMDB-BIN, IMDB-MULTI, REDDIT-BIN
"""


class PTC_MR(Graphs):
    def __init__(self, **kwargs):
        super(PTC_MR, self).__init__('PTC_MR', **kwargs)

    @classmethod
    def weighted(cls):
        return True

    @classmethod
    def attributed(cls):
        return True

    @classmethod
    def directed(cls):
        return True


class IMDB_BINARY(Graphs):
    def __init__(self, **kwargs):
        super(IMDB_BINARY, self).__init__('IMDB-BINARY', **kwargs)

    @classmethod
    def weighted(cls):
        return True

    @classmethod
    def attributed(cls):
        return False

    @classmethod
    def directed(cls):
        return True


class IMDB_MULTI(Graphs):
    def __init__(self, **kwargs):
        super(IMDB_MULTI, self).__init__('IMDB-MULTI', **kwargs)

    @classmethod
    def weighted(cls):
        return True

    @classmethod
    def attributed(cls):
        return False

    @classmethod
    def directed(cls):
        return True


class REDDIT_BINARY(Graphs):
    def __init__(self, **kwargs):
        super(REDDIT_BINARY, self).__init__('REDDIT-BINARY', **kwargs)

    @classmethod
    def weighted(cls):
        return True

    @classmethod
    def attributed(cls):
        return False

    @classmethod
    def directed(cls):
        return True
