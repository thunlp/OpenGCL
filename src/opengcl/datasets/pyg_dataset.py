from abc import ABC

from torch_geometric.datasets import CitationFull as pyg_CitationFull, \
    WikiCS as pyg_WikiCS, Coauthor as pyg_Coauthor, Amazon as pyg_Amazon
from .base_graph import Adapter
import numpy as np
import networkx as nx
import torch
from tqdm import tqdm


class PyG(Adapter, ABC):
    def __init__(self, dataset_class, dataset_args, **kwargs):
        super().__init__(dataset_class, self.root_dir, **dataset_args)
        for kw in set(kwargs):
            self.__setattr__(kw, kwargs.get(kw))
        # self.num = len(self.data)

    def load_data(self):
        self.debug("[datasets] Load data.", flush=True)
        if not self.attributed():
            self.debug("[datasets] Not self.attributed(): set attribute as 1", flush=True)
        self.data = self.data[0]
        new_graph = nx.DiGraph()
        new_graph.add_nodes_from(np.arange(len(self.data.x)))
        new_graph.add_edges_from(self.data.edge_index.t().numpy())
        self.set_g(new_graph)

        feat = self.data.x.numpy()

        self.set_node_features(featurevectors=feat)

        self.set_node_label(self.data.y.reshape(-1,1).tolist())



class DBLP(PyG):
    def __init__(self, **kwargs):
        super().__init__(pyg_CitationFull, {'name':'dblp'}, **kwargs)
    @classmethod
    def weighted(cls):
        return False

    @classmethod
    def attributed(cls):
        return True

    @classmethod
    def directed(cls):
        return True


class Coauthor_CS(PyG):
    def __init__(self, **kwargs):
        super().__init__(pyg_Coauthor, {'name':'cs'}, **kwargs)

    @classmethod
    def weighted(cls):
        return False

    @classmethod
    def attributed(cls):
        return True

    @classmethod
    def directed(cls):
        return True

class Coauthor_Phy(PyG):
    def __init__(self, **kwargs):
        super().__init__(pyg_Coauthor, {'name':'physics'}, **kwargs)

    @classmethod
    def weighted(cls):
        return False

    @classmethod
    def attributed(cls):
        return True

    @classmethod
    def directed(cls):
        return True

class WikiCS(PyG):
    def __init__(self, **kwargs):
        super().__init__(pyg_WikiCS, {}, **kwargs)

    @classmethod
    def weighted(cls):
        return False

    @classmethod
    def attributed(cls):
        return True

    @classmethod
    def directed(cls):
        return True

class Amazon_Computers(PyG):
    def __init__(self, **kwargs):
        super().__init__(pyg_Amazon, {'name':'computers'}, **kwargs)

    @classmethod
    def weighted(cls):
        return False

    @classmethod
    def attributed(cls):
        return True

    @classmethod
    def directed(cls):
        return True


class Amazon_Photo(PyG):
    def __init__(self, **kwargs):
        super().__init__(pyg_Amazon, {'name':'photo'}, **kwargs)

    @classmethod
    def weighted(cls):
        return False

    @classmethod
    def attributed(cls):
        return True

    @classmethod
    def directed(cls):
        return True
