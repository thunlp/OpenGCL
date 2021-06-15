class InputParameter:
    def __init__(self, **kwargs):
        s1 = self.params()
        for k, v in s1.items():
            self.__setattr__(k, v)
        for k, v in kwargs.items():
            k = k.replace('-', '_')
            if k in s1:
                self.__setattr__(k, v)

    def params(self):
        ret = {}
        for k, v in self.__class__.__dict__.items():
            if callable(v):
                continue
            if isinstance(k, str) and k.startswith('__'):
                continue
            ret[k] = v
        for k, v in self.__dict__.items():
            ret[k] = v
        return ret


class ModuleParameter(InputParameter):
    enc = 'gcn'
    readout = 'mean'
    dec = 'inner'
    sampler = 'dgi'
    est = 'jsd'


class HyperParameter(InputParameter):
    output_dim = 64
    hidden_size = 0
    learning_rate = 0.01
    early_stopping = 20
    patience = 3
    clf_ratio = 0.8
    epochs = 500
    batch_size = 4096
    dropout = 0.
    ff = True


class ModelInput:
    GRAPHS = "graphs"
    NODES = "nodes"
    TYPES = [GRAPHS, NODES]

    def __init__(self, typ, adj, start_idx, feature, repeat=False, num_graphs=1, actual_indices=None):
        """
        batch graph input
        @param typ: "graphs"/"nodes"
        @param graphs: a collection
        @param feature: collection of feat
        @param repeat: (only for typ graphs) if True, embedding of graph will be repeated (num_node) times
        """
        self.typ = typ
        assert self.typ in self.TYPES
        self.adj = adj
        self.start_idx = start_idx
        self.feat = feature
        self.repeat = repeat
        self.num_graphs = num_graphs
        self.actual_indices = actual_indices


class GraphInput:
    def __init__(self, x, y, edge_idx, edge_weight=None):
        """

        @param x:
        @param y:
        @param edge_idx:
        @param edge_weight: None (all 1) by default
        """
        self.x = x
        self.y = y
        self.edge_index = edge_idx
        self.edge_weight = edge_weight
