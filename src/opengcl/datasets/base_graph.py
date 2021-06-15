"""
refer to Dataset of CogDL
"""
from abc import ABC

from torch.utils.data import Dataset
import os.path as osp
import networkx as nx
import numpy as np
import torch
import os
from ..utils import files_exist, download_url, makedirs


class Graph(Dataset, ABC):
    def __init__(self, resource_url, root_dir, name_dict, **kwargs):
        """
        :param resource_url:
        :param root_dir:
        :param filenames: local filenames
        """
        super(Graph, self).__init__()
        self.resource_url = resource_url
        self.dir = root_dir
        self.G = None
        self.look_up_dict = {}
        self.look_back_list = []
        self.name_dict = name_dict
        self.ngraph = 1
        for kw in set(kwargs):
            self.__setattr__(kw, kwargs.get(kw))

        self.filenames = [i for k, i in name_dict.items()]
        self.paths = [self.full(f) for f in self.filenames]
        rootprompt = ""
        if self.dir:
            rootprompt = "from root dir: {}".format(osp.abspath(self.dir))
        self.debug("[datasets] Loading {} Dataset {}".format(type(self).__name__, rootprompt),
                   flush=True)
        self.load_data()
        self.debug(f"[datasets] This is a {self.modifier()} dataset with {self.ngraph} "
                   f"{'graph' if self.ngraph == 1 else 'graphs'}, "
                   f"{self.nodesize} nodes and {self.edgesize} edges.",
                   flush=True)

    def modifier(self):
        if self.nodesize < 2000:
            scale = 'small'
        elif self.nodesize < 10000:
            scale = 'medium-sized'
        else:
            scale = 'large'
        edge_density = self.edgesize * self.ngraph / (self.nodesize ** 2)
        if edge_density < 0.01 and self.edgesize < self.nodesize * 15:
            density = 'sparse'
        elif edge_density < 0.25:
            density = 'medium-dense'
        else:
            density = 'dense'
        return scale + ' ' + density


    def __getitem__(self, item):
        return self.G

    def load_data(self):
        if not files_exist(self.paths):
            if self.resource_url is None:
                errmsg = '\n'.join([f for f in self.paths if osp.exists(f)])
                raise FileNotFoundError("[exception] Cannot find required files:\n{}".format(errmsg))
            makedirs(self.dir)
            self.debug('[datasets] Downloading dataset "{}" from "{}".\n'
                       '[datasets] Files will be saved to "{}".'.format(type(self).__name__, self.resource_url, self.dir))
            self.download()
            self.debug('[datasets] Dataset successfully downloaded.')
        self.read()

    def download(self):
        for name in self.filenames:
            download_url('{}/{}'.format(self.resource_url, name), self.dir)

    @property
    def read_operation(self):
        return {'edgefile': self.read_edgelist,
                'adjfile': self.read_adjlist,
                'labelfile': self.read_node_label,
                'features': self.read_node_features,
                'status': self.read_node_status}

    def read(self):
        name_dict = self.name_dict
        for k, v in name_dict.items():
            if k in self.read_operation:
                self.read_operation[k](self.full(v))

    def full(self, filename):
        if self.dir:
            return osp.join(self.dir, filename)
        return filename

    @classmethod
    def directed(cls):
        raise NotImplementedError

    @classmethod
    def weighted(cls):
        raise NotImplementedError

    @classmethod
    def attributed(cls):
        raise NotImplementedError

    def features(self):
        if not self.attributed():
            return np.ones((self.G.number_of_nodes(), 1), dtype=np.float32)
        return np.vstack([self.G.nodes[self.look_back_list[i]]['feature']
                          for i in range(self.G.number_of_nodes())])

    def setfeatures(self, features):
        for i in range(self.G.number_of_nodes()):
            self.G.nodes[self.look_back_list[i]]['feature'] = features[i]

    def adjmat(self, directed=True, weighted=True, scaled=None, sparse=False):
        """
        Returns a **numpy array**.
        @param directed:
        @param weighted:
        @param scaled:
        @param sparse:
        @return:
        """
        G = self.G
        if type(self).directed() and not directed:
            G = nx.to_undirected(G)
        A = nx.adjacency_matrix(G).astype(np.float32)  # a csr sparse matrix
        if not sparse:
            A = np.array(nx.adjacency_matrix(G).todense())
        if not weighted:
            A = A.astype(np.bool).astype(np.float32)
        if scaled is not None:  # e.g. scaled = 1
            # print(scaled)
            A = A / A.sum(scaled, keepdims=True)
        return A

    def labels(self):
        X = []
        Y = []
        for i in self.G.nodes:
            X.append(i)
            Y.append(self.G.nodes[i]['label'])
        return X, Y

    @property
    def nodesize(self):
        return self.G.number_of_nodes()

    @property
    def edgesize(self):
        return self.G.number_of_edges()

    def encode_node(self):
        # self.debug("Encoding nodes...")
        look_up = self.look_up_dict
        look_back = self.look_back_list
        look_up.clear()
        look_back.clear()
        for node in (self.G.nodes()):
            look_up[node] = len(look_back)
            look_back.append(node)
            self.G.nodes[node]['status'] = ''

    def set_g(self, g):
        self.G = g
        self.encode_node()

    def read_adjlist(self, filename):
        """ Read graph from adjacency file in which the edge must be unweighted
            the format of each line: v1 n1 n2 n3 ... nk
            :param filename: the filename of input file
        """
        self.G = nx.read_adjlist(filename, create_using=nx.DiGraph())
        for i, j in self.G.edges():
            self.G[i][j]['weight'] = 1.0
        self.encode_node()

    def read_edgelist(self, filename):
        self.G = nx.DiGraph()

        if self.directed():
            def read_unweighted(l):
                src, dst = l.split()
                self.G.add_edge(src, dst)
                self.G[src][dst]['weight'] = 1.0

            def read_weighted(l):
                src, dst, w = l.split()
                self.G.add_edge(src, dst)
                self.G[src][dst]['weight'] = float(w)
        else:
            def read_unweighted(l):
                src, dst = l.split()
                self.G.add_edge(src, dst)
                self.G.add_edge(dst, src)
                self.G[src][dst]['weight'] = 1.0
                self.G[dst][src]['weight'] = 1.0

            def read_weighted(l):
                src, dst, w = l.split()
                self.G.add_edge(src, dst)
                self.G.add_edge(dst, src)
                self.G[src][dst]['weight'] = float(w)
                self.G[dst][src]['weight'] = float(w)
        fin = open(filename, 'r')
        func = read_unweighted
        if self.weighted():
            func = read_weighted
        while 1:
            l = fin.readline()
            if l == '':
                break
            func(l)
        fin.close()
        self.encode_node()

    # use after G is not none
    def read_node_label(self, filename):
        fin = open(filename, 'r')
        while 1:
            l = fin.readline()
            if l == '':
                break
            vec = l.split()
            self.G.nodes[vec[0]]['label'] = vec[1:]
        fin.close()

    # use after G is not none
    def set_node_label(self, labelvectors, split=False):
        for i, vec in enumerate(labelvectors):
            if split:
                self.G.nodes[vec[0]]['label'] = vec[1:]
            else:
                self.G.nodes[i]['label'] = vec

    # use after G is not none
    def read_node_features(self, filename):
        fin = open(filename, 'r')
        for l in fin.readlines():
            vec = l.split()
            self.G.nodes[vec[0]]['feature'] = np.array([float(x) for x in vec[1:]])
        fin.close()

    # use after G is not none
    def set_node_features(self, featurevectors, split=False):
        for i, vec in enumerate(featurevectors):
            if split:
                self.G.nodes[vec[0]]['feature'] = vec[1:]
            else:
                # print(i)
                self.G.nodes[i]['feature'] = vec

    # use after encode_node()
    def read_node_status(self, filename):
        fin = open(filename, 'r')
        while 1:
            l = fin.readline()
            if l == '':
                break
            vec = l.split()
            self.G.nodes[vec[0]]['status'] = vec[1]  # train test valid
        fin.close()

    def read_edge_label(self, filename):
        fin = open(filename, 'r')
        while 1:
            l = fin.readline()
            if l == '':
                break
            vec = l.split()
            self.G[vec[0]][vec[1]]['label'] = vec[2:]
        fin.close()

    def set_edge_attr(self, edgelist, edgeattrvectors):
        for i in range(len(edgelist)):
            self.G[edgelist[i][0]][edgelist[i][1]]['feature'] = edgeattrvectors[i]

    def _split(self, train_percent, validate_percent=0, validate_size=None, seed=None):
        """
            split dataset
            if validate_size is assigned then validate_percent will be disabled
            returns X_train, ..., val..., X_test, Y_test
            self.X_train, Y_train... can only be accessed after calling this
            do not call this directly
            call this only if you want to do a force re-split to the dataset;
            otherwise call get_split_data()
        """
        assert train_percent + validate_percent < 1
        X, Y = self.labels()
        state = torch.random.get_rng_state()
        training_size = int(train_percent * len(X))
        if validate_size is not None:
            if training_size < validate_size * 2:  # training set too small
                validate_size = training_size // 2  # force 50%
            training_size -= validate_size
        else:
            validate_size = int(validate_percent * len(X))
        if seed is not None:
            torch.random.manual_seed(seed)
        shuffle_indices = torch.randperm(len(X))
        self.shuffle_indices = shuffle_indices
        X_train = [X[shuffle_indices[i]] for i in range(training_size)]
        Y_train = [Y[shuffle_indices[i]] for i in range(training_size)]
        X_val = [X[shuffle_indices[i + training_size]] for i in range(validate_size)]
        Y_val = [Y[shuffle_indices[i + training_size]] for i in range(validate_size)]
        X_test = [X[shuffle_indices[i]] for i in range(training_size + validate_size, len(X))]
        Y_test = [Y[shuffle_indices[i]] for i in range(training_size + validate_size, len(X))]
        self.train_percent = train_percent
        self.validate_percent = validate_percent
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_val = X_val
        self.Y_val = Y_val
        self.X_test = X_test
        self.Y_test = Y_test

        def sample_mask(begin, end):
            mask = torch.zeros(self.nodesize)
            for i in range(begin, end):
                mask[shuffle_indices[i]] = 1
            return mask

        self.train_mask = sample_mask(0, len(self.X_train))
        self.val_mask = sample_mask(len(self.X_train), len(self.X_train) + len(self.X_val))
        self.test_mask = sample_mask(len(self.X_train) + len(self.X_val), self.nodesize)
        torch.random.set_rng_state(state)
        # print(X_train[:20])
        return X_train, Y_train, X_val, Y_val, X_test, Y_test

    def get_split_data(self, train_percent=None, validate_percent=None, validate_size=None, seed=None):
        """
            if validate_size is assigned then validate_percent will be disabled
            call this if you only want to get certain split.
            if you want to resplit, call resplit()
        """
        # print("SPLIT...")
        if hasattr(self, 'X_train') and hasattr(self, 'train_percent') and \
                (train_percent is None or train_percent == self.train_percent) and \
                (validate_percent is None or validate_percent == self.validate_percent):
            # print("SPLIT_ALREADY_DONE")
            return self.X_train, self.Y_train, self.X_val, self.Y_val, self.X_test, self.Y_test
        if validate_percent is None:
            validate_percent = 0
        return self._split(train_percent=train_percent, validate_percent=validate_percent,
                           validate_size=validate_size, seed=seed)

    def resplit(self, train_percent, validate_percent=0, validate_size=0, seed=None):
        """
        if validate_size is assigned then validate_percent will be disabled
        """
        return self._split(train_percent=train_percent, validate_percent=validate_percent,
                           validate_size=validate_size, seed=seed)

    def debug(self, *args, **kwargs):
        if not getattr(self, 'silent', False):
            print(*args, **kwargs)



class Adapter(Graph, ABC):
    def __init__(self, AdapteeClass, *args, **kwargs):
        self.data = AdapteeClass(*args, **kwargs)
        super(Adapter, self).__init__(None, self.root_dir, {}, **kwargs)

    @property
    def root_dir(self):
        return osp.join('..', 'data', type(self).__name__)

    def download(self):
        pass
