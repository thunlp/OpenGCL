from typing import List
import numpy as np
from sklearn.linear_model import LogisticRegressionCV

from .classify import Classifier
from ..utils import process_graphs, preprocess_features, get_device, debug, \
    ModelInput, HyperParameter, ModuleParameter, GraphInput
from .framework import Framework
import math
import torch.cuda
from torch.cuda.amp import autocast
from time import time


class Trainer(torch.nn.Module):

    def __init__(self, dataset, module_params: ModuleParameter, hyper_params: HyperParameter):
        t_start = time()
        debug('[GCL] Building framework... ', end='')
        super().__init__()
        self.module_params = module_params
        self.hyper_params = hyper_params
        self.dataset = dataset
        # Create framework
        self.model = Framework(dataset=dataset, module_params=module_params, hyper_params=hyper_params)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hyper_params.learning_rate)

        # early stops
        self.cost_val = []
        self.c_up = 0
        self.emergency_stop = False
        debug(f'done in {time() - t_start}s.')
        t_start = time()
        self._do_training()
        debug(f'[GCL] Framework training finished in {time() - t_start}s.')

    def _do_training(self):
        self.to(get_device())
        self.train(True)
        epochs = self.hyper_params.epochs
        debug(f"[GCL] Start training for {epochs} epochs...")
        time_start = time()
        for i in range(epochs):
            debug_info = self._step()
            current_total = i + 1
            if current_total % 5 == 0:
                time_end = time()
                debug(f'[GCL]   epoch {current_total}: '
                      f'{debug_info}; '
                      f'time used = {time_end - time_start}s')
                time_start = time_end
            if self._early_stopping_judge():
                debug("[GCL] Early stopping condition satisfied. Abort training.")
                break
        self.train(False)

    def _step(self):
        # Train framework
        output, train_loss = self.evaluate()
        self.cost_val.append(train_loss)
        if math.isnan(train_loss):
            exit(1)
        if get_device() != torch.device('cpu'):
            debug_info = f"loss: {'{:.5f}'.format(train_loss)}; Allocated: {torch.cuda.memory_allocated()}; " \
                              f"Reserved: {getattr(torch.cuda, 'memory_reserved', torch.cuda.memory_cached)()}"
        else:
            debug_info = f"loss: {'{:.5f}'.format(train_loss)}"
        if np.isnan(train_loss):
            self.emergency_stop = True
        return debug_info

    def evaluate(self):
        cur_loss = 0.
        output = None
        if self.training:
            self.optimizer.zero_grad()
        batch_num = len(self.model.sampler)
        for batch in self.model.sampler:
            loss = torch.tensor(0., dtype=torch.float32, device=get_device())
            for bx, bpos, bneg in batch:
                loss += self.model(bx, bpos, bneg)

            loss /= batch_num
            if self.training:
                loss.backward()
            cur_loss += loss.item()
            if get_device() != torch.device('cpu'):
                torch.cuda.empty_cache()

        if self.training:
            self.optimizer.step()

        return output, cur_loss

    def _early_stopping_judge(self):
        if self.emergency_stop:
            return True
        if self.hyper_params.patience > len(self.cost_val) - self.hyper_params.early_stopping:
            return False
        if self.cost_val[-1] > self.cost_val[-2]:
            self.c_up += 1
            if self.c_up >= self.hyper_params.patience:
                return True
        else:
            self.c_up = 0
        return False

    def get_embeddings(self, input_flag):
        self.requires_grad_(False)  # detach
        graphs_data: List[GraphInput] = self.model.graphs_data  #
        slices = self.model.sampler.sampler.sample_slicer([g.x for g in graphs_data])
        embeddings = []
        processed_nodes = 0
        for i in slices:
            adj, start_idx = process_graphs(graphs_data[i], get_device())
            feature_slice = slice(processed_nodes, processed_nodes+start_idx[-1])
            all_graphs = ModelInput(input_flag, adj, start_idx, [self.model.features[feature_slice]], repeat=False)
            processed_nodes += start_idx[-1]
            if self.module_params.enc == 'gat' and adj.shape[0] > 10000:  # autocast to save space
                with autocast():
                    sub_embeddings = self.model.embed(all_graphs).detach()
            else:
                sub_embeddings = self.model.embed(all_graphs).detach()
            embeddings.append(sub_embeddings)
        return torch.cat(embeddings).cpu()

    def get_graph_embeddings(self):
        return self.get_embeddings(ModelInput.GRAPHS)

    def get_node_embeddings(self):
        return self.get_embeddings(ModelInput.NODES)

    def _get_vectors(self, input_flag):
        embeddings = self.get_embeddings(input_flag)
        vectors = {}
        for i, embedding in enumerate(embeddings):
            vectors[i] = embedding
        return vectors

    def classify(self, input_flag):
        clf = Classifier(vectors=self._get_vectors(input_flag),
                         clf=LogisticRegressionCV(cv=5, random_state=0))

        X_train, Y_train, _, _, X_test, Y_test = self.dataset.get_split_data(self.hyper_params.clf_ratio, seed=0)
        clf.train(X_train, Y_train, self.dataset.labels()[1])
        return clf.evaluate(X_test, Y_test)
