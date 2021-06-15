import numpy as np
import scipy.sparse
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score, accuracy_score  # data process
from sklearn.preprocessing import MultiLabelBinarizer  # data process
import warnings
warnings.filterwarnings("ignore")

class TopKRanker(OneVsRestClassifier):
    def __init__(self, estimator):
        super(TopKRanker, self).__init__(estimator)
        self.top_k_list = None

    def predict(self, X):
        probs = super(TopKRanker, self).predict_proba(X)  # assume X as a Tensor
        all_labels = []
        for i, k in enumerate(self.top_k_list):
            probs_i = probs[i, :]
            labels = self.classes_[probs_i.argsort()[-k:]].tolist()
            probs_i[:] = 0
            probs_i[labels] = 1  # mark top k labels
            all_labels.append(probs_i)
        return np.stack(all_labels)


class Classifier(object):
    def __init__(self, vectors, clf):
        self.embeddings = vectors
        self.clf = TopKRanker(clf)
        self.binarizer = MultiLabelBinarizer(sparse_output=True)

    def train(self, X, Y, Y_all):
        self.binarizer.fit(Y_all)
        X_train = np.stack([self.embeddings[x] for x in X])
        Y = self.binarizer.transform(Y)  # lhs Y a numpy array
        self.clf.fit(X_train, Y)

    def evaluate(self, X, Y):  # X Y tensor
        top_k_list = [len(l) for l in Y]
        y_pred = self.predict(X, top_k_list)  # Y_ Tensor
        y_true = self.binarizer.transform(Y)  # Y  np array
        result = f1_score(y_true, y_pred, average="micro")
        return result

    def predict(self, X, top_k_list):
        X_ = np.stack([self.embeddings[x] for x in X])
        self.clf.top_k_list = top_k_list
        Y = self.clf.predict(X_)
        return Y

    def train_and_evaluate(self, graph, train_percent, seed=None):
        X_train, Y_train, _, _, X_test, Y_test = graph.get_split_data(train_percent, seed=seed)
        self.train(X_train, Y_train, graph.labels()[1])
        return self.evaluate(X_test, Y_test)
