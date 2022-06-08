import numpy as np
from collections import Counter


def euclidian_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


class k_nearest_neighbors:
    def __init__(self, k):
        self.k = k

    def knn_fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def knn_predict(self, X):
        predicted_labels = [self._predict(x) for x in X]
        return np.array(predicted_labels)

    def _predict(self, x):
        distance = [euclidian_distance(x, x_train) for x_train in self.x_train]
        k_indices = np.argsort(distance)[: self.k]
        k_nerest_labels = [self.y_train[i] for i in k_indices]
        majority_vote = Counter(k_nerest_labels).most_common(1)
        return majority_vote[0][0]
