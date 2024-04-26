from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np


class MKNN:
    """
        Modified K-Nearest Neighbors (MKNN)

    This module implements a modified version of the K-Nearest Neighbors (KNN) algorithm for classification tasks.

    Attributes:
        n_neighbors (int): Number of neighbors to consider.
        distance (str): Distance metric to use. Default is "euclidean".

    Methods:
        - __init__(self, n_neighbors: int = 5, distance: str = "euclidean"): Initializes the MKNN classifier.
        - fit(self, X: np.ndarray, y: np.ndarray) -> object: Fits the MKNN classifier to the training data.
        - predict(self, X: np.ndarray) -> np.ndarray: Predicts the class labels for the input data.
        - score(self, X_test: np.ndarray, y_true: np.ndarray) -> float: Computes the accuracy score of the classifier.
        - compare_with_knn(self, X: np.ndarray, y: np.ndarray) -> dict: Compares the performance of MKNN with standard KNN.
        - knn(self, X: np.ndarray) -> np.ndarray: Predicts class labels using the underlying KNN model.
        - weight_preview(self, X: np.ndarray, y: Optional[np.ndarray] = None): Displays the weights and predictions for the input data.
        - get_params(self) -> dict: Returns the parameters of the MKNN model.
        - get_distances_indices(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]: Returns distances and indices of nearest neighbors for input data.

    Notes:
        - This implementation modifies the standard KNN by introducing a weight factor for each neighbor based on the distance and validation against nearest neighbors.
        - The distance metric used can be specified during initialization. Supported metrics include "euclidean", "manhattan", "chebyshev", "minkowski", and more.
        - Before calling predict, fit method should be called with training data.
        - To compare the performance of MKNN with standard KNN, use compare_with_knn method.

    Raises:
        - ValueError: If fit is called before initializing the classifier, or if X_test has incompatible dimensions with X_train.

    """

    def __init__(self, n_neighbors: int = 5, distance: str = "euclidean"):
        self.n_neighbors = n_neighbors
        self.distance = distance
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors, metric=distance)

    def __str__(self):
        return "Modified K-Nearest Neighbors"

    def get_params(self):
        return self.model.get_params()

    def __validate(X, y, model):
        indeces = model.kneighbors(X, n_neighbors=2)[1]
        validate = []
        for v in indeces:
            label_neighbors = y[v[1]]
            label = y[v[0]]
            if label_neighbors == label:
                validate.append(1)
            else:
                validate.append(0)
        return validate

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.X = X
        self.y = y
        self.model.fit(X, y)
        self.validate = MKNN.__validate(X, y, model=self.model)
        return self

    def get_distances_indices(self, X):
        if not hasattr(self, "X"):
            raise ValueError("do a fit first !")
        return self.model.kneighbors(X)

    def weight_preview(self, X, y=None):
        dist, indices = self.model.kneighbors(X)
        validity = self.validate
        y_train = self.y
        for i, v in enumerate(indices):
            if y is None:
                actual = None
            else:
                actual = y[i]

            dist_ = dist[i]
            w_ = 0
            idx = 0
            for j, k in enumerate(v):
                val = validity[k]
                d = dist_[j]
                w = val * 1 / (d + 0.5)
                if w > w_:
                    w_ = w
                    idx = k
                print(
                    f"idx : {k} | val : {val} | d : {d} | weight : {w} | label : {y_train[k]} | actual : {actual}"
                )
            print(f"Max Weight : {w_} | predicted : {y_train[idx]}")
            print()

    def __weight(val, idx, dist):
        idx_val = [
            (i, val[i], dist[j], val[i] * 1 / (dist[j] + 0.5))
            for j, i in enumerate(idx)
        ]
        weight = [(i[0], i[-1]) for i in idx_val]
        sorted_weight = sorted(weight, key=lambda x: x[1], reverse=True)[0]
        return sorted_weight

    def score(self, X_test: np.ndarray, y_true: np.ndarray):
        pred = self.predict(X_test)
        return accuracy_score(y_true=y_true, y_pred=pred)

    def __X_test(X: np.ndarray, X_train: np.ndarray):
        if X.ndim > X_train.ndim:
            raise ValueError("Undefined for sequences of unequal length.")
        if X.ndim == 1:
            if len(X) != X_train.shape[1]:
                raise ValueError("Undefined for sequences of unequal length.")
            X = np.expand_dims(X, axis=True)
        return X

    def knn(self, X: np.ndarray):
        X = MKNN.__X_test(X, self.X)
        pred = self.model.predict(X)
        return pred

    def compare_with_knn(self, X: np.ndarray, y: np.ndarray) -> dict:
        X = MKNN.__X_test(X, self.X)
        mknn = self.score(X, y)
        knn = self.model.score(X, y)
        return {"MKNN": mknn, "KNN": knn}

    def predict(self, X: np.ndarray):
        X = MKNN.__X_test(X, self.X)
        dist, idx = self.model.kneighbors(X)
        val = self.validate
        y_pred = []
        for i, v in enumerate(idx):
            indeces = MKNN.__weight(val, v, dist[i])
            pred = self.y[indeces[0]]
            y_pred.append(pred)
        return np.array(y_pred)
