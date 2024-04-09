from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np


class MKNN:
    """
    Modified K-Nearest Neighbors classifier.

    Parameters:
    -----------
    n_neighbors : int, optional (default=5)
        Number of neighbors to use for classification.
    distance : str, optional (default='euclidean')
        Distance metric to use. Supported metrics are 'euclidean', 'manhattan', 'chebyshev', and 'minkowski'.

    Attributes:
    -----------
    n_neighbors : int
        Number of neighbors used for classification.
    distance : str
        Distance metric used for classification.
    model : sklearn.neighbors.KNeighborsClassifier
        K-Nearest Neighbors classifier model.

    Methods:
    --------
    fit(X, y)
        Fit the model according to the given training data.
    get_distances(X)
        Get distances of the nearest neighbors of input samples.
    score(X_test, y_true)
        Return the mean accuracy on the given test data and labels.
    knn(X)
        Predict the class labels for the provided data.
    compare_with_knn(X, y)
        Compare the performance of MKNN with standard KNN.
    predict(X)
        Predict the class labels for the provided data.

    Raises:
    -------
    ValueError
        If 'X' is not fitted before calling 'get_distances', or if 'X' has invalid dimensionality.

    """

    def __init__(self, n_neighbors: int = 5, distance: str = "euclidean"):
        """
        Initialize MKNN with the given parameters.

        Parameters:
        -----------
        n_neighbors : int, optional (default=5)
            Number of neighbors to use for classification.
        distance : str, optional (default='euclidean')
            Distance metric to use. Supported metrics are 'euclidean', 'manhattan', 'chebyshev', and 'minkowski'.
        """
        self.n_neighbors = n_neighbors
        self.distance = distance
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors, metric=distance)

    def __str__(self):
        return "Modified K-Nearest Neighbors"

    def __validate(X, y, model):
        neighbors = model.kneighbors(X)[1]
        validate = []
        for i, v in enumerate(neighbors):
            selected_ = y[v]
            if np.bincount(selected_).argmax() == y[i]:
                validate.append(1)
            else:
                validate.append(0)
        return validate

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the model according to the given training data.

        Parameters:
        -----------
        X : numpy.ndarray
            Training data.
        y : numpy.ndarray
            Target values.

        Returns:
        --------
        self : MKNN
            Returns an instance of self.
        """
        self.X = X
        self.y = y
        self.model.fit(X, y)
        self.validate = MKNN.__validate(X, y, model=self.model)
        return self

    def get_distances(self, X):
        """
        Get distances of the nearest neighbors of input samples.

        Parameters:
        -----------
        X : numpy.ndarray
            Input samples.

        Returns:
        --------
        numpy.ndarray
            Distances of the nearest neighbors.
        """
        if not hasattr(self, "X"):
            raise ValueError("do a fit first !")
        return self.model.kneighbors(X)[0]

    def __weight(val, idx, dist):
        idx_val = [
            (i, val[i], dist[j], val[i] * dist[j] + 0.5) for j, i in enumerate(idx)
        ]
        weight = [(i[0], i[-1]) for i in idx_val]
        sorted_weight = sorted(weight, key=lambda x: x[1], reverse=True)[0]
        return sorted_weight

    def score(self, X_test: np.ndarray, y_true: np.ndarray):
        """
        Return the mean accuracy on the given test data and labels.

        Parameters:
        -----------
        X_test : numpy.ndarray
            Test samples.
        y_true : numpy.ndarray
            True labels for X_test.

        Returns:
        --------
        float
            Mean accuracy.
        """
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
        """
        Predict the class labels for the provided data.

        Parameters:
        -----------
        X : numpy.ndarray
            Input samples.

        Returns:
        --------
        numpy.ndarray
            Predicted class labels.
        """
        X = MKNN.__X_test(X, self.X)
        pred = self.model.predict(X)
        return pred

    def compare_with_knn(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Compare the performance of MKNN with standard KNN.

        Parameters:
        -----------
        X : numpy.ndarray
            Input samples.
        y : numpy.ndarray
            Target values.

        Returns:
        --------
        dict
            A dictionary containing MKNN and KNN accuracies.
        """
        X = MKNN.__X_test(X, self.X)
        mknn = self.score(X, y)
        knn = self.model.score(X, y)
        return {"MKNN": mknn, "KNN": knn}

    def predict(self, X: np.ndarray):
        """
        Predict the class labels for the provided data.

        Parameters:
        -----------
        X : numpy.ndarray
            Input samples.

        Returns:
        --------
        numpy.ndarray
            Predicted class labels.
        """
        X = MKNN.__X_test(X, self.X)
        dist, idx = self.model.kneighbors(X)
        val = self.validate
        y_pred = []
        for i, v in enumerate(idx):
            weight = MKNN.__weight(val, v, dist[i])
            pred = self.y[weight[0]]
            y_pred.append(pred)
        return np.array(y_pred)
