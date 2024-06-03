import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import rbf_kernel


class ClusterSimilarity(BaseEstimator, TransformerMixin):
    def __init__(
        self, n_clusters: int = 10, gamma: float = 1, random_state: int = 42
    ) -> None:
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y=None, sample_weight=None) -> "ClusterSimilarity":
        check_array(X)
        self.n_features_in_ = X.shape[1]
        self.kmeans_ = KMeans(
            n_clusters=self.n_clusters, random_state=self.random_state, n_init=10
        )
        self.kmeans_.fit(X, sample_weight=sample_weight)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self)
        check_array(X)
        return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)

    def get_feature_names_out(self, names=None) -> list[str]:
        return [f"Cluster {i} similarity" for i in range(self.n_clusters)]


class CustomPCA(BaseEstimator, TransformerMixin):
    def __init__(self, standardise: bool = True, n_components: int = 0.95) -> None:
        self.standardise = standardise
        self.n_components = n_components

    def fit(self, X: pd.DataFrame, y=None) -> "CustomPCA":
        check_array(X)
        self.n_features_in_ = X.shape[1]
        self.pca_ = PCA(n_components=self.n_components)
        self.pca_.fit(X)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self)
        check_array(X)

        if self.standardise:
            X = (X - X.mean()) / X.std(axis=0)

        return self.pca_.transform(X)

    def get_feature_names_out(self, names=None) -> list[str]:
        return [f"PC {i}" for i in range(self.pca_.n_components_)]
