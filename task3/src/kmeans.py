import numpy as np


class KMeansCustom:
    def __init__(
        self,
        n_clusters: int = 3,
        max_iter: int = 300,
        tol: float = 1e-4,
        random_state: int | None = 42,
    ) -> None:
        if n_clusters <= 0:
            raise ValueError("n_clusters must be > 0")
        if max_iter <= 0:
            raise ValueError("max_iter must be > 0")
        if tol < 0:
            raise ValueError("tol must be >= 0")

        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

        self.cluster_centers_: np.ndarray | None = None
        self.labels_: np.ndarray | None = None
        self.inertia_: float | None = None
        self.n_iter_: int = 0

    def _init_centroids(self, x: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng(self.random_state)
        n_samples = x.shape[0]
        if self.n_clusters > n_samples:
            raise ValueError("n_clusters cannot be greater than number of samples")

        # k-means++ initialization
        centroids = np.empty((self.n_clusters, x.shape[1]), dtype=float)
        first_idx = rng.integers(0, n_samples)
        centroids[0] = x[first_idx]

        for i in range(1, self.n_clusters):
            d2 = np.min(((x[:, None, :] - centroids[None, :i, :]) ** 2).sum(axis=2), axis=1)
            probs = d2 / d2.sum()
            next_idx = rng.choice(n_samples, p=probs)
            centroids[i] = x[next_idx]
        return centroids

    @staticmethod
    def _assign_labels(x: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        # Euclidean distance to each centroid
        distances = np.sqrt(((x[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2))
        return np.argmin(distances, axis=1)

    def fit(self, x: np.ndarray) -> "KMeansCustom":
        if x.ndim != 2:
            raise ValueError("x must be a 2D array")
        x = x.astype(float, copy=False)

        centroids = self._init_centroids(x)

        for i in range(1, self.max_iter + 1):
            labels = self._assign_labels(x, centroids)
            new_centroids = np.empty_like(centroids)

            for k in range(self.n_clusters):
                members = x[labels == k]
                if len(members) == 0:
                    # Reinitialize empty cluster with a random sample
                    rng = np.random.default_rng(self.random_state + i + k if self.random_state is not None else None)
                    new_centroids[k] = x[rng.integers(0, x.shape[0])]
                else:
                    new_centroids[k] = members.mean(axis=0)

            shift = np.sqrt(((new_centroids - centroids) ** 2).sum(axis=1)).max()
            centroids = new_centroids
            self.n_iter_ = i

            if shift <= self.tol:
                break

        self.cluster_centers_ = centroids
        self.labels_ = self._assign_labels(x, centroids)
        self.inertia_ = float(((x - centroids[self.labels_]) ** 2).sum())
        return self

    def fit_predict(self, x: np.ndarray) -> np.ndarray:
        self.fit(x)
        assert self.labels_ is not None
        return self.labels_
