import numpy as np


class DBSCANCustom:
    def __init__(self, eps: float = 0.5, min_samples: int = 5) -> None:
        if eps <= 0:
            raise ValueError("eps must be > 0")
        if min_samples <= 0:
            raise ValueError("min_samples must be > 0")
        self.eps = eps
        self.min_samples = min_samples
        self.labels_: np.ndarray | None = None

    def _region_query(self, x: np.ndarray, idx: int) -> np.ndarray:
        distances = np.sqrt(((x - x[idx]) ** 2).sum(axis=1))
        return np.where(distances <= self.eps)[0]

    def fit(self, x: np.ndarray) -> "DBSCANCustom":
        if x.ndim != 2:
            raise ValueError("x must be a 2D array")
        x = x.astype(float, copy=False)

        n_samples = x.shape[0]
        labels = np.full(n_samples, -1, dtype=int)  # -1 means noise
        visited = np.zeros(n_samples, dtype=bool)
        cluster_id = 0

        for i in range(n_samples):
            if visited[i]:
                continue
            visited[i] = True

            neighbors = self._region_query(x, i)
            if len(neighbors) < self.min_samples:
                labels[i] = -1
                continue

            labels[i] = cluster_id
            seeds = list(neighbors)
            j = 0
            while j < len(seeds):
                point_idx = seeds[j]
                if not visited[point_idx]:
                    visited[point_idx] = True
                    point_neighbors = self._region_query(x, point_idx)
                    if len(point_neighbors) >= self.min_samples:
                        for pn in point_neighbors:
                            if pn not in seeds:
                                seeds.append(int(pn))

                if labels[point_idx] == -1:
                    labels[point_idx] = cluster_id
                j += 1

            cluster_id += 1

        self.labels_ = labels
        return self

    def fit_predict(self, x: np.ndarray) -> np.ndarray:
        self.fit(x)
        assert self.labels_ is not None
        return self.labels_
