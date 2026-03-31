import numpy as np
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from scipy.optimize import linear_sum_assignment


def clustering_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute clustering accuracy by optimal label mapping (Hungarian algorithm)."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    # Ignore noise points for accuracy calculation (DBSCAN label -1)
    valid = y_pred != -1
    y_true = y_true[valid]
    y_pred = y_pred[valid]
    if len(y_true) == 0:
        return 0.0

    true_labels = np.unique(y_true)
    pred_labels = np.unique(y_pred)

    cost = np.zeros((len(true_labels), len(pred_labels)), dtype=int)
    for i, t in enumerate(true_labels):
        for j, p in enumerate(pred_labels):
            cost[i, j] = np.sum((y_true == t) & (y_pred == p))

    row_ind, col_ind = linear_sum_assignment(cost.max() - cost)
    correct = cost[row_ind, col_ind].sum()
    return float(correct / len(y_true))


def evaluate_clustering(x: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    x = np.asarray(x)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    valid = y_pred != -1
    x_eval = x[valid]
    y_pred_eval = y_pred[valid]

    n_clusters = len(set(y_pred_eval.tolist()))
    if len(y_pred_eval) < 2 or n_clusters < 2:
        sil = float("nan")
        ch = float("nan")
    else:
        sil = float(silhouette_score(x_eval, y_pred_eval))
        ch = float(calinski_harabasz_score(x_eval, y_pred_eval))

    return {
        "accuracy": clustering_accuracy(y_true, y_pred),
        "silhouette": sil,
        "calinski_harabasz": ch,
        "n_clusters": n_clusters,
        "n_noise": int(np.sum(y_pred == -1)),
    }
