import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

from dbscan import DBSCANCustom
from kmeans import KMeansCustom
from metrics_custom import evaluate_clustering


def load_iris() -> tuple[np.ndarray, np.ndarray]:
    iris = datasets.load_iris()
    x = iris.data
    y = iris.target
    x = StandardScaler().fit_transform(x)
    return x, y


def run_kmeans(x: np.ndarray, y: np.ndarray, k: int, max_iter: int, tol: float) -> tuple[np.ndarray, dict]:
    model = KMeansCustom(n_clusters=k, max_iter=max_iter, tol=tol, random_state=42)
    labels = model.fit_predict(x)
    metrics = evaluate_clustering(x, y, labels)
    return labels, metrics


def run_dbscan(x: np.ndarray, y: np.ndarray, eps: float, min_samples: int) -> tuple[np.ndarray, dict]:
    model = DBSCANCustom(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(x)
    metrics = evaluate_clustering(x, y, labels)
    return labels, metrics


def plot_clusters(x: np.ndarray, labels: np.ndarray, title: str, save_path: Path) -> None:
    plt.figure(figsize=(6, 5))
    unique_labels = sorted(set(labels.tolist()))
    for lab in unique_labels:
        mask = labels == lab
        if lab == -1:
            plt.scatter(x[mask, 0], x[mask, 1], s=35, c="k", marker="x", label="noise")
        else:
            plt.scatter(x[mask, 0], x[mask, 1], s=35, label=f"cluster {lab}")
    plt.title(title)
    plt.xlabel("feature 1 (standardized)")
    plt.ylabel("feature 2 (standardized)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Task3 clustering comparison on Iris dataset")
    parser.add_argument("--k", type=int, default=3, help="number of clusters for KMeans")
    parser.add_argument("--max_iter", type=int, default=300)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--eps", type=float, default=0.6, help="eps for DBSCAN")
    parser.add_argument("--min_samples", type=int, default=5, help="min_samples for DBSCAN")
    args = parser.parse_args()

    x, y = load_iris()

    output_dir = Path(__file__).resolve().parents[1] / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    km_labels, km_metrics = run_kmeans(x, y, args.k, args.max_iter, args.tol)
    db_labels, db_metrics = run_dbscan(x, y, args.eps, args.min_samples)

    result = {
        "kmeans": {"params": {"k": args.k, "max_iter": args.max_iter, "tol": args.tol}, "metrics": km_metrics},
        "dbscan": {"params": {"eps": args.eps, "min_samples": args.min_samples}, "metrics": db_metrics},
    }

    result_path = output_dir / "metrics.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    plot_clusters(x, km_labels, f"KMeans (k={args.k})", output_dir / "kmeans_scatter.png")
    plot_clusters(
        x,
        db_labels,
        f"DBSCAN (eps={args.eps}, min_samples={args.min_samples})",
        output_dir / "dbscan_scatter.png",
    )

    print("=== KMeans Metrics ===")
    for k, v in km_metrics.items():
        print(f"{k}: {v}")
    print("=== DBSCAN Metrics ===")
    for k, v in db_metrics.items():
        print(f"{k}: {v}")
    print(f"Saved metrics to: {result_path}")


if __name__ == "__main__":
    main()
