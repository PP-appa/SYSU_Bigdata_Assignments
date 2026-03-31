import csv
from pathlib import Path

from main import load_iris, run_dbscan, run_kmeans


def run_sweep() -> None:
    x, y = load_iris()

    output_dir = Path(__file__).resolve().parents[1] / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    k_values = [2, 3, 4, 5]
    eps_values = [0.4, 0.5, 0.6, 0.7, 0.8]
    min_samples_values = [3, 5, 8]

    km_rows = []
    for k in k_values:
        _, m = run_kmeans(x, y, k=k, max_iter=300, tol=1e-4)
        km_rows.append(
            {
                "k": k,
                "accuracy": m["accuracy"],
                "silhouette": m["silhouette"],
                "calinski_harabasz": m["calinski_harabasz"],
                "n_clusters": m["n_clusters"],
                "n_noise": m["n_noise"],
            }
        )

    db_rows = []
    for eps in eps_values:
        for min_samples in min_samples_values:
            _, m = run_dbscan(x, y, eps=eps, min_samples=min_samples)
            db_rows.append(
                {
                    "eps": eps,
                    "min_samples": min_samples,
                    "accuracy": m["accuracy"],
                    "silhouette": m["silhouette"],
                    "calinski_harabasz": m["calinski_harabasz"],
                    "n_clusters": m["n_clusters"],
                    "n_noise": m["n_noise"],
                }
            )

    km_path = output_dir / "kmeans_sweep.csv"
    with km_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["k", "accuracy", "silhouette", "calinski_harabasz", "n_clusters", "n_noise"],
        )
        writer.writeheader()
        writer.writerows(km_rows)

    db_path = output_dir / "dbscan_sweep.csv"
    with db_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["eps", "min_samples", "accuracy", "silhouette", "calinski_harabasz", "n_clusters", "n_noise"],
        )
        writer.writeheader()
        writer.writerows(db_rows)

    print(f"Saved: {km_path}")
    print(f"Saved: {db_path}")


if __name__ == "__main__":
    run_sweep()
