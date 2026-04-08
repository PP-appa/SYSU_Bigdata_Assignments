import argparse

from sklearn.svm import LinearSVC

from src.data import load_imdb_data
from src.evaluate import (
    classification_metrics,
    compute_confusion_matrix,
    save_confusion_matrix_figure,
)
from src.features import build_tfidf_features
from src.utils import save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TF-IDF + LinearSVC on IMDB.")
    parser.add_argument("--max-features", type=int, default=20000)
    parser.add_argument("--max-iter", type=int, default=3000)
    parser.add_argument("--metrics-path", type=str, default="outputs/metrics/svm_metrics.json")
    parser.add_argument(
        "--cm-fig-path",
        type=str,
        default="outputs/figures/svm_confusion_matrix.png",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 1) Load raw IMDB data (same split as LR script)
    train_texts, train_labels, test_texts, test_labels = load_imdb_data(lowercase=True)

    # 2) Convert text to TF-IDF sparse features (same feature extractor as LR)
    x_train, x_test, _ = build_tfidf_features(
        train_texts=train_texts,
        test_texts=test_texts,
        max_features=args.max_features,
    )

    # 3) Train Linear SVM classifier
    model = LinearSVC(max_iter=args.max_iter, random_state=42)
    model.fit(x_train, train_labels)

    # 4) Evaluate on test set
    y_pred = model.predict(x_test)
    metrics = classification_metrics(test_labels, y_pred)
    cm = compute_confusion_matrix(test_labels, y_pred)

    print("Evaluation Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    print("Confusion Matrix:")
    print(cm)

    # 5) Save outputs
    metrics_to_save = {**metrics, "confusion_matrix": cm}
    save_json(metrics_to_save, args.metrics_path)
    save_confusion_matrix_figure(cm, args.cm_fig_path)

    print(f"\nMetrics saved to: {args.metrics_path}")
    print(f"Confusion matrix figure saved to: {args.cm_fig_path}")


if __name__ == "__main__":
    main()
