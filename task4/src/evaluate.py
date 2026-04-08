from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


def classification_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary"
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1-score": float(f1),
    }


def compute_confusion_matrix(y_true: List[int], y_pred: List[int]) -> List[List[int]]:
    cm = confusion_matrix(y_true, y_pred)
    return cm.tolist()


def save_confusion_matrix_figure(cm: List[List[int]], save_path: str) -> None:
    """Save a simple confusion matrix heatmap figure."""
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks([0, 1], labels=["neg(0)", "pos(1)"])
    ax.set_yticks([0, 1], labels=["neg(0)", "pos(1)"])

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i][j]), ha="center", va="center")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

