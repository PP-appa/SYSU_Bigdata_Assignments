from pathlib import Path
from typing import Dict, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def regression_metrics(y_true: Iterable[float], y_pred: Iterable[float]) -> Dict[str, float]:
    y_true_arr = np.asarray(list(y_true), dtype=float)
    y_pred_arr = np.asarray(list(y_pred), dtype=float)
    mae = mean_absolute_error(y_true_arr, y_pred_arr)
    rmse = np.sqrt(mean_squared_error(y_true_arr, y_pred_arr))
    return {"mae": float(mae), "rmse": float(rmse)}


def save_forecast_figure(
    dates: pd.Series,
    actual: Iterable[float],
    predicted: Iterable[float],
    title: str,
    save_path: str,
) -> None:
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(dates, list(actual), marker="o", label="Actual")
    ax.plot(dates, list(predicted), marker="o", linestyle="--", label="Predicted")
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Close Price")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
