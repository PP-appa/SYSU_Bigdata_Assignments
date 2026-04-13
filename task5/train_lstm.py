import argparse
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.data import inverse_minmax_scale, load_stock_csv, minmax_scale
from src.evaluate import regression_metrics, save_forecast_figure
from src.utils import save_json, set_seed


class SequenceDataset(Dataset):
    def __init__(self, series: np.ndarray, window_size: int):
        self.features, self.targets = build_sequences(series, window_size)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.tensor(self.features[idx], dtype=torch.float32).unsqueeze(-1)
        y = torch.tensor(self.targets[idx], dtype=torch.float32)
        return x, y


class StockLSTM(nn.Module):
    def __init__(self, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        effective_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=effective_dropout,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs, _ = self.lstm(x)
        last_hidden = outputs[:, -1, :]
        return self.fc(last_hidden).squeeze(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LSTM model for stock forecasting.")
    parser.add_argument("--data-path", type=str, default="data/aapl_us_daily.csv")
    parser.add_argument("--forecast-horizon", type=int, default=7)
    parser.add_argument("--window-size", type=int, default=20)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--metrics-path", type=str, default="outputs/metrics/lstm_metrics.json")
    parser.add_argument(
        "--figure-path",
        type=str,
        default="outputs/figures/lstm_forecast.png",
    )
    parser.add_argument(
        "--curve-fig-path",
        type=str,
        default="outputs/figures/lstm_training_curve.png",
    )
    parser.add_argument(
        "--forecast-path",
        type=str,
        default="outputs/forecasts/lstm_forecast.csv",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="outputs/model/lstm_state_dict.pt",
    )
    return parser.parse_args()


def build_sequences(series: np.ndarray, window_size: int) -> Tuple[np.ndarray, np.ndarray]:
    xs: List[np.ndarray] = []
    ys: List[float] = []
    for i in range(len(series) - window_size):
        xs.append(series[i : i + window_size])
        ys.append(series[i + window_size])
    if not xs:
        raise ValueError("Not enough training data for the selected window size.")
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


def recursive_forecast(
    model: nn.Module,
    history: np.ndarray,
    window_size: int,
    horizon: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    rolling = history.copy().astype(np.float32)
    preds: List[float] = []
    with torch.no_grad():
        for _ in range(horizon):
            window = rolling[-window_size:]
            x = torch.tensor(window, dtype=torch.float32, device=device).view(1, window_size, 1)
            next_pred = model(x).item()
            preds.append(next_pred)
            rolling = np.append(rolling, next_pred)
    return np.asarray(preds, dtype=np.float32)


def save_training_curve(history: List[float], save_path: str) -> None:
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(history) + 1), history, marker="o")
    ax.set_title("LSTM Training Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    set_seed(42)

    dataset = load_stock_csv(args.data_path, forecast_horizon=args.forecast_horizon)
    train_scaled, test_scaled, train_min, train_max = minmax_scale(
        dataset.train_values, dataset.test_values
    )
    train_dataset = SequenceDataset(train_scaled, window_size=args.window_size)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StockLSTM(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    loss_history: List[float] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for features, targets in train_loader:
            features = features.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            preds = model(features)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        loss_history.append(float(avg_loss))
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(f"Epoch {epoch:03d}/{args.epochs} - train_loss: {avg_loss:.6f}")

    scaled_forecast = recursive_forecast(
        model=model,
        history=train_scaled,
        window_size=args.window_size,
        horizon=args.forecast_horizon,
        device=device,
    )
    forecast = inverse_minmax_scale(scaled_forecast, train_min=train_min, train_max=train_max)
    metrics = regression_metrics(dataset.test_values, forecast)

    forecast_df = pd.DataFrame(
        {
            "Date": dataset.test_dates.dt.strftime("%Y-%m-%d"),
            "Actual": dataset.test_values,
            "Predicted": forecast,
        }
    )
    Path(args.forecast_path).parent.mkdir(parents=True, exist_ok=True)
    forecast_df.to_csv(args.forecast_path, index=False)
    Path(args.model_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.model_path)

    metrics_to_save = {
        **metrics,
        "window_size": args.window_size,
        "epochs": args.epochs,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "train_loss_history": loss_history,
        "train_size": int(len(dataset.train_values)),
        "test_size": int(len(dataset.test_values)),
    }
    save_json(metrics_to_save, args.metrics_path)
    save_forecast_figure(
        dates=dataset.test_dates,
        actual=dataset.test_values,
        predicted=forecast,
        title="LSTM Forecast",
        save_path=args.figure_path,
    )
    save_training_curve(loss_history, args.curve_fig_path)

    print("Evaluation Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    print(f"Metrics saved to: {args.metrics_path}")
    print(f"Forecast CSV saved to: {args.forecast_path}")
    print(f"Model saved to: {args.model_path}")
    print(f"Forecast figure saved to: {args.figure_path}")
    print(f"Training curve saved to: {args.curve_fig_path}")


if __name__ == "__main__":
    main()
