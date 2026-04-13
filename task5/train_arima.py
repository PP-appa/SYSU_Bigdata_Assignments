import argparse
from pathlib import Path
from typing import Tuple

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from src.data import load_stock_csv
from src.evaluate import regression_metrics, save_forecast_figure
from src.utils import save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ARIMA model for stock forecasting.")
    parser.add_argument("--data-path", type=str, default="data/aapl_us_daily.csv")
    parser.add_argument("--forecast-horizon", type=int, default=7)
    parser.add_argument("--max-p", type=int, default=3)
    parser.add_argument("--max-d", type=int, default=2)
    parser.add_argument("--max-q", type=int, default=3)
    parser.add_argument("--metrics-path", type=str, default="outputs/metrics/arima_metrics.json")
    parser.add_argument(
        "--figure-path",
        type=str,
        default="outputs/figures/arima_forecast.png",
    )
    parser.add_argument(
        "--forecast-path",
        type=str,
        default="outputs/forecasts/arima_forecast.csv",
    )
    return parser.parse_args()


def select_best_order(
    train_values, max_p: int, max_d: int, max_q: int
) -> Tuple[Tuple[int, int, int], float]:
    best_order = None
    best_aic = float("inf")

    for p in range(max_p + 1):
        for d in range(max_d + 1):
            for q in range(max_q + 1):
                order = (p, d, q)
                try:
                    result = ARIMA(train_values, order=order).fit()
                except Exception:
                    continue
                if result.aic < best_aic:
                    best_aic = float(result.aic)
                    best_order = order

    if best_order is None:
        raise RuntimeError("Failed to fit any ARIMA configuration. Please check the input data.")
    return best_order, best_aic


def main() -> None:
    args = parse_args()
    dataset = load_stock_csv(args.data_path, forecast_horizon=args.forecast_horizon)
    best_order, best_aic = select_best_order(
        dataset.train_values,
        max_p=args.max_p,
        max_d=args.max_d,
        max_q=args.max_q,
    )

    model = ARIMA(dataset.train_values, order=best_order)
    result = model.fit()
    forecast = result.forecast(steps=args.forecast_horizon)
    metrics = regression_metrics(dataset.test_values, forecast)

    print(f"Best ARIMA order: {best_order}, AIC={best_aic:.4f}")
    print("Evaluation Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    forecast_df = pd.DataFrame(
        {
            "Date": dataset.test_dates.dt.strftime("%Y-%m-%d"),
            "Actual": dataset.test_values,
            "Predicted": forecast,
        }
    )
    Path(args.forecast_path).parent.mkdir(parents=True, exist_ok=True)
    forecast_df.to_csv(args.forecast_path, index=False)

    metrics_to_save = {
        **metrics,
        "best_order": list(best_order),
        "best_aic": best_aic,
        "train_size": int(len(dataset.train_values)),
        "test_size": int(len(dataset.test_values)),
    }
    save_json(metrics_to_save, args.metrics_path)
    save_forecast_figure(
        dates=dataset.test_dates,
        actual=dataset.test_values,
        predicted=forecast,
        title=f"ARIMA Forecast (order={best_order})",
        save_path=args.figure_path,
    )

    print(f"Metrics saved to: {args.metrics_path}")
    print(f"Forecast CSV saved to: {args.forecast_path}")
    print(f"Figure saved to: {args.figure_path}")


if __name__ == "__main__":
    main()
