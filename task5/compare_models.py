import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare ARIMA and LSTM metrics.")
    parser.add_argument("--arima-metrics", type=str, default="outputs/metrics/arima_metrics.json")
    parser.add_argument("--lstm-metrics", type=str, default="outputs/metrics/lstm_metrics.json")
    parser.add_argument(
        "--output-csv",
        type=str,
        default="outputs/metrics/model_comparison.csv",
    )
    return parser.parse_args()


def load_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    arima = load_json(args.arima_metrics)
    lstm = load_json(args.lstm_metrics)

    comparison_df = pd.DataFrame(
        [
            {"Model": "ARIMA", "MAE": arima["mae"], "RMSE": arima["rmse"]},
            {"Model": "LSTM", "MAE": lstm["mae"], "RMSE": lstm["rmse"]},
        ]
    )
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(output_path, index=False)

    print(comparison_df.to_string(index=False))
    print(f"Comparison CSV saved to: {output_path}")


if __name__ == "__main__":
    main()
