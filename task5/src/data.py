from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


@dataclass
class StockDataset:
    raw_df: pd.DataFrame
    train_dates: pd.Series
    test_dates: pd.Series
    train_values: np.ndarray
    test_values: np.ndarray


def load_stock_csv(csv_path: str, forecast_horizon: int = 7) -> StockDataset:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(path)
    date_col = _find_date_column(df)
    price_col = _find_price_column(df)

    df = df[[date_col, price_col]].copy()
    df.columns = ["Date", "Close"]
    df["Date"] = pd.to_datetime(df["Date"])
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna().sort_values("Date").reset_index(drop=True)

    if len(df) <= forecast_horizon + 30:
        raise ValueError(
            f"Not enough rows in {csv_path}. Need more than {forecast_horizon + 30} valid records."
        )

    train_df = df.iloc[:-forecast_horizon].copy()
    test_df = df.iloc[-forecast_horizon:].copy()
    return StockDataset(
        raw_df=df,
        train_dates=train_df["Date"],
        test_dates=test_df["Date"],
        train_values=train_df["Close"].to_numpy(dtype=float),
        test_values=test_df["Close"].to_numpy(dtype=float),
    )


def minmax_scale(
    train_values: np.ndarray, test_values: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    train_min = float(train_values.min())
    train_max = float(train_values.max())
    denom = max(train_max - train_min, 1e-8)
    train_scaled = (train_values - train_min) / denom
    test_scaled = (test_values - train_min) / denom
    return train_scaled, test_scaled, train_min, train_max


def inverse_minmax_scale(values: np.ndarray, train_min: float, train_max: float) -> np.ndarray:
    return values * max(train_max - train_min, 1e-8) + train_min


def _find_date_column(df: pd.DataFrame) -> str:
    candidates = ["Date", "date", "Datetime", "datetime", "日期"]
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"Date column not found. Available columns: {list(df.columns)}")


def _find_price_column(df: pd.DataFrame) -> str:
    candidates = ["Adj Close", "adj_close", "Close", "close", "收盘", "收盘价"]
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"Close price column not found. Available columns: {list(df.columns)}")
