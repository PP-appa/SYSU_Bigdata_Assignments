from __future__ import annotations

import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd


MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"


def ensure_movielens(data_dir: Path) -> Path:
    """Return MovieLens small directory, downloading it when absent."""
    data_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = data_dir / "ml-latest-small"
    if dataset_dir.exists():
        return dataset_dir

    zip_path = data_dir / "ml-latest-small.zip"
    if not zip_path.exists():
        urlretrieve(MOVIELENS_URL, zip_path)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(data_dir)
    return dataset_dir


def load_movielens(data_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset_dir = ensure_movielens(Path(data_dir))
    ratings = pd.read_csv(dataset_dir / "ratings.csv")
    movies = pd.read_csv(dataset_dir / "movies.csv")
    return ratings, movies


def leave_one_out_split(
    ratings: pd.DataFrame,
    min_rating: float = 4.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Use each user's latest positive interaction as test data."""
    positives = ratings[ratings["rating"] >= min_rating].copy()
    positives = positives.sort_values(["userId", "timestamp"])
    test_idx = positives.groupby("userId").tail(1).index
    test = positives.loc[test_idx, ["userId", "movieId"]].copy()
    train = ratings.drop(index=test_idx).copy()
    return train, test
