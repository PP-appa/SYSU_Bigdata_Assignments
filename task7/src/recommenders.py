from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer


def _seen_by_user(train: pd.DataFrame) -> dict[int, set[int]]:
    return train.groupby("userId")["movieId"].apply(set).to_dict()


def recommend_item_cf(train: pd.DataFrame, top_k: int = 10) -> pd.DataFrame:
    user_item = train.pivot_table(
        index="userId",
        columns="movieId",
        values="rating",
        fill_value=0.0,
    )
    item_user = user_item.T
    sim = cosine_similarity(item_user)
    movie_ids = item_user.index.to_numpy()
    seen = _seen_by_user(train)

    rows: list[dict[str, float | int]] = []
    for user_id, ratings in user_item.iterrows():
        rated = ratings.to_numpy()
        scores = sim @ rated
        norm = np.abs(sim) @ (rated > 0)
        scores = np.divide(scores, norm, out=np.zeros_like(scores), where=norm > 0)

        user_seen = seen.get(user_id, set())
        candidates = [
            (movie_id, score)
            for movie_id, score in zip(movie_ids, scores)
            if movie_id not in user_seen
        ]
        candidates.sort(key=lambda x: x[1], reverse=True)
        for rank, (movie_id, score) in enumerate(candidates[:top_k], start=1):
            rows.append(
                {"userId": user_id, "movieId": int(movie_id), "score": float(score), "rank": rank}
            )
    return pd.DataFrame(rows)


def recommend_content_based(
    train: pd.DataFrame,
    movies: pd.DataFrame,
    top_k: int = 10,
    min_rating: float = 4.0,
) -> pd.DataFrame:
    movies = movies.copy()
    movies["genre_list"] = movies["genres"].fillna("").str.split("|")
    genre_matrix = MultiLabelBinarizer().fit_transform(movies["genre_list"])
    movie_ids = movies["movieId"].to_numpy()
    movie_to_idx = {movie_id: idx for idx, movie_id in enumerate(movie_ids)}
    seen = _seen_by_user(train)

    rows: list[dict[str, float | int]] = []
    for user_id, user_ratings in train.groupby("userId"):
        liked_ids = user_ratings[user_ratings["rating"] >= min_rating]["movieId"]
        liked_idx = [movie_to_idx[movie_id] for movie_id in liked_ids if movie_id in movie_to_idx]
        if not liked_idx:
            continue

        profile = genre_matrix[liked_idx].mean(axis=0).reshape(1, -1)
        scores = cosine_similarity(profile, genre_matrix).ravel()
        user_seen = seen.get(user_id, set())
        candidates = [
            (movie_id, score)
            for movie_id, score in zip(movie_ids, scores)
            if movie_id not in user_seen
        ]
        candidates.sort(key=lambda x: x[1], reverse=True)
        for rank, (movie_id, score) in enumerate(candidates[:top_k], start=1):
            rows.append(
                {"userId": user_id, "movieId": int(movie_id), "score": float(score), "rank": rank}
            )
    return pd.DataFrame(rows)


def recommend_hybrid(
    cf_recs: pd.DataFrame,
    cb_recs: pd.DataFrame,
    alpha: float = 0.7,
    top_k: int = 10,
) -> pd.DataFrame:
    cf = cf_recs.rename(columns={"score": "cf_score"})[["userId", "movieId", "cf_score"]]
    cb = cb_recs.rename(columns={"score": "cb_score"})[["userId", "movieId", "cb_score"]]
    merged = pd.merge(cf, cb, on=["userId", "movieId"], how="outer").fillna(0.0)
    merged["cf_score"] = _minmax_by_user(merged, "cf_score")
    merged["cb_score"] = _minmax_by_user(merged, "cb_score")
    merged["score"] = alpha * merged["cf_score"] + (1 - alpha) * merged["cb_score"]
    merged = merged.sort_values(["userId", "score"], ascending=[True, False])
    merged["rank"] = merged.groupby("userId").cumcount() + 1
    return merged[merged["rank"] <= top_k][["userId", "movieId", "score", "rank"]]


def _minmax_by_user(df: pd.DataFrame, column: str) -> pd.Series:
    grouped = df.groupby("userId")[column]
    min_score = grouped.transform("min")
    max_score = grouped.transform("max")
    denom = (max_score - min_score).replace(0, 1.0)
    return (df[column] - min_score) / denom
