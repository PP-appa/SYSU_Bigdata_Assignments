from __future__ import annotations

import pandas as pd


def recall_at_k(recommendations: pd.DataFrame, test: pd.DataFrame, k: int) -> float:
    hits = 0
    total = 0
    truth = dict(zip(test["userId"], test["movieId"]))
    for user_id, target_movie in truth.items():
        user_rec = recommendations[recommendations["userId"] == user_id].head(k)
        if user_rec.empty:
            continue
        total += 1
        hits += int(target_movie in set(user_rec["movieId"]))
    return hits / total if total else 0.0
