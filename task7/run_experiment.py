from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data import leave_one_out_split, load_movielens
from src.metrics import recall_at_k
from src.recommenders import recommend_content_based, recommend_hybrid, recommend_item_cf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="task7/data")
    parser.add_argument("--output-dir", default="task7/outputs/metrics")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--candidate-k", type=int, default=100)
    parser.add_argument("--min-rating", type=float, default=4.0)
    parser.add_argument("--hybrid-alpha", type=float, default=0.7)
    args = parser.parse_args()

    ratings, movies = load_movielens(args.data_dir)
    train, test = leave_one_out_split(ratings, min_rating=args.min_rating)

    candidate_k = max(args.candidate_k, args.top_k)
    cf_candidates = recommend_item_cf(train, top_k=candidate_k)
    cb_candidates = recommend_content_based(
        train,
        movies,
        top_k=candidate_k,
        min_rating=args.min_rating,
    )
    cf_recs = cf_candidates[cf_candidates["rank"] <= args.top_k]
    cb_recs = cb_candidates[cb_candidates["rank"] <= args.top_k]
    hybrid_recs = recommend_hybrid(
        cf_candidates,
        cb_candidates,
        alpha=args.hybrid_alpha,
        top_k=args.top_k,
    )

    metrics = {
        "dataset": "MovieLens ml-latest-small",
        "num_ratings": int(len(ratings)),
        "num_users": int(ratings["userId"].nunique()),
        "num_movies": int(movies["movieId"].nunique()),
        "split": "leave-one-out latest positive rating per user",
        "top_k": args.top_k,
        "candidate_k": candidate_k,
        "min_positive_rating": args.min_rating,
        "hybrid_alpha": args.hybrid_alpha,
        "recall": {
            "item_cf": recall_at_k(cf_recs, test, args.top_k),
            "content_based": recall_at_k(cb_recs, test, args.top_k),
            "hybrid": recall_at_k(hybrid_recs, test, args.top_k),
        },
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
