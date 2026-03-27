import ast
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = DATA_DIR / "processed"


def parse_json_like(value):
    if pd.isna(value) or value == "":
        return []
    return ast.literal_eval(value)


def build_movie_nodes(movies_df):
    movie_nodes = movies_df[
        [
            "id",
            "title",
            "original_title",
            "overview",
            "release_date",
            "vote_average",
            "vote_count",
            "popularity",
        ]
    ].copy()
    movie_nodes = movie_nodes.rename(columns={"id": "movie_id"})
    return movie_nodes.drop_duplicates(subset=["movie_id"])


def build_genre_nodes_and_relationships(movies_df):
    genre_nodes = []
    movie_genre_edges = []

    for _, row in movies_df[["id", "genres"]].iterrows():
        movie_id = row["id"]
        genres = parse_json_like(row["genres"])

        for genre in genres:
            genre_id = genre.get("id")
            genre_name = genre.get("name")
            if genre_id is None or not genre_name:
                continue

            genre_nodes.append(
                {
                    "genre_id": genre_id,
                    "name": genre_name,
                }
            )
            movie_genre_edges.append(
                {
                    "movie_id": movie_id,
                    "genre_id": genre_id,
                }
            )

    genre_nodes_df = pd.DataFrame(genre_nodes).drop_duplicates(subset=["genre_id"])
    movie_genre_edges_df = pd.DataFrame(movie_genre_edges).drop_duplicates()
    return genre_nodes_df, movie_genre_edges_df


def build_person_nodes_and_relationships(credits_df):
    person_nodes = []
    acted_in_edges = []
    directed_edges = []

    for _, row in credits_df[["movie_id", "cast", "crew"]].iterrows():
        movie_id = row["movie_id"]
        cast_list = parse_json_like(row["cast"])
        crew_list = parse_json_like(row["crew"])

        for cast_member in cast_list:
            person_id = cast_member.get("id")
            person_name = cast_member.get("name")
            if person_id is None or not person_name:
                continue

            person_nodes.append(
                {
                    "person_id": person_id,
                    "name": person_name,
                }
            )
            acted_in_edges.append(
                {
                    "person_id": person_id,
                    "movie_id": movie_id,
                    "character": cast_member.get("character"),
                    "cast_order": cast_member.get("order"),
                }
            )

        for crew_member in crew_list:
            if crew_member.get("job") != "Director":
                continue

            person_id = crew_member.get("id")
            person_name = crew_member.get("name")
            if person_id is None or not person_name:
                continue

            person_nodes.append(
                {
                    "person_id": person_id,
                    "name": person_name,
                }
            )
            directed_edges.append(
                {
                    "person_id": person_id,
                    "movie_id": movie_id,
                }
            )

    person_nodes_df = pd.DataFrame(person_nodes).drop_duplicates(subset=["person_id"])
    acted_in_edges_df = pd.DataFrame(acted_in_edges).drop_duplicates(
        subset=["person_id", "movie_id", "character"]
    )
    directed_edges_df = pd.DataFrame(directed_edges).drop_duplicates()
    return person_nodes_df, acted_in_edges_df, directed_edges_df


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    movies_path = DATA_DIR / "tmdb_5000_movies.csv"
    credits_path = DATA_DIR / "tmdb_5000_credits.csv"

    movies_df = pd.read_csv(movies_path)
    credits_df = pd.read_csv(credits_path)

    movie_nodes_df = build_movie_nodes(movies_df)
    genre_nodes_df, movie_genre_edges_df = build_genre_nodes_and_relationships(movies_df)
    person_nodes_df, acted_in_edges_df, directed_edges_df = build_person_nodes_and_relationships(
        credits_df
    )

    movie_nodes_df.to_csv(OUTPUT_DIR / "movie_nodes.csv", index=False, encoding="utf-8-sig")
    genre_nodes_df.to_csv(OUTPUT_DIR / "genre_nodes.csv", index=False, encoding="utf-8-sig")
    person_nodes_df.to_csv(OUTPUT_DIR / "person_nodes.csv", index=False, encoding="utf-8-sig")
    movie_genre_edges_df.to_csv(
        OUTPUT_DIR / "movie_genre_edges.csv", index=False, encoding="utf-8-sig"
    )
    acted_in_edges_df.to_csv(OUTPUT_DIR / "acted_in_edges.csv", index=False, encoding="utf-8-sig")
    directed_edges_df.to_csv(OUTPUT_DIR / "directed_edges.csv", index=False, encoding="utf-8-sig")

    print("Preprocessing completed.")
    print(f"Movie nodes: {len(movie_nodes_df)}")
    print(f"Genre nodes: {len(genre_nodes_df)}")
    print(f"Person nodes: {len(person_nodes_df)}")
    print(f"ACTED_IN edges: {len(acted_in_edges_df)}")
    print(f"DIRECTED edges: {len(directed_edges_df)}")
    print(f"BELONGS_TO edges: {len(movie_genre_edges_df)}")


if __name__ == "__main__":
    main()
