from pathlib import Path

import pandas as pd
import spacy


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = DATA_DIR / "processed"

TARGET_LABELS = {"PERSON", "ORG", "GPE", "LOC", "DATE"}


def normalize_text(text):
    return " ".join(str(text).split()).strip()


def build_entity_tables(movie_nodes_df, nlp):
    entity_nodes = []
    movie_entity_edges = []
    entity_key_to_id = {}
    next_entity_id = 1

    rows = movie_nodes_df[["movie_id", "overview"]].fillna("").to_dict("records")
    texts = [row["overview"] for row in rows]

    for row, doc in zip(rows, nlp.pipe(texts, batch_size=64)):
        movie_id = row["movie_id"]
        seen_in_movie = set()

        for ent in doc.ents:
            if ent.label_ not in TARGET_LABELS:
                continue

            entity_name = normalize_text(ent.text)
            if len(entity_name) < 2:
                continue

            key = (entity_name.lower(), ent.label_)
            if key not in entity_key_to_id:
                entity_key_to_id[key] = next_entity_id
                entity_nodes.append(
                    {
                        "entity_id": next_entity_id,
                        "name": entity_name,
                        "label": ent.label_,
                    }
                )
                next_entity_id += 1

            entity_id = entity_key_to_id[key]
            if entity_id in seen_in_movie:
                continue

            movie_entity_edges.append(
                {
                    "movie_id": movie_id,
                    "entity_id": entity_id,
                    "relation": "MENTIONS",
                }
            )
            seen_in_movie.add(entity_id)

    entity_nodes_df = pd.DataFrame(entity_nodes).drop_duplicates(subset=["entity_id"])
    movie_entity_edges_df = pd.DataFrame(movie_entity_edges).drop_duplicates()
    return entity_nodes_df, movie_entity_edges_df


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    movie_nodes_path = OUTPUT_DIR / "movie_nodes.csv"
    movie_nodes_df = pd.read_csv(movie_nodes_path)

    nlp = spacy.load("en_core_web_sm")

    entity_nodes_df, movie_entity_edges_df = build_entity_tables(movie_nodes_df, nlp)

    entity_nodes_df.to_csv(OUTPUT_DIR / "overview_entity_nodes.csv", index=False, encoding="utf-8-sig")
    movie_entity_edges_df.to_csv(
        OUTPUT_DIR / "movie_overview_entity_edges.csv", index=False, encoding="utf-8-sig"
    )

    print("Overview entity extraction completed.")
    print(f"Overview entity nodes: {len(entity_nodes_df)}")
    print(f"Movie-entity edges: {len(movie_entity_edges_df)}")
    if not entity_nodes_df.empty:
        print("Entity label counts:")
        print(entity_nodes_df["label"].value_counts().to_string())


if __name__ == "__main__":
    main()
