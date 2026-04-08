from typing import List, Tuple

from datasets import load_dataset


def load_imdb_data(lowercase: bool = True) -> Tuple[List[str], List[int], List[str], List[int]]:
    """
    Load IMDB train/test splits from Hugging Face datasets.
    This module only handles data loading and optional lowercase preprocessing.
    """
    dataset = load_dataset("imdb")
    train_texts = list(dataset["train"]["text"])
    train_labels = list(dataset["train"]["label"])
    test_texts = list(dataset["test"]["text"])
    test_labels = list(dataset["test"]["label"])

    if lowercase:
        train_texts = [text.lower().strip() for text in train_texts]
        test_texts = [text.lower().strip() for text in test_texts]

    return train_texts, train_labels, test_texts, test_labels

