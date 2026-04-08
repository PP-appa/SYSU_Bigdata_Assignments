from typing import List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer


def build_tfidf_features(
    train_texts: List[str],
    test_texts: List[str],
    max_features: int = 20000,
    ngram_range: Tuple[int, int] = (1, 2),
) -> Tuple[object, object, TfidfVectorizer]:
    """Convert raw texts into sparse TF-IDF features."""
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
    x_train = vectorizer.fit_transform(train_texts)
    x_test = vectorizer.transform(test_texts)
    return x_train, x_test, vectorizer
