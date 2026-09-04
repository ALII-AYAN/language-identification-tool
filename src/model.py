"""Model definition: character n-grams + Multinomial Naive Bayes."""

from __future__ import annotations

from pathlib import Path

import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline, make_pipeline

from .config import ALPHA, NGRAM_RANGE


def build_pipeline(
    ngram_range: tuple[int, int] = NGRAM_RANGE, alpha: float = ALPHA
) -> Pipeline:
    """Vectoriser + classifier as a single, picklable object."""
    vectorizer = CountVectorizer(analyzer="char", ngram_range=ngram_range)
    classifier = MultinomialNB(alpha=alpha)
    return make_pipeline(vectorizer, classifier)


def save_model(pipeline: Pipeline, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    return path


def load_model(path: Path) -> Pipeline:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Trained model not found at {path}.\n"
            "Train it first:  python -m src.train"
        )
    return joblib.load(path)
