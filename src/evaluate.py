"""Evaluation metrics and report artefacts.

Every figure is *returned* (and optionally saved to disk) - nothing calls
``plt.show()``, so the pipeline also runs on headless machines and in CI.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe backend

# Chinese / Urdu glyphs are missing from many default fonts; matplotlib warns
# about every missing glyph, which floods the console during report generation.
# Install a font such as Noto Sans CJK / Noto Nastaliq Urdu to render them.
# NB: filterwarnings(message=...) matches from the *start* of the text.
for _msg in (r".*missing from font", r".*does not support Arabic natively"):
    warnings.filterwarnings("ignore", message=_msg, category=UserWarning)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

PALETTE = ["#0078D7", "#E81123", "#2D7D9A"]


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def evaluate_model(pipeline, X_test, y_test) -> dict:
    y_pred = pipeline.predict(X_test)
    labels = list(pipeline.classes_)
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "report": classification_report(y_test, y_pred, labels=labels, output_dict=True),
        "labels": labels,
    }


def misclassified_examples(X_test, y_test, y_pred, limit: int = 10) -> pd.DataFrame:
    return (
        pd.DataFrame(
            {
                "text": list(X_test),
                "true": list(y_test),
                "pred": list(y_pred),
            }
        )
        .loc[list(np.asarray(y_test) != np.asarray(y_pred))]
        .head(limit)
        .reset_index(drop=True)
    )


def top_ngrams(pipeline, X_train, y_train, lang: str, top_n: int = 10) -> list[tuple[str, int]]:
    """Most frequent character n-grams for one language.

    Uses the *already fitted* vectoriser inside the pipeline and boolean masks
    on the sparse matrix (no ``.toarray()``, no re-fitting).
    """
    vectorizer = pipeline.named_steps["countvectorizer"]
    X_vec = vectorizer.transform(X_train)
    vocab = np.asarray(vectorizer.get_feature_names_out())

    mask = (np.asarray(y_train) == lang)
    counts = np.asarray(X_vec[mask].sum(axis=0)).ravel()

    top_idx = counts.argsort()[-top_n:][::-1]
    return [(str(vocab[i]), int(counts[i])) for i in top_idx if counts[i] > 0]


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------
def save_fig(fig: plt.Figure, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_confusion_matrix(y_test, y_pred, labels) -> plt.Figure:
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels).plot(
        ax=ax, cmap=plt.cm.Blues, xticks_rotation=45, colorbar=False
    )
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    return fig


def plot_per_class_metrics(y_test, y_pred, labels) -> plt.Figure:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, labels=labels
    )
    x = np.arange(len(labels))
    width = 0.26
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - width, precision, width, label="Precision", color=PALETTE[0])
    ax.bar(x, recall, width, label="Recall", color=PALETTE[1])
    ax.bar(x + width, f1, width, label="F1-score", color=PALETTE[2])
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Precision / Recall / F1 per Language")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_class_distribution(y_train) -> plt.Figure:
    counts = pd.Series(y_train).value_counts()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(counts.index, counts.values, color=PALETTE)
    ax.set_title("Training Data Distribution")
    ax.set_ylabel("Number of Samples")
    fig.tight_layout()
    return fig


def plot_probability_distribution(y_prob, labels) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, lang in enumerate(labels):
        ax.hist(y_prob[:, i], bins=20, alpha=0.5, label=lang)
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Frequency")
    ax.set_title("Prediction Probability Distribution")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_top_ngrams(ngrams: list[tuple[str, int]], lang: str) -> plt.Figure:
    grams = [g for g, _ in ngrams]
    values = [v for _, v in ngrams]
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.bar(grams, values, color=PALETTE[2])
    ax.set_title(f"Top {len(grams)} Character N-grams - {lang}")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return fig
