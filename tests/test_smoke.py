"""End-to-end smoke test on a tiny synthetic corpus (no real data required)."""

from __future__ import annotations

import csv
import random
import sys
from pathlib import Path

import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import (  # noqa: E402
    build_dataset,
    clean_text,
    detect_labeled_columns,
)
from src.evaluate import evaluate_model, top_ngrams  # noqa: E402
from src.model import build_pipeline  # noqa: E402
from src.predict import predict, predict_proba  # noqa: E402

EN = ["the cat sat on the mat", "he went to school today", "the weather is nice",
      "my friend works in a big city", "life is short and time is fast",
      "water and food for the house", "i love this book at night"]
UR = ["یہ ایک کتاب ہے", "پانی گھر میں ہے", "وہ اسکول جاتا ہے", "آج موسم اچھا ہے",
      "میرا دوست شہر میں کام کرتا ہے", "زندگی اور محبت", "رات اور دن خواب"]
ZH = ["这是一本书", "我们在学校学习中文", "今天天气很好", "他的朋友在工作",
      "城市的生活很有意思", "我喜欢吃米饭", "老师和学生都很忙"]


def _make_corpus(tmp_path: Path, n: int = 30, seed: int = 7) -> Path:
    rng = random.Random(seed)
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    def synth(samples, joiner=" "):
        return [joiner.join(rng.choice(samples) for _ in range(rng.randint(1, 3)))
                for _ in range(n)]

    (data_dir / "english-corpus.txt").write_text(
        "\n".join(synth(EN)), encoding="utf-8")
    (data_dir / "urdu-corpus.txt").write_text(
        "\n".join(synth(UR)), encoding="utf-8")
    with (data_dir / "english-chinese.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["english", "chinese"])
        for row in zip(synth(EN), synth(ZH, joiner="")):
            writer.writerow(row)
    return data_dir


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    data_dir = _make_corpus(tmp_path_factory.mktemp("corpus"))
    df = build_dataset(data_dir, min_samples_per_lang=10)
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["lang"], test_size=0.3, random_state=42, stratify=df["lang"]
    )
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    return pipeline, X_train, X_test, y_train, y_test


def test_dataset_is_balanced_and_clean(trained):
    _, _, _, y_train, _ = trained
    assert set(y_train) == {"english", "urdu", "chinese"}
    assert len(y_train) >= 10


def test_model_learns(trained):
    pipeline, _, X_test, _, y_test = trained
    results = evaluate_model(pipeline, X_test, y_test)
    assert results["accuracy"] >= 0.9, results


def test_top_ngrams_are_returned(trained):
    pipeline, X_train, _, y_train, _ = trained
    ngrams = top_ngrams(pipeline, X_train, y_train, "chinese", top_n=5)
    assert len(ngrams) == 5
    assert all(isinstance(gram, str) and count > 0 for gram, count in ngrams)


def test_prediction_helpers(trained):
    pipeline, _, _, _, _ = trained
    assert predict("the cat sat on the mat", pipeline) == "english"
    scores = predict_proba("这是一本书", pipeline)
    assert max(scores, key=scores.get) == "chinese"
    assert pytest.approx(sum(scores.values()), abs=1e-6) == 1.0


def test_clean_text_normalises():
    assert clean_text("  Visit  http://x.com  now ", "english") == "visit now"
    assert clean_text("یہ   ایک", "urdu") == "یہ ایک"


# --------------------------------------------------------------------------
# Kaggle / WiLI-2018 labelled-CSV input path
# --------------------------------------------------------------------------
def test_detect_labeled_columns_is_header_agnostic(tmp_path):
    """The Kaggle export is seen with 'Text'/'language' and 'text'/'Language'."""
    for text_col, label_col in [("Text", "language"), ("text", "Language")]:
        path = tmp_path / f"{text_col}{label_col}.csv"
        pd.DataFrame(
            {
                text_col: ["the cat sat on the mat", "这是一本书", "یہ ایک کتاب ہے"],
                label_col: ["English", "Chinese", "Urdu"],
            }
        ).to_csv(path, index=False)
        df = pd.read_csv(path)
        detected_text, detected_label = detect_labeled_columns(
            df, ("chinese", "english", "urdu")
        )
        assert (detected_text, detected_label) == (text_col, label_col)


def test_labeled_csv_keeps_only_requested_languages(tmp_path):
    """Training from the Kaggle CSV must ignore the other 19 languages."""
    path = tmp_path / "dataset.csv"
    en = [f"the cat sat on the mat number {i}" for i in range(12)]
    zh = [f"这是一本书 number {i}" for i in range(12)]
    ur = [f"یہ ایک کتاب ہے number {i}" for i in range(12)]
    fr = [f"c'est un livre de francais {i}" for i in range(12)]
    pd.DataFrame(
        {
            "Text": en + zh + ur + fr,
            "language": ["English"] * 12 + ["Chinese"] * 12 + ["Urdu"] * 12 + ["French"] * 12,
        }
    ).to_csv(path, index=False)

    df = build_dataset(csv_path=path, min_samples_per_lang=10)
    assert set(df["lang"]) == {"english", "urdu", "chinese"}
    assert "french" not in set(df["lang"])
