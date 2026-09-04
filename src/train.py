"""Train the language identifier and write a full evaluation report.

Usage
-----
    python -m src.train                       # defaults
    python -m src.train --cv 5                # + 5-fold cross-validation
    python -m src.train --data-dir /path/data --model-out models/model.joblib
    python -m src.train --no-plots --top-n 15
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from .config import (
    ALPHA,
    DATASET_CSV,
    DATA_DIR,
    METRICS_PATH,
    MODEL_PATH,
    NGRAM_RANGE,
    OUTPUT_DIR,
    RANDOM_STATE,
    TEST_SIZE,
)
from .data import build_dataset
from .evaluate import (
    evaluate_model,
    misclassified_examples,
    plot_class_distribution,
    plot_confusion_matrix,
    plot_per_class_metrics,
    plot_probability_distribution,
    plot_top_ngrams,
    save_fig,
    top_ngrams,
)
from .model import build_pipeline, save_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the language identifier")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="train from one labelled CSV (Kaggle / WiLI-2018 style, columns: text + language)",
    )
    parser.add_argument("--model-out", type=Path, default=MODEL_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--test-size", type=float, default=TEST_SIZE)
    parser.add_argument("--ngram-min", type=int, default=NGRAM_RANGE[0])
    parser.add_argument("--ngram-max", type=int, default=NGRAM_RANGE[1])
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument(
        "--cv", type=int, default=0, help="k-fold cross-validation on the full set (0 = off)"
    )
    parser.add_argument("--top-n", type=int, default=10, help="n-grams to report per language")
    parser.add_argument("--no-plots", action="store_true", help="skip PNG artefacts")
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Data -------------------------------------------------------------
    df = build_dataset(
        args.data_dir, save_csv=output_dir / DATASET_CSV.name, csv_path=args.csv
    )
    X, y = df["text"], df["lang"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )
    print(f"Train: {len(X_train)}   Test: {len(X_test)}")

    # 2. Model ------------------------------------------------------------
    pipeline = build_pipeline(
        ngram_range=(args.ngram_min, args.ngram_max), alpha=args.alpha
    )
    print("Training model...")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)

    # 3. Metrics ----------------------------------------------------------
    results = evaluate_model(pipeline, X_test, y_test)
    labels = results["labels"]
    print("\n=========== RESULTS ===========")
    print(f"Hold-out accuracy: {results['accuracy']:.4f}")
    print(classification_report(y_test, y_pred, labels=labels))

    if args.cv and args.cv > 1:
        cv = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=args.random_state)
        scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
        results["cv_mean"] = float(scores.mean())
        results["cv_std"] = float(scores.std())
        print(f"{args.cv}-fold CV accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")

    # 4. Artefacts --------------------------------------------------------
    if not args.no_plots:
        save_fig(plot_confusion_matrix(y_test, y_pred, labels), output_dir / "confusion_matrix.png")
        save_fig(plot_per_class_metrics(y_test, y_pred, labels), output_dir / "metrics_per_language.png")
        save_fig(plot_class_distribution(y_train), output_dir / "class_distribution.png")
        save_fig(plot_probability_distribution(y_prob, labels), output_dir / "probability_distribution.png")

    ngram_report = {}
    for lang in labels:
        ngrams = top_ngrams(pipeline, X_train, y_train, lang, top_n=args.top_n)
        ngram_report[lang] = ngrams
        print(f"\nTop {args.top_n} n-grams for {lang}:")
        for gram, count in ngrams:
            print(f"  {gram!r}: {count}")
        if not args.no_plots and ngrams:
            save_fig(plot_top_ngrams(ngrams, lang), output_dir / f"top_ngrams_{lang}.png")

    errors = misclassified_examples(X_test, y_test, y_pred)
    if len(errors):
        print(f"\nMisclassified samples: {len(errors)} (top {min(len(errors), 10)} shown)")
        print(errors.head(10).to_string(max_colwidth=60))
    errors.to_csv(output_dir / "misclassified.csv", index=False)

    # 5. Persist ----------------------------------------------------------
    results["ngram_range"] = [args.ngram_min, args.ngram_max]
    results["alpha"] = args.alpha
    results["n_samples"] = int(len(df))
    results["top_ngrams"] = {k: v[: args.top_n] for k, v in ngram_report.items()}
    (output_dir / METRICS_PATH.name).write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    model_path = save_model(pipeline, args.model_out)
    print(f"\nModel saved to: {model_path}")
    print(f"Reports saved to: {output_dir}")
    print("DONE")


if __name__ == "__main__":
    main()
