"""Command-line inference with the trained language identifier.

Usage
-----
    python -m src.predict "This is a sentence"
    python -m src.predict --file notes.txt        # one sample per line
    python -m src.predict --stdin                 # pipe text in
    python -m src.predict --interactive
"""

from __future__ import annotations

import argparse
import sys
from functools import lru_cache
from pathlib import Path

import pandas as pd

from .config import MODEL_PATH
from .model import load_model


@lru_cache(maxsize=1)
def get_model(path: str = str(MODEL_PATH)):
    return load_model(Path(path))


def predict(text: str, model=None) -> str:
    model = model or get_model()
    return str(model.predict([text])[0])


def predict_proba(text: str, model=None) -> dict[str, float]:
    model = model or get_model()
    probs = model.predict_proba([text])[0]
    return {str(lang): float(p) for lang, p in zip(model.classes_, probs)}


def predict_many(texts, model=None):
    model = model or get_model()
    labels = model.predict(list(texts))
    return list(labels)


def predict_file(path: Path, model=None) -> pd.DataFrame:
    path = Path(path)
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    lines = [line for line in lines if len(line) > 1]
    return pd.DataFrame({"text": lines, "lang": predict_many(lines, model)})


def _print_result(text: str, model) -> None:
    scores = predict_proba(text, model)
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    print(f"Text      : {text[:80]}{'...' if len(text) > 80 else ''}")
    print(f"Language  : {ranked[0][0]}")
    for lang, prob in ranked:
        print(f"  {lang:<8} {prob:.4f}")
    print("-" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect the language of a text")
    parser.add_argument("text", nargs="*", help="text to classify")
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--file", type=Path, help="file with one sample per line")
    parser.add_argument("--csv", type=Path, help="CSV file + --column to classify in bulk")
    parser.add_argument("--column", type=str, default=None)
    parser.add_argument("--out", type=Path, help="where to save CSV predictions")
    parser.add_argument("--stdin", action="store_true", help="read text from stdin")
    parser.add_argument("--interactive", action="store_true", help="REPL loop")
    args = parser.parse_args()

    try:
        model = get_model(str(args.model))
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)

    if args.file:
        df = predict_file(args.file, model)
        out = args.out or args.file.with_name(f"{args.file.stem}_predicted.csv")
        df.to_csv(out, index=False)
        print(df.to_string(max_colwidth=60))
        print(f"\nSaved -> {out}")
        return

    if args.csv:
        df = pd.read_csv(args.csv)
        column = args.column or df.columns[0]
        df["predicted_lang"] = predict_many(df[column].astype(str).tolist(), model)
        out = args.out or args.csv.with_name(f"{args.csv.stem}_predicted.csv")
        df.to_csv(out, index=False)
        print(df.head(20).to_string(max_colwidth=40))
        print(f"\nSaved -> {out}")
        return

    if args.stdin:
        text = sys.stdin.read().strip()
        _print_result(text, model) if text else None
        return

    if args.interactive:
        print("Type text to detect (blank line or Ctrl+C to quit).")
        try:
            while True:
                text = input("> ").strip()
                if not text:
                    break
                _print_result(text, model)
        except (KeyboardInterrupt, EOFError):
            print("\nBye.")
        return

    if args.text:
        _print_result(" ".join(args.text), model)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
