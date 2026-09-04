"""Download / prepare the Kaggle "Language Identification" dataset.

The dataset is a 22-language subset of **WiLI-2018** (Wikipedia Language
Identification benchmark):

    https://www.kaggle.com/datasets/zarajamshaid/language-identification-datasst

Two ways to use it:

1. Train straight from the CSV (nothing to prepare):

       python -m src.train --csv data/dataset.csv

2. Or split it into the three files this project also understands:

       python scripts/prepare_dataset.py --csv data/dataset.csv --out data

Usage
-----
    python scripts/prepare_dataset.py --csv data/dataset.csv --out data
    python scripts/prepare_dataset.py --csv data/dataset.csv --languages english urdu chinese
    python scripts/prepare_dataset.py --csv data/dataset.csv --inspect   # stats only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.data import detect_labeled_columns, load_labeled_csv  # noqa: E402

KAGGLE_DATASET = "zarajamshaid/language-identification-datasst"


def inspect(path: Path, languages) -> pd.DataFrame:
    df = pd.read_csv(path, engine="python")
    text_col, label_col = detect_labeled_columns(df, languages)
    print(f"File      : {path}")
    print(f"Rows      : {len(df)}")
    print(f"Text col  : {text_col!r}")
    print(f"Label col : {label_col!r}")
    print("\nSamples per language:")
    print(df[label_col].value_counts().to_string())
    return df


def export(path: Path, out_dir: Path, languages) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading...")
    df = load_labeled_csv(path, languages)

    for lang, subset in df.groupby("lang"):
        text = "\n".join(subset["text"].tolist())
        if lang == "chinese":
            # keep a CSV so the CJK auto-detection path stays exercised
            target = out_dir / "english-chinese.csv"
            pd.DataFrame({"language": lang, "text": subset["text"].tolist()}).to_csv(
                target, index=False, encoding="utf-8"
            )
        else:
            target = out_dir / f"{lang}-corpus.txt"
            target.write_text(text, encoding="utf-8")
        print(f"  {target.name}: {len(subset)} samples")

    print(f"\nDone. Train with:  python -m src.train --data-dir {out_dir}")


def download(out_dir: Path) -> None:
    """Pull the dataset with the official Kaggle CLI (needs kaggle.json)."""
    import subprocess

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {KAGGLE_DATASET} -> {out_dir}")
    result = subprocess.run(
        ["kaggle", "datasets", "download", "-d", KAGGLE_DATASET, "-p", str(out_dir)],
        check=False,
    )
    if result.returncode != 0:
        raise SystemExit(
            "Kaggle CLI failed. Set up credentials first:\n"
            "  pip install kaggle\n"
            "  # create an API token at https://www.kaggle.com/settings -> 'Create New Token'\n"
            "  # then place kaggle.json in ~/.kaggle/ (chmod 600)\n"
            "Or download dataset.csv manually from the dataset page and put it in data/."
        )

    import zipfile

    for archive in out_dir.glob("*.zip"):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(out_dir)
        archive.unlink()
        print(f"Extracted {archive.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path("data/dataset.csv"))
    parser.add_argument("--out", type=Path, default=Path("data"))
    parser.add_argument(
        "--languages", nargs="+", default=["english", "urdu", "chinese"],
        help="which languages to keep (default: english urdu chinese)",
    )
    parser.add_argument("--inspect", action="store_true", help="print stats and exit")
    parser.add_argument("--download", action="store_true", help="fetch from Kaggle first")
    args = parser.parse_args()

    if args.download:
        download(args.out)

    if args.inspect:
        inspect(args.csv, args.languages)
        return

    export(args.csv, args.out, args.languages)


if __name__ == "__main__":
    main()
