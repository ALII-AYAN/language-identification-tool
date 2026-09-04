"""Dataset loading, cleaning and assembly.

Expected files inside ``DATA_DIR`` (override with ``LID_DATA_DIR``):

* ``english-corpus.txt``  - one English sentence / document per line
* ``urdu-corpus.txt``     - one Urdu sentence / document per line
* ``english-chinese.csv`` - any CSV containing at least one Chinese column
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from .config import (
    CHINESE_FILE,
    DATA_DIR,
    ENGLISH_FILE,
    LABELED_CSV,
    URDU_FILE,
)

# CJK unified ideographs + extension A + compatibility ideographs
CJK_PATTERN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
URL_PATTERN = re.compile(r"http\S+|www\.\S+|\S+@\S+")
WHITESPACE_PATTERN = re.compile(r"\s+")

# Scripts where case-folding actually carries information
LOWERCASE_LANGS = {"english"}


# --------------------------------------------------------------------------
# Cleaning
# --------------------------------------------------------------------------
def clean_text(text: str, lang: str) -> str:
    """Normalise a single sample: strip URLs/emails, collapse whitespace."""
    text = str(text).strip()
    text = URL_PATTERN.sub(" ", text)
    text = WHITESPACE_PATTERN.sub(" ", text)
    if lang in LOWERCASE_LANGS:
        text = text.lower()
    return text.strip()


def _read_text_file(path: Path) -> str:
    """Read a corpus file, tolerating BOM / non-UTF-8 exports."""
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="ignore")


def _lines_to_frame(lines, lang: str) -> pd.DataFrame:
    rows = []
    seen = set()
    dropped = 0
    for raw in lines:
        text = clean_text(raw, lang)
        if len(text) > 1 and text not in seen:
            seen.add(text)
            rows.append({"text": text, "lang": lang})
        else:
            dropped += 1
    print(f"  {lang}: kept {len(rows)} unique samples (dropped {dropped} blank/duplicate)")
    return pd.DataFrame(rows, columns=["text", "lang"])


# --------------------------------------------------------------------------
# Loaders
# --------------------------------------------------------------------------
def load_english(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    path = Path(data_dir) / ENGLISH_FILE
    if not path.exists():
        raise FileNotFoundError(f"Missing English corpus: {path}")
    return _lines_to_frame(_read_text_file(path).splitlines(), "english")


def load_urdu(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    path = Path(data_dir) / URDU_FILE
    if not path.exists():
        raise FileNotFoundError(f"Missing Urdu corpus: {path}")
    return _lines_to_frame(_read_text_file(path).splitlines(), "urdu")


def _cjk_ratio(text: str) -> float:
    chars = [ch for ch in text if not ch.isspace()]
    if not chars:
        return 0.0
    hits = sum(1 for ch in chars if CJK_PATTERN.match(ch))
    return hits / len(chars)


def detect_chinese_column(df: pd.DataFrame, threshold: float = 0.3) -> str:
    """Pick the column that is *most* Chinese instead of the first lucky hit."""
    best_col, best_ratio = None, 0.0
    for col in df.columns:
        ratio = _cjk_ratio("".join(df[col].astype(str).head(200)))
        if ratio > best_ratio:
            best_col, best_ratio = col, ratio
    if best_col is None or best_ratio < threshold:
        raise ValueError(
            f"No Chinese column found in {CHINESE_FILE} "
            f"(best CJK ratio was {best_ratio:.2f})"
        )
    return best_col


def load_chinese_from_csv(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    path = Path(data_dir) / CHINESE_FILE
    if not path.exists():
        raise FileNotFoundError(f"Missing Chinese CSV: {path}")
    df = pd.read_csv(path, engine="python")
    chinese_col = detect_chinese_column(df)
    print(f"  Chinese column detected: {chinese_col!r}")
    return _lines_to_frame(df[chinese_col].dropna().astype(str).tolist(), "chinese")


# --------------------------------------------------------------------------
# Alternative input: one labelled CSV (Kaggle / WiLI-2018 style)
# --------------------------------------------------------------------------
def detect_labeled_columns(df: pd.DataFrame, languages) -> tuple[str, str]:
    """Guess (text_column, label_column) in a labelled CSV.

    The Kaggle export is seen in the wild with headers ``Text``/``language``,
    ``text``/``Language`` and a few other casings, so both columns are detected
    instead of hard-coded:

    * the label column is the one whose unique values overlap the languages we
      asked for;
    * the text column is the remaining one with the longest average content.
    """
    wanted = {str(lang).strip().lower() for lang in languages}

    label_col = None
    for col in df.columns:
        values = {str(v).strip().lower() for v in df[col].dropna().unique()}
        if values & wanted:
            label_col = col
            break
    if label_col is None:
        raise ValueError(
            f"No label column found (looked for {sorted(wanted)} in "
            f"{list(df.columns)})"
        )

    candidates = [c for c in df.columns if c != label_col]
    if not candidates:
        raise ValueError("CSV has only one column - need text + label")
    text_col = max(
        candidates, key=lambda c: df[c].astype(str).str.len().mean()
    )
    return text_col, label_col


def load_labeled_csv(
    path: Path, languages=("chinese", "english", "urdu")
) -> pd.DataFrame:
    """Read a labelled CSV and keep only the requested languages."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing labelled CSV: {path}")

    df = pd.read_csv(path, engine="python")
    text_col, label_col = detect_labeled_columns(df, languages)
    print(f"  columns detected: text={text_col!r}, label={label_col!r}")

    df = df[[text_col, label_col]].copy()
    df.columns = ["text", "lang"]
    df["lang"] = df["lang"].astype(str).str.strip().str.lower()

    wanted = {str(lang).strip().lower() for lang in languages}
    df = df[df["lang"].isin(wanted)].dropna(subset=["text"])

    frames = []
    for lang in sorted(wanted):
        subset = df[df["lang"] == lang]
        if subset.empty:
            raise ValueError(f"Language {lang!r} not present in {path.name}")
        frames.append(_lines_to_frame(subset["text"].astype(str).tolist(), lang))

    return pd.concat(frames, ignore_index=True)


def _has_three_file_layout(data_dir: Path) -> bool:
    return all(
        (Path(data_dir) / name).exists()
        for name in (ENGLISH_FILE, URDU_FILE, CHINESE_FILE)
    )


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------
def build_dataset(
    data_dir: Path = DATA_DIR,
    save_csv: Path | None = None,
    min_samples_per_lang: int | None = None,
    csv_path: Path | None = None,
    languages=("chinese", "english", "urdu"),
) -> pd.DataFrame:
    """Load, clean, de-duplicate and merge the corpora.

    Two input layouts are supported:

    1. three separate files (``english-corpus.txt``, ``urdu-corpus.txt``,
       ``english-chinese.csv``);
    2. a single labelled CSV such as the Kaggle / WiLI-2018 export
       (``dataset.csv``) - pass ``csv_path=`` explicitly, or just drop it into
       ``data/`` and it will be picked up automatically.
    """
    from .config import MIN_SAMPLES_PER_LANG

    min_samples_per_lang = min_samples_per_lang or MIN_SAMPLES_PER_LANG
    data_dir = Path(data_dir)

    if csv_path is not None:
        print(f"Loading labelled CSV: {csv_path}")
        df = load_labeled_csv(csv_path, languages)
    elif _has_three_file_layout(data_dir):
        print("Loading English corpus...")
        df_eng = load_english(data_dir)
        print("Loading Urdu corpus...")
        df_urdu = load_urdu(data_dir)
        print("Loading Chinese corpus...")
        df_chi = load_chinese_from_csv(data_dir)
        df = pd.concat([df_eng, df_urdu, df_chi], ignore_index=True)
    elif (data_dir / LABELED_CSV).exists():
        print(f"Loading labelled CSV: {data_dir / LABELED_CSV}")
        df = load_labeled_csv(data_dir / LABELED_CSV, languages)
    else:
        raise FileNotFoundError(
            "No usable dataset found.\n"
            f"Expected either the three files ({ENGLISH_FILE}, {URDU_FILE}, "
            f"{CHINESE_FILE}) or a labelled {LABELED_CSV} inside {data_dir}.\n"
            "See data/README.md for how to download the Kaggle dataset."
        )

    df = df[df["lang"].isin([str(l).lower() for l in languages])]

    # De-duplicate globally: the same sentence appearing twice would otherwise
    # land in both train and test splits and leak.
    before = len(df)
    df = df.drop_duplicates(subset=["text", "lang"]).reset_index(drop=True)
    if len(df) < before:
        print(f"  removed {before - len(df)} cross-language duplicate sample(s)")

    counts = df["lang"].value_counts().to_dict()
    print(f"Dataset counts: {counts}")
    for lang, count in counts.items():
        if count < min_samples_per_lang:
            raise ValueError(
                f"Language {lang!r} has only {count} samples "
                f"(minimum is {min_samples_per_lang})"
            )

    if save_csv is not None:
        save_csv = Path(save_csv)
        save_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_csv, index=False)
        print(f"Saved merged dataset -> {save_csv}")

    return df
