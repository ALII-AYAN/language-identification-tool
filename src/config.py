"""Central configuration for the language-identification project.

Every path can be overridden with an environment variable, so the code runs on
Windows / macOS / Linux and in CI without editing a single line.

    LID_DATA_DIR    where the corpora live            (default: <repo>/data)
    LID_MODEL_DIR   where the trained model is saved  (default: <repo>/models)
    LID_OUTPUT_DIR  where reports & plots are written (default: <repo>/outputs)
"""

from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

DATA_DIR: Path = Path(os.getenv("LID_DATA_DIR", PROJECT_ROOT / "data"))
MODEL_DIR: Path = Path(os.getenv("LID_MODEL_DIR", PROJECT_ROOT / "models"))
OUTPUT_DIR: Path = Path(os.getenv("LID_OUTPUT_DIR", PROJECT_ROOT / "outputs"))

MODEL_PATH: Path = MODEL_DIR / "language_identifier_model.joblib"
METRICS_PATH: Path = OUTPUT_DIR / "metrics.json"
DATASET_CSV: Path = OUTPUT_DIR / "final_dataset.csv"

# --------------------------------------------------------------------------
# Input file names (inside DATA_DIR)
# --------------------------------------------------------------------------
ENGLISH_FILE = "english-corpus.txt"
URDU_FILE = "urdu-corpus.txt"
CHINESE_FILE = "english-chinese.csv"

# Alternative input: the Kaggle "Language Identification" dataset
# (a 22-language subset of WiLI-2018), a single labelled CSV.
# https://www.kaggle.com/datasets/zarajamshaid/language-identification-datasst
LABELED_CSV = "dataset.csv"

# --------------------------------------------------------------------------
# Model hyper-parameters
# --------------------------------------------------------------------------
LABELS = ("chinese", "english", "urdu")  # alphabetical == sklearn's class order
RANDOM_STATE = 42
TEST_SIZE = 0.15
NGRAM_RANGE = (1, 4)  # character n-grams
ALPHA = 0.5  # MultinomialNB smoothing
MIN_SAMPLES_PER_LANG = 20  # refuse to train on a toy / broken dataset
