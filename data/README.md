# Data folder

This project accepts **two input layouts**. Pick whichever you have.

## Option A — single labelled CSV (recommended, Kaggle / WiLI-2018)

Download the dataset and keep `dataset.csv` here:

| | |
| --- | --- |
| Source | <https://www.kaggle.com/datasets/zarajamshaid/language-identification-datasst> |
| Origin | an excerpt of the **WiLI-2018** Wikipedia Language Identification benchmark ([arXiv:1801.07779](https://arxiv.org/abs/1801.07779)) |
| Rows | 22,000 (22 languages × 1,000 paragraphs) |
| Columns | one text column + one label column — the headers vary between exports (`Text`/`language`, `text`/`Language`, …), so **both columns are auto-detected** |
| Languages | Arabic, Chinese, Dutch, English, Estonian, French, Hindi, Indonesian, Japanese, Korean, Latin, Persian, Pushto, Portugese, Romanian, Russian, Spanish, Swedish, Tamil, Thai, Turkish, Urdu |

This project uses **English, Urdu and Chinese**; the other 19 languages are
filtered out automatically.

### Download

```bash
pip install kaggle
# create an API token: https://www.kaggle.com/settings -> "Create New Token"
# place the downloaded kaggle.json in ~/.kaggle/ and run: chmod 600 ~/.kaggle/kaggle.json

python scripts/prepare_dataset.py --download --out data
```

No Kaggle account? Open the dataset page, click **Download**, unzip, and drop
`dataset.csv` into this folder manually.

### Use it

```bash
python -m src.train                          # dataset.csv is auto-detected
python -m src.train --csv /path/dataset.csv  # or point at it explicitly
python scripts/prepare_dataset.py --inspect  # see the per-language counts
```

## Option B — three separate files

| File | Format | Notes |
| --- | --- | --- |
| `english-corpus.txt` | one sentence / document per line, UTF-8 | lower-cased during cleaning |
| `urdu-corpus.txt` | one sentence / document per line, UTF-8 | |
| `english-chinese.csv` | any CSV with at least one Chinese column | the Chinese column is auto-detected by CJK character ratio |

Convert Option A into Option B at any time:

```bash
python scripts/prepare_dataset.py --csv data/dataset.csv --out data
```

## Keeping the data elsewhere

The corpora are **not** committed to git (see `.gitignore`) — they are large and
often separately licensed. Point the code at another folder instead:

```bash
export LID_DATA_DIR=/path/to/your/dataset      # Linux / macOS
set LID_DATA_DIR=C:\path\to\your\dataset       # Windows CMD
python -m src.train
```

## Licensing

The original data comes from Wikipedia; the Kaggle mirror is provided by
Zara Jamshaid. Check the dataset page for the current licence terms before
redistributing it, and cite WiLI-2018 if you publish results obtained with it.
