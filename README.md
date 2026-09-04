# Language Identification — English / Urdu / Chinese

A small, reproducible text-classification project: given a sentence, decide
whether it is **English**, **Urdu** or **Chinese** using character n-grams
(1–4) and a Multinomial Naive Bayes classifier.

```
Input :  "یہ ایک کتاب ہے"
Output:  urdu  (p = 0.998)
```

## Why character n-grams?

No tokenizer, no stop-word list, no language-specific rules: the model only
looks at *which characters follow which*. That is enough to separate a Latin
script (English), an Arabic-derived script (Urdu) and a logographic script
(Chinese), and it degrades gracefully on short or noisy input.

## Features

- Single scikit-learn `Pipeline` (vectoriser + classifier) — one `joblib` file,
  no separate vocabulary to ship.
- Configurable paths via environment variables, so it runs on Windows/macOS/Linux/CI.
- Headless evaluation: every chart is **saved** to `outputs/`, nothing calls `plt.show()`.
- CLI for training, batch prediction and interactive use, plus a Tkinter desktop app.
- Deduplication + stratified split + optional k-fold cross-validation, so the
  reported accuracy actually means something.
- Smoke tests that run without the real dataset (`pytest -q`).

## Project structure

```
language-identification/
├── src/
│   ├── config.py      # paths + hyper-parameters (env-overridable)
│   ├── data.py        # loading, cleaning, de-duplication, dataset assembly
│   ├── model.py       # pipeline definition + save/load
│   ├── train.py       # training entry point (CLI)
│   ├── evaluate.py    # metrics + report charts
│   ├── predict.py     # inference helpers + CLI
│   └── gui.py         # Tkinter desktop app
├── scripts/
│   └── prepare_dataset.py   # download / split the Kaggle dataset
├── tests/
│   └── test_smoke.py  # end-to-end test on synthetic data
├── data/              # <- put the corpora here (not committed)
├── models/            # <- trained model (not committed)
├── outputs/           # <- metrics, charts, misclassified samples (not committed)
├── legacy/            # original scripts, kept for reference only
├── requirements.txt
├── CLEANUP.md         # what was kept, what was dropped, and why
└── README.md
```

## Quick start

```bash
git clone <your-repo-url> language-identification
cd language-identification

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 1. Add the data

**Recommended** — the [Kaggle Language Identification dataset](https://www.kaggle.com/datasets/zarajamshaid/language-identification-datasst),
a 22-language × 1,000-paragraph excerpt of the **WiLI-2018** Wikipedia
benchmark ([paper](https://arxiv.org/abs/1801.07779)). This project uses its
English, Urdu and Chinese rows.

```bash
pip install kaggle          # then add ~/.kaggle/kaggle.json (chmod 600)
python scripts/prepare_dataset.py --download --out data
# or: download dataset.csv from the dataset page and drop it into data/
```

Prefer your own corpora? Three separate files also work (see
[`data/README.md`](data/README.md)):

```
data/english-corpus.txt     # one English sentence per line
data/urdu-corpus.txt        # one Urdu sentence per line
data/english-chinese.csv    # any CSV containing a Chinese column
```

Keep the data elsewhere? Point the code at it:

```bash
export LID_DATA_DIR=/path/to/dataset        # Windows CMD: set LID_DATA_DIR=...
```

### 2. Train

```bash
python -m src.train                 # defaults: char 1-4 grams, alpha=0.5, 15% hold-out
python -m src.train --cv 5          # additionally run 5-fold cross-validation
python -m src.train --no-plots      # metrics only, no PNGs
python -m src.train --ngram-max 3 --alpha 0.1 --test-size 0.2
python -m src.train --csv /path/to/dataset.csv    # train from an explicit CSV
```

### 3. Use it

```bash
# one sentence
python -m src.predict "今天我们学校有中文课"

# a whole file (one sample per line) -> <file>_predicted.csv
python -m src.predict --file samples.txt

# a CSV column -> <file>_predicted.csv
python -m src.predict --csv reviews.csv --column text

# interactive
python -m src.predict --interactive
```

As a library:

```python
from src.predict import predict, predict_proba

predict("This is a test sentence")          # -> 'english'
predict_proba("یہ ایک کتاب ہے")             # -> {'chinese': 0.0, 'english': 0.0, 'urdu': 1.0}
```

Desktop app:

```bash
python -m src.gui                            # or: python -m src.gui --model models/other.joblib
```

## CLI reference

| Command | Key options |
| --- | --- |
| `python -m src.train` | `--csv --data-dir --model-out --output-dir --test-size --ngram-min --ngram-max --alpha --cv --top-n --no-plots --random-state` |
| `python -m src.predict` | `TEXT... --file --csv --column --out --stdin --interactive --model` |
| `python -m src.gui` | `--model` |
| `python scripts/prepare_dataset.py` | `--csv --out --languages --inspect --download` |

## Dataset

| | |
| --- | --- |
| Name | Language Identification dataset |
| Source | [kaggle.com/datasets/zarajamshaid/language-identification-datasst](https://www.kaggle.com/datasets/zarajamshaid/language-identification-datasst) |
| Origin | WiLI-2018, the Wikipedia Language Identification benchmark ([arXiv:1801.07779](https://arxiv.org/abs/1801.07779)) |
| Size | 22,000 rows — 22 languages × 1,000 paragraphs |
| Used here | English, Urdu, Chinese (1,000 paragraphs each) |

Column headers vary between exports, so the text and label columns are
**auto-detected**; every other language is filtered out. Forks adding more
languages only need to pass `--languages` / extend `LABELS` in `src/config.py`.

## Generated artefacts

After training, `outputs/` contains:

| File | Content |
| --- | --- |
| `metrics.json` | accuracy, per-class P/R/F1, CV score, hyper-parameters, top n-grams |
| `confusion_matrix.png` | where the model confuses languages |
| `metrics_per_language.png` | Precision / Recall / F1 per language |
| `class_distribution.png` | training samples per language |
| `probability_distribution.png` | how confident the model is |
| `top_ngrams_<lang>.png` | most frequent character n-grams per language |
| `misclassified.csv` | every wrong prediction for error analysis |
| `final_dataset.csv` | the cleaned, merged, de-duplicated dataset |

## Results

Fill in your own numbers after running `python -m src.train --cv 5`:

| Metric | Value |
| --- | --- |
| Hold-out accuracy (15%) | _e.g. 0.99_ |
| 5-fold CV accuracy | _e.g. 0.99 ± 0.01_ |
| Macro F1 | _e.g. 0.99_ |

| Language | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| english | | | | |
| urdu | | | | |
| chinese | | | | |

## Tests

```bash
pip install pytest
pytest -q
```

The suite builds a tiny synthetic corpus, trains the model end to end and
checks accuracy, top n-grams and the prediction helpers — it does **not**
need your real dataset.

## Troubleshooting

| Problem | Fix |
| --- | --- |
| `Trained model not found at ...` | run `python -m src.train` first |
| `No Chinese column found in english-chinese.csv` | make sure the CSV really contains Chinese text and is UTF-8 |
| `FileNotFoundError: Missing ... corpus` | check the file names / `LID_DATA_DIR` |
| Empty boxes in `top_ngrams_urdu.png` | install a font with Arabic/Urdu coverage (e.g. Noto Nastaliq Urdu) |
| `ModuleNotFoundError: No module named 'tkinter'` | `sudo apt install python3-tk` (Linux; it is bundled on Windows/macOS) |

## What changed vs. the original scripts

The project started as four near-identical scripts with hard-coded Windows
paths and a few indexing bugs. [`CLEANUP.md`](CLEANUP.md) lists exactly which
code was kept, which was dropped, and the bugs that were fixed.

## Roadmap

- [ ] Add more of the 22 WiLI languages (Arabic, Hindi, Japanese, …) — the pipeline is language-agnostic
- [ ] Compare against a linear SVM / logistic regression baseline
- [ ] `pyproject.toml` + published package
- [ ] REST API (FastAPI) wrapper around `src.predict`

## Credits

* Dataset: [Language Identification dataset](https://www.kaggle.com/datasets/zarajamshaid/language-identification-datasst)
  by Zara Jamshaid, an excerpt of **WiLI-2018** —
  *Thoma, M. (2018). The WiLI benchmark dataset for written language identification.* [arXiv:1801.07779](https://arxiv.org/abs/1801.07779)

## License

Code: MIT — see [LICENSE](LICENSE).
The dataset keeps its own licence; check the Kaggle page before redistributing it.
