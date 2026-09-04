# Language Identification

Identify whether a piece of text is **English**, **Urdu** or **Chinese** using
character n-grams and a Multinomial Naive Bayes classifier.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange)
![accuracy](https://img.shields.io/badge/accuracy-99.3%25-brightgreen)
![tests](https://img.shields.io/badge/tests-7%20passing-brightgreen)
![license](https://img.shields.io/badge/license-MIT-green)

```
$ python -m src.predict "یہ ایک کتاب ہے"
Text      : یہ ایک کتاب ہے
Language  : urdu
  urdu     0.9998
  english  0.0002
  chinese  0.0000
```

## The idea

Language identification looks like it needs dictionaries, stop-word lists and
tokenizers per language. It mostly doesn't. This project only asks one question:
**which characters tend to follow which?**

Feed a classifier the character 1-to-4-grams of a sentence and the script gives
the answer away. Latin script means English, the Arabic-derived Perso-Arabic
script means Urdu, and logographic CJK means Chinese. No tokenizer, no
language-specific rules, and it still behaves sensibly on short or messy input
because it never depends on recognizing a real word.

## Results

Trained on 3,000 paragraphs (1,000 per language) from the
[Kaggle Language Identification dataset](https://www.kaggle.com/datasets/zarajamshaid/language-identification-datasst),
stratified 85/15 split, character 1–4 grams, `alpha=0.5`:

| Metric | Score |
| --- | --- |
| Hold-out accuracy | **0.9933** (447/450) |
| 5-fold CV accuracy | **0.9903 ± 0.0047** |
| Macro F1 | 0.9934 |
| Weighted F1 | 0.9933 |

| Language | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| chinese | 1.0000 | 1.0000 | 1.0000 | 150 |
| english | 0.9933 | 0.9867 | 0.9900 | 150 |
| urdu | 0.9868 | 0.9933 | 0.9900 | 150 |
| **macro avg** | 0.9934 | 0.9933 | 0.9933 | 450 |

Confusion matrix (rows = true, columns = predicted):

```
                predicted
              chi   eng   urd
true  chinese  150    0    0
      english    0  148    2
      urdu       0    1  149
```

Only 3 of 450 test paragraphs were wrong. All three are code-switching cases:
two English paragraphs quoted enough Urdu words to flip, and one Urdu paragraph
was mostly Latin-script transliteration. Dumping `outputs/misclassified.csv`
after a run is the fastest way to see these.

Most frequent character n-grams per language (straight counts, no TF-IDF):

| chinese | english | urdu |
| --- | --- | --- |
| `的` | ` ` (space) | ` ` (space) |
| `一` | `e` | `ا` |
| `是` | `t` | `ر` |
| `在` | `th` | `ک` |
| `国` | `he` | `ی` |

English and Urdu are dominated by spaces and their single most common letter,
which is exactly why the n-gram range matters: at 1-grams alone the two would
collide on whitespace, and the 3–4 gram features are what pull them apart.

## Features

- **One artifact** — a scikit-learn `Pipeline` (vectorizer + classifier) saved as
  a single `joblib` file. No separate vocabulary or config to ship alongside.
- **Two dataset layouts** — train straight from a labelled CSV, or from three
  separate corpus files. Column names and the Chinese column are auto-detected.
- **Headless by default** — every chart is written to `outputs/`. Nothing calls
  `plt.show()`, so it runs on a server and in CI.
- **Three entry points** — CLI training, CLI prediction (single text, file, CSV
  column, stdin, interactive) and a Tkinter desktop app.
- **Honest numbers** — per-language deduplication, stratified split, optional
  k-fold cross-validation, and a JSON dump of every metric.
- **Tests that don't need your data** — the suite builds a synthetic corpus and
  trains end to end.

## Project structure

```
language-identification/
├── src/
│   ├── config.py      # paths + hyper-parameters, overridable by env vars
│   ├── data.py        # loading, cleaning, dedup, dataset assembly
│   ├── model.py       # pipeline definition + save/load
│   ├── train.py       # training entry point
│   ├── evaluate.py    # metrics + report charts
│   ├── predict.py     # inference helpers + CLI
│   └── gui.py         # Tkinter desktop app
├── scripts/
│   └── prepare_dataset.py   # download / inspect / split the Kaggle dataset
├── tests/
│   └── test_smoke.py  # end-to-end tests on synthetic data
├── data/              # corpora go here (not committed)
├── models/            # trained model (not committed)
├── outputs/           # metrics, charts, error analysis (not committed)
├── legacy/            # older scripts, kept for reference
├── requirements.txt
└── README.md
```

## Installation

```bash
git clone https://github.com/<your-username>/language-identification.git
cd language-identification

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Requires Python 3.10+. The only real dependencies are scikit-learn, pandas,
numpy, joblib and matplotlib.

## Dataset

| | |
| --- | --- |
| Name | Language Identification dataset |
| Source | [kaggle.com/datasets/zarajamshaid/language-identification-datasst](https://www.kaggle.com/datasets/zarajamshaid/language-identification-datasst) |
| Origin | WiLI-2018, the Wikipedia Language Identification benchmark |
| Size | 22,000 rows — 22 languages × 1,000 paragraphs |
| Used here | English, Urdu, Chinese (1,000 paragraphs each) |

Set up the Kaggle CLI and fetch it:

```bash
pip install kaggle
# https://www.kaggle.com/settings -> "Create New Token" -> save as ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

python scripts/prepare_dataset.py --download --out data
python scripts/prepare_dataset.py --inspect         # per-language counts
```

No Kaggle account? Open the dataset page, hit Download, unzip, and drop
`dataset.csv` into `data/`. The other 19 languages are filtered out
automatically — only English, Urdu and Chinese rows are used.

## Usage

### Train

```bash
python -m src.train                          # char 1-4 grams, alpha=0.5, 15% hold-out
python -m src.train --cv 5                   # add 5-fold cross-validation
python -m src.train --no-plots               # metrics only, no PNGs
python -m src.train --ngram-max 3 --alpha 0.1 --test-size 0.2
python -m src.train --csv /path/to/dataset.csv
```

### Predict

```bash
python -m src.predict "今天我们学校有中文课"       # one sentence
python -m src.predict --file samples.txt          # one sample per line -> CSV
python -m src.predict --csv reviews.csv --column text
cat notes.txt | python -m src.predict --stdin
python -m src.predict --interactive
python -m src.gui                                 # desktop app
```

### Use as a library

```python
from src.predict import predict, predict_proba

predict("This is a test sentence")
# -> 'english'

predict_proba("یہ ایک کتاب ہے")
# -> {'chinese': 0.0, 'english': 0.0002, 'urdu': 0.9998}
```

## Outputs

Every run writes to `outputs/`:

| File | Contents |
| --- | --- |
| `metrics.json` | accuracy, per-class P/R/F1, CV score, hyper-parameters, top n-grams |
| `confusion_matrix.png` | where languages get confused |
| `metrics_per_language.png` | precision / recall / F1 per language |
| `class_distribution.png` | training samples per language |
| `probability_distribution.png` | how confident predictions are |
| `top_ngrams_<lang>.png` | most frequent character n-grams |
| `misclassified.csv` | every wrong prediction, for error analysis |
| `final_dataset.csv` | the cleaned, merged, deduplicated dataset |

## How it works

```python
CountVectorizer(analyzer="char", ngram_range=(1, 4))   # ~100k features
    -> MultinomialNB(alpha=0.5)
```

1. **Load** the corpora, strip URLs/emails, collapse whitespace, lowercase
   English (case is meaningless for Urdu and Chinese, so they're left alone).
2. **Deduplicate** per language — duplicate lines would otherwise land in both
   the train and test split and quietly inflate accuracy.
3. **Vectorize** into character n-gram counts. Sub-linear TF is off: raw counts
   suit a multinomial model.
4. **Split** 85/15, stratified so each language keeps its proportions.
5. **Fit** and evaluate, with optional k-fold cross-validation over the full set.

## Tests

```bash
pip install pytest
pytest -q
```

Seven tests build a synthetic corpus, train the pipeline end to end and check
accuracy, top-n-gram extraction, the prediction helpers, and the CSV column
auto-detection. They run in about two seconds and don't touch your real data.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `Trained model not found at ...` | run `python -m src.train` first |
| `No usable dataset found` | check the file names, or pass `--csv` / `--data-dir` |
| `No Chinese column found in ...` | make sure the CSV really holds Chinese text and is UTF-8 |
| Empty boxes in `top_ngrams_urdu.png` | install a font with Arabic coverage (e.g. Noto Nastaliq Urdu) |
| `ModuleNotFoundError: No module named 'tkinter'` | `sudo apt install python3-tk` on Linux (bundled on Windows/macOS) |

## Project history

This started as a handful of ad-hoc scripts with hard-coded Windows paths and
near-identical copies of the same training loop. It was consolidated into the
package above. A few things worth knowing if you compare against the old
versions:

- Paths are resolved relative to the project root and can be overridden with
  `LID_DATA_DIR`, `LID_MODEL_DIR` and `LID_OUTPUT_DIR`.
- The top-n-gram routine used a pandas label index to slice a matrix, which
  picked the wrong rows. It now uses a boolean mask on the sparse matrix.
- The chart labeled "Accuracy per Language" was plotting precision. It's now a
  grouped precision/recall/F1 chart.
- Training refuses to run below 20 samples per language instead of producing a
  silently broken model.

The originals are in `legacy/` for reference; delete the folder whenever you like.

## Credits

Dataset: [Language Identification dataset](https://www.kaggle.com/datasets/zarajamshaid/language-identification-datasst)
by Zara Jamshaid, an excerpt of WiLI-2018 — *Thoma, M. (2018). The WiLI
benchmark dataset for written language identification*, arXiv:1801.07779.

## License

Code is MIT — see [LICENSE](LICENSE). The dataset carries its own licence;
check the Kaggle page before redistributing it.
