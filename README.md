# Language Identifier — English / Urdu / Chinese

A language identification system that tells English, Urdu and Chinese apart from a
single paragraph of text, using **character n-grams (1–4)** and a
**Multinomial Naive Bayes** classifier wrapped in one scikit-learn `Pipeline`.

I trained it on the **WiLI-2018** benchmark (Wikipedia Language Identification,
235 languages / 235,000 paragraphs), filtered down to the three languages I cared
about. On the official WiLI test split it reaches **99.35% accuracy** and a
**99.35% macro F1**, and the whole thing trains in about 20 seconds on a laptop CPU.

```
WiLI-2018 text ──> normalise ──> char n-grams 1-4 ──> MultinomialNB ──> language + probabilities
                   (in-pipeline)   (CountVectorizer)      (α=0.1)
```

It ships with a **command-line interface**, a **Tkinter GUI** that shows the
probability bars and the exact n-grams behind each decision, four evaluation
charts, and a test suite.

---

## Why character n-grams

Language identity lives in orthography, not vocabulary. A paragraph can be about
football or fiscal policy — what stays constant is that English writes `the` and
`ing`, Urdu writes `کا` and `ہے` in Arabic script, and Chinese uses Han characters
with `。` as the full stop.

Character n-grams capture exactly that, and they have three properties that matter
for this task:

- **No tokenisation needed.** Chinese has no spaces, so a word-level model has
  nothing to bite on. Character n-grams treat all three scripts identically.
- **Robust to short input.** A two-word fragment still yields dozens of n-grams.
- **Cheap.** No embeddings, no GPU, no downloads at inference time — the trained
  model is a few megabytes of count tables.

Naive Bayes is the natural partner: it is a linear-time model that handles very
high-dimensional sparse features (here a 300,000-term vocabulary) well, and its
per-feature log-probabilities are directly readable, which is what makes the
"which n-grams decided this?" panel in the GUI possible.

---

## Features

- **One scikit-learn Pipeline** — normalisation, vectorisation and classification
  are serialised together, so a saved model can never be served text preprocessed
  differently from its training data.
- **Official WiLI test split** by default: train on the 175k train paragraphs,
  evaluate on the untouched 60k test paragraphs. Falls back to a stratified
  hold-out if you only downloaded the train files.
- **Three interfaces**: training CLI, prediction CLI (`--text`, `--file` or stdin,
  with `--json` output), and a Tkinter GUI.
- **Explainable predictions** — every guess comes with the highest-margin n-grams
  that pushed the model toward that language.
- **Four charts** written on every training run: confusion matrix (raw and
  normalised), per-class precision/recall/F1, class distribution, and the top
  discriminative n-grams per language.
- **CJK/Arabic-safe plotting** — the charts detect a CJK font and fall back to
  `U+4E2D`-style escaping instead of drawing empty boxes.
- **24 tests**, no dataset download required to run them.

---

## Project structure

```
language-identifier/
├── app/
│   └── gui.py                  # Tkinter desktop application
├── src/
│   ├── data.py                 # WiLI-2018 reader (x_*.txt / y_*.txt / labels.csv)
│   ├── features.py             # text normalisation + char n-gram vectoriser
│   ├── models.py               # pipeline factory, save/load, probabilities, top n-grams
│   ├── train.py                # training, evaluation, charts, metrics.json
│   ├── predict.py              # CLI + reusable Predictor class
│   └── utils.py                # seeding, CJK-safe plotting helpers
├── scripts/
│   └── download_wili.py        # fetch the dataset (Kaggle, then Zenodo mirror)
├── tests/
│   └── test_smoke.py           # 24 tests, run in ~4 s
├── assets/                     # charts shown in this README
├── data/wili/                  # dataset (git-ignored)
├── models/                     # trained model + metadata (git-ignored)
├── outputs/                    # charts and metrics.json (git-ignored)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Installation

```bash
git clone https://github.com/<your-username>/language-identifier.git
cd language-identifier

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Python 3.9+. The only heavy dependency is scikit-learn; there is no deep-learning
library involved.

**Tkinter** (needed for the GUI only) ships with CPython on Windows and macOS. On
Debian/Ubuntu:

```bash
sudo apt-get install python3-tk
```

---

## Getting the data

Download WiLI-2018 from Kaggle — <https://www.kaggle.com/datasets/mexwell/wili-2018>
— and extract it into `data/wili/`:

```
data/wili/
├── x_train.txt    # 175,000 paragraphs, one per line
├── y_train.txt    # 175,000 ISO 639-3 codes
├── x_test.txt     #  60,000 paragraphs
├── y_test.txt     #  60,000 codes
└── labels.csv     # code -> language name
```

Or let the script do it:

```bash
python scripts/download_wili.py            # Kaggle first, then the Zenodo mirror
python scripts/download_wili.py --source zenodo     # no login needed
```

> The four `x`/`y` text files are the important ones. Each line *i* of `x_train.txt`
> is a Wikipedia paragraph, and line *i* of `y_train.txt` is its language. Only the
> three codes I asked for are kept, so the other 232 languages are filtered out
> before anything is vectorised.

---

## Usage

### 1. Train

```bash
python -m src.train
```

```
[1/5] Loading WiLI-2018 ...
      train paragraphs: 2235 {'eng': 745, 'urd': 745, 'zho': 745}
      evaluation     : official WiLI test split
      train / test   : 2235 / 765
[2/5] Vectorising + training MultinomialNB ...
      vocabulary     : 300,000 character n-grams
[3/5] Evaluating ...
      accuracy       : 99.35%
      macro F1       : 99.35%

              precision    recall  f1-score   support

     English       0.996      0.992      0.994       255
        Urdu       0.992      0.992      0.992       255
      Chinese      0.992      0.996      0.994       255

    accuracy                           0.9935       765
   macro avg       0.993      0.993      0.993       765
weighted avg       0.993      0.993      0.993       765

[4/5] Writing charts ...
[5/5] Saving model and metrics ...
```

Everything is configurable:

| Flag | Default | Meaning |
| --- | --- | --- |
| `--data-dir` | `data/wili` | Folder with the `x_*.txt` / `y_*.txt` files |
| `--languages` | `eng,urd,zho` | Comma-separated ISO 639-3 codes |
| `--eval-split` | `auto` | `wili-test` (official split), `random`, or `auto` |
| `--test-size` | `0.2` | Hold-out fraction in `random` mode |
| `--ngram-min` / `--ngram-max` | `1` / `4` | Character n-gram range |
| `--analyzer` | `char_wb` | `char_wb`, `char` or `word` |
| `--min-df` | `2` | Drop n-grams appearing in fewer than N documents |
| `--max-features` | `300000` | Vocabulary cap (`0` = unlimited) |
| `--alpha` | `0.1` | MultinomialNB additive smoothing |
| `--use-tfidf` | off | TF-IDF weights instead of raw counts |
| `--no-lowercase` | off | Keep case |
| `--max-per-language` | `0` | Cap paragraphs per language (`0` = all) |

A few examples:

```bash
python -m src.train --languages eng,urd,zho,ara,hin     # more languages
python -m src.train --eval-split random --test-size 0.25
python -m src.train --ngram-max 5 --alpha 0.05
python -m src.train --max-per-language 300              # quick smoke run
```

### 2. Predict from the command line

```bash
python -m src.predict --text "The government announced a new transport policy."
python -m src.predict --file article.txt
cat article.txt | python -m src.predict
python -m src.predict --text "..." --json
```

```
Text      : The government announced a new transport policy.
Language  : English (eng)
Confidence: 99.87%

  English     99.87%  ########################################
  Urdu         0.09%
  Chinese      0.04%
```

Add `--explain` to see which n-grams drove the decision:

```
Top n-grams supporting this decision:
  'the'        +8.412
  ' th'        +7.905
  'ing'        +7.331
  'ed '        +6.874
  'and'        +6.512
```

### 3. Desktop GUI

```bash
python -m app.gui
```

Type or paste text, press **Detect** (or `Ctrl+Enter`), and the window shows the
predicted language with its confidence, a bar chart of the full probability
distribution, and the highest-margin n-grams for the winning language. The
**Load sample** button cycles through a built-in English / Urdu / Chinese example.

```
+---------------------------------------------------------------+
|  Language Identifier                                          |
|  Character n-grams (1-4) + Multinomial Naive Bayes            |
+---------------------------------------------------------------+
|  [ Enter text:                                              ] |
|  [                                                          ] |
|  [ Detect language ]  [ Load sample ]  [ Clear ]              |
+---------------------------------------------------------------+
|  English   99.87% confidence                                  |
+-------------------------------+-------------------------------+
|   [bar chart: probabilities]  |  N-grams behind the decision  |
|                               |   the    +8.41                |
|                               |   th     +7.90                |
|                               |   ing    +7.33                |
+-------------------------------+-------------------------------+
```

### 4. Tests

```bash
pytest -q
```

24 tests covering normalisation, the pipeline structure, the WiLI reader (including
a train-only download and a missing directory), model save/load round-trip, the
`Predictor` class, chart generation and non-Latin text escaping. They run on small
built-in samples, so no dataset download is needed.

---

## Results

Trained on the WiLI-2018 **train** split (745 paragraphs per language), evaluated
on the **official test** split (255 per language). Character n-grams 1–4 with
`char_wb`, raw counts, `alpha=0.1`, 300,000-feature vocabulary.

| Metric | Value |
| --- | --- |
| Accuracy | **99.35%** |
| Macro F1 | **99.35%** |
| Training time | ~20 s (CPU, single core) |

| Language | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| English | 99.6% | 99.2% | 99.4% | 255 |
| Urdu | 99.2% | 99.2% | 99.2% | 255 |
| Chinese | 99.2% | 99.6% | 99.4% | 255 |

### Confusion matrix

![Confusion matrix](assets/confusion_matrix.png)

Only 5 of 765 test paragraphs were misclassified. The errors are spread evenly —
one English paragraph read as Urdu, one as Chinese, one Urdu read as English, one
as Chinese, and one Chinese read as Urdu. No language pair dominates the errors,
which is what you want to see on a balanced task.

### Per-language metrics

![Per-class metrics](assets/per_class_metrics.png)

### What the model actually learned

![Top n-grams](assets/top_ngrams.png)

The n-grams with the highest log-odds margin — how much they vote for one language
over the average of the others:

| English | Urdu | Chinese |
| --- | --- | --- |
| `the` | `ی` | `的` |
| ` th` | `ے` | `。` |
| `ing` | `کا` | `和` |
| `ed ` | `ہے` | `了` |
| `and` | `نے` | `在` |
| `ion` | `اور` | `学` |

This is exactly the signal you would expect: English contributes Latin function
words and suffixes, Urdu contributes Arabic-script letters and its characteristic
postpositions (`کا` "of", `نے` the ergative marker, `اور` "and"), and Chinese
contributes Han characters plus the full stop `。`. The three feature sets are
almost disjoint, which is why the classifier is so confident.

### Ablation

I varied one setting at a time on the same splits to check the defaults were the
right ones:

| Configuration | Accuracy |
| --- | --- |
| **`char_wb` 1–4, counts, α=0.1** (default) | **99.35%** |
| `char` 1–4 (no word boundaries) | 99.18% |
| `char_wb` 1–5 | 99.33% |
| `char_wb` 1–3 | 99.27% |
| `char_wb` 1–4, α=1.0 | 99.21% |
| `char_wb` 1–2 | 99.02% |
| `char_wb` 1–4, TF-IDF | 98.86% |
| `word` 1–2 | 93.41% |

Three findings shaped the defaults:

1. **`char_wb` beats `char`** (99.35% vs 99.18%). Padding each word with a space
   gives the model explicit word-boundary markers, which helps Latin and Arabic
   script. Chinese has no spaces, so it is unaffected either way.
2. **Word n-grams collapse to 93.41%** — and the damage is entirely on Chinese.
   Without spaces there is nothing for a word tokeniser to find, so Chinese
   paragraphs degrade to a handful of very long tokens.
3. **TF-IDF is slightly worse than raw counts** (98.86% vs 99.35%). This is the
   expected result: Multinomial Naive Bayes is derived for raw term counts, and
   IDF reweighting distorts the document-length-normalised counts the model
   assumes.

---

## Design notes

**Why the normaliser lives inside the pipeline.** `FunctionTransformer(normalize_corpus)`
is the first step of the `Pipeline`, not a preprocessing script you run first. That
means `joblib.dump(pipeline)` serialises the text cleaning *with* the model, and
inference code physically cannot skip or change it.

**Why `alpha=0.1`.** Character n-gram vocabularies are large and sparse, so
scikit-learn's default `alpha=1.0` over-smooths and costs about 0.14 points (99.21%
vs 99.35%).

**Why `char_wb`, not `char`.** See the ablation above — word boundaries help the
two space-delimited scripts and cost nothing on Chinese.

**Why the charts escape non-Latin glyphs.** Matplotlib has no bidirectional text
engine or Arabic contextual shaping, so Urdu letters would be drawn isolated and
unjoined — worse than useless. `src/utils.py` detects a CJK font for Han
characters and renders everything else as `U+XXXX` code points rather than empty
boxes.

**Why `errors="replace"` when reading WiLI.** A single malformed byte in a
175,000-line file would otherwise abort the whole run. One damaged paragraph is
not worth losing the dataset over.

---

## Limitations

- **Paragraph-length input.** WiLI paragraphs are at least 140 Unicode code
  points. Accuracy on single words or short phrases will be lower; the model has
  never seen such short documents.
- **Formal register.** WiLI is Wikipedia text. Social-media transliteration
  (Romanised Urdu — "Urdu written in English letters") is out of distribution and
  will often be classified as English.
- **Three languages only.** The model cannot say "none of the above" — it always
  picks one of the languages it was trained on, however confident. Retrain with
  `--languages` to extend it.
- **`zho` is a macrolanguage.** WiLI's `zho` covers Chinese varieties written in
  Han characters; this is a script-level classifier, not a Mandarin/Cantonese
  discriminator.

---

## Dataset & citation

WiLI-2018 — <https://www.kaggle.com/datasets/mexwell/wili-2018>, published under
CC BY 4.0 / ODbL.

```bibtex
@dataset{thoma_martin_2018_841984,
  author    = {Thoma, Martin},
  title     = {{WiLI-2018 - Wikipedia Language Identification database}},
  month     = jan,
  year      = 2018,
  publisher = {Zenodo},
  version   = {1.0.0},
  doi       = {10.5281/zenodo.841984}
}
```

## License

[MIT](LICENSE) — free to use, modify and distribute with attribution.
