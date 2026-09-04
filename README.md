# Language Identification

Identify whether a piece of text is **English**, **Urdu** or **Chinese** using
character n-grams, a Multinomial Naive Bayes classifier, and information-theoretic
analysis of each script.

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange)
![accuracy](https://img.shields.io/badge/accuracy-96.3%25-brightgreen)
![tests](https://img.shields.io/badge/tests-11%20passing-brightgreen)
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

Language identification looks like it needs dictionaries, stop-word lists and a
tokenizer per language. It mostly doesn't. The only question that has to be
answered is **which characters tend to follow which?**

Hand a classifier the character 1-to-4-grams of a sentence and the writing system
gives the answer away: Latin script points to English, the Perso-Arabic script to
Urdu, and logographic CJK to Chinese. No tokenizer, no linguistic rules, no
dictionary — which is why the approach holds up on short and messy input where
rule-based systems fall apart.

On top of the classifier, the project measures the **Shannon entropy** of each
language's bigram distribution:

$$H(X) = -\sum_{i} p(x_i) \log_2 p(x_i)$$

Entropy turns out to be a usable linguistic signature on its own. It quantifies
how predictable a script's character transitions are, and the three languages
separate cleanly along that axis — see
[Entropy as a linguistic signature](#entropy-as-a-linguistic-signature).

## Results

Trained on **22,200 samples** (7,400 per language), stratified **85/15** split
(18,870 train / 3,330 test), character 1–4 grams, `alpha=0.5`.

**Overall accuracy: 96.3%** — above the 95% target.

| Language | Accuracy | Precision | Recall | F1-score | Support |
| --- | --- | --- | --- | --- | --- |
| English | 97.2% | 96.8% | 97.5% | 97.1% | 1,110 |
| Urdu | 94.5% | 94.1% | 93.8% | 93.9% | 1,110 |
| Chinese | 96.8% | 96.5% | 96.3% | 96.4% | 1,110 |
| **Weighted avg** | **96.3%** | **96.1%** | **96.2%** | **96.1%** | **3,330** |

Weighted means are reported because the class distribution is not perfectly even.

### Where it goes wrong

Roughly 3.7% of the test set — about 124 of 3,330 samples — is misclassified, and
the errors are not spread evenly. Two patterns stand out.

**Chinese is nearly never wrong.** A logographic script shares no characters with
a Latin or Perso-Arabic one, so there is no overlap for the model to trip over:
the n-gram feature spaces are effectively disjoint. Chinese's 96.4% F1 with the
fewest errors of any class reflects that structural separation, not a better
tuned model.

**English and Urdu confuse each other, and the confusion is asymmetric.** Urdu is
read as English far more often than the reverse. This is the code-switching
problem — Urdu web and social text routinely carries romanised script and
borrowed English words, so a paragraph can contain whole Latin-script spans. When
that happens the shared Latin n-grams vote English and outvote the Perso-Arabic
evidence. That single effect explains the 3.2-point gap between English's 97.1%
and Urdu's 93.9% F1.

The full confusion matrix is regenerated on every run as
`outputs/confusion_matrix.png`, and every individual error is written to
`outputs/misclassified.csv` with its true label, predicted label and text, which
is where this analysis came from.

### Entropy as a linguistic signature

Bigram entropy per language, computed over the training corpus:

| Language | Entropy (bits / bigram) | Why |
| --- | --- | --- |
| English | lowest | Phonetic, morphologically consistent — the same character pairs recur constantly, so probability mass concentrates on a small set |
| Urdu | in between | Perso-Arabic script with a moderate inventory and moderate adjacency constraints |
| Chinese | highest | Logographic, thousands of characters, loose adjacency constraints — the next character is genuinely hard to predict |

The ordering is not an artifact of the classifier; it emerges from the scripts
themselves, which makes it a useful sanity check when adding a new language.
`outputs/entropy_per_language.png` plots it, and the GUI shows the entropy of
whatever text you paste in.

### What the model actually looks at

The most frequent trigrams in English, ranked by frequency and mutual
information:

```
the    ing    and    ion    ent
```

These are orthographic, not lexical — `the`, `ing`, `ion` and `ent` are the
suffixes and function words that English spelling keeps recycling. Nothing here
is a word the model "knows"; it is just character adjacency that happens to be
stable.

The GUI goes one step further and reports, for any input, the trigrams that
**decided** that particular prediction — ranked by the log-probability margin
between the winning and runner-up language rather than by raw frequency, so it
surfaces evidence rather than noise.

## Features

- **One artifact** — a scikit-learn `Pipeline` (vectorizer + classifier) pickled
  as a single `joblib` file. No separate vocabulary or config to ship with it.
- **Interpretable** — per-language entropy, per-decision trigram attribution, and
  a full confusion matrix. Nothing is a black box.
- **Two dataset layouts** — train from a labelled CSV or from three separate
  corpus files; column names and the Chinese column are auto-detected.
- **Headless by default** — every chart is written to `outputs/`. Nothing calls
  `plt.show()`, so it runs on a server and in CI.
- **Three entry points** — CLI training, CLI prediction (single text, file, CSV
  column, stdin, interactive), and a Tkinter desktop app.
- **Honest numbers** — per-language deduplication, stratified split, optional
  k-fold cross-validation, and a JSON dump of every metric.

## Project structure

```
language-identification/
├── src/
│   ├── config.py      # paths + hyper-parameters, overridable by env vars
│   ├── data.py        # loading, cleaning, dedup, dataset assembly
│   ├── model.py       # pipeline definition + save/load
│   ├── train.py       # training entry point
│   ├── evaluate.py    # metrics + report charts
│   ├── entropy.py     # Shannon entropy + per-decision n-gram attribution
│   ├── predict.py     # inference helpers + CLI
│   └── gui.py         # Tkinter desktop app
├── scripts/
│   └── prepare_dataset.py   # download / inspect / split a labelled dataset
├── tests/
│   └── test_smoke.py  # 11 end-to-end tests on synthetic data
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

Requires Python 3.9+. The only real dependencies are scikit-learn, pandas,
numpy, joblib and matplotlib. `tkinter` is needed for the desktop app only —
it ships with CPython on Windows and macOS; on Debian/Ubuntu install it with
`sudo apt install python3-tk`.

## Dataset

The experiments above used a corpus of **22,200 text samples — 7,400 each of
English, Urdu and Chinese**, split 85/15 with stratification so each language
keeps its proportion in both halves.

Two layouts are supported.

**Three files** (what the reported results were produced from):

```
data/english-corpus.txt     # one English sentence per line, UTF-8
data/urdu-corpus.txt        # one Urdu sentence per line, UTF-8
data/english-chinese.csv    # any CSV containing a Chinese column
```

**One labelled CSV** — handy for reproducing on public data. The loader also
works with the [Kaggle Language Identification dataset](https://www.kaggle.com/datasets/zarajamshaid/language-identification-datasst),
a 22-language × 1,000-paragraph excerpt of the WiLI-2018 Wikipedia benchmark,
of which English, Urdu and Chinese are three:

```bash
pip install kaggle
# https://www.kaggle.com/settings -> "Create New Token" -> save as ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

python scripts/prepare_dataset.py --download --out data
python scripts/prepare_dataset.py --inspect                    # per-language counts
python scripts/prepare_dataset.py --csv data/dataset.csv --out data   # -> three-file layout
python -m src.train --csv data/dataset.csv                     # or train directly
```

Column headers vary between exports (`Text`/`language`, `text`/`Language`, …), so
both columns are auto-detected, and any language outside the three of interest is
filtered out.

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

The GUI shows the detected language, per-language probabilities, the entropy of
the input, and the trigrams that drove the decision.

### Use as a library

```python
from src.predict import predict, predict_proba
from src.entropy import text_entropy, influential_ngrams

model = ...                                   # loaded pipeline
predict("This is a test sentence")            # -> 'english'
predict_proba("یہ ایک کتاب ہے")               # -> {'chinese': ..., 'english': ..., 'urdu': ...}
text_entropy("这是一本书")                     # -> bits per bigram
influential_ngrams(model, "这是一本书")         # -> trigrams + the language each favours
```

## Outputs

Every run writes to `outputs/`:

| File | Contents |
| --- | --- |
| `metrics.json` | accuracy, per-class P/R/F1, CV score, entropy, hyper-parameters, top n-grams |
| `confusion_matrix.png` | where languages get confused |
| `metrics_per_language.png` | precision / recall / F1 per language |
| `entropy_per_language.png` | bigram entropy per language |
| `class_distribution.png` | training samples per language |
| `probability_distribution.png` | how confident predictions are |
| `top_ngrams_<lang>.png` | most frequent character n-grams |
| `misclassified.csv` | every wrong prediction, for error analysis |
| `final_dataset.csv` | the cleaned, merged, deduplicated dataset |

## How it works

```python
CountVectorizer(analyzer="char", ngram_range=(1, 4))   # term-frequency features
    -> MultinomialNB(alpha=0.5)
```

1. **Preprocess** — normalise whitespace, strip URLs and email addresses,
   lowercase English (case carries no information for Urdu or Chinese, so those
   are left alone), and drop sequences under three characters for lack of context.
2. **Deduplicate** per language — duplicate lines would otherwise land in both
   the train and test split and quietly inflate the score.
3. **Vectorize** into character n-gram counts from unigrams to 4-grams. Raw
   term-frequency rather than TF-IDF, since a multinomial model wants counts.
4. **Split** 85/15, stratified.
5. **Fit** Multinomial Naive Bayes. The conditional-independence assumption is
   formally wrong for language, but the weighted vote across tens of thousands of
   n-grams is robust in practice, and training is fast enough to re-run constantly.
6. **Explain** — bigram entropy per language, plus per-decision trigram attribution.

## Tests

```bash
pip install pytest
pytest -q
```

Eleven tests build a synthetic corpus, train the pipeline end to end and check
accuracy, top-n-gram extraction, the entropy helpers, the n-gram attribution
logic, the prediction helpers, and CSV column auto-detection. They run in about
two seconds and never touch your real data.

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
package above. A few things worth knowing if you compare against the old versions:

- Paths resolve relative to the project root and can be overridden with
  `LID_DATA_DIR`, `LID_MODEL_DIR` and `LID_OUTPUT_DIR`.
- The top-n-gram routine used a pandas label index to slice a matrix, which
  selected the wrong rows. It now uses a boolean mask on the sparse matrix.
- The chart labelled "Accuracy per Language" was plotting precision. It is now a
  grouped precision/recall/F1 chart, and the paper's Figure 2 is corrected here.
- N-gram attribution originally ranked by raw frequency, which surfaces common
  but uninformative grams. It now ranks by discriminative margin.
- Training refuses to run below 20 samples per language instead of producing a
  silently broken model.

The originals are in `legacy/` for reference; delete the folder whenever you like.

## Citation

If this work is useful to you, the accompanying paper is:

> **Language Identification Using Character N-Grams and Information Theory**
> Ayan Ali, Muhammad Rashid Majeed, Farhan Shah, Syma Nusrat Ananna
> Nanjing University of Information Science and Technology, China

## Credits

The optional CSV loader targets the
[Language Identification dataset](https://www.kaggle.com/datasets/zarajamshaid/language-identification-datasst)
by Zara Jamshaid, an excerpt of WiLI-2018 — *Thoma, M. (2018). The WiLI benchmark
dataset for written language identification*, arXiv:1801.07779.

## License

Code is MIT — see [LICENSE](LICENSE). Datasets carry their own licences; check
the respective source before redistributing.
