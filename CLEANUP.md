# Clean-up report: useful code vs. extra code

## 1. Useful code — kept and refactored

| What it does | Original location | Now lives in |
| --- | --- | --- |
| Load `english-corpus.txt` / `urdu-corpus.txt` / Chinese CSV | `backup training.py` §2 | `src/data.py` |
| Auto-detect the Chinese column in the CSV | all 4 scripts | `src/data.py::detect_chinese_column` |
| `clean_text()` (URL removal, whitespace, lower-case) | all 4 scripts | `src/data.py::clean_text` |
| Merge three corpora into one DataFrame | all 4 scripts | `src/data.py::build_dataset` |
| `CountVectorizer(analyzer="char", ngram_range=(1,4))` + `MultinomialNB(alpha=0.5)` | all 4 scripts | `src/model.py::build_pipeline` |
| Stratified 85/15 split, accuracy + classification report | all 4 scripts | `src/train.py` |
| Confusion matrix, per-class metrics, class distribution, top n-grams, probability histogram | all 4 scripts | `src/evaluate.py` |
| Misclassified sample dump | `Untitled-22.py` §7 | `src/evaluate.py::misclassified_examples` |
| `joblib` save / load | all 4 scripts | `src/model.py` |
| Tkinter GUI | `import joblib.py` | `src/gui.py` |
| Single-text prediction | `import joblib.py` | `src/predict.py` |

## 2. Extra code — removed from the pipeline (originals kept in `legacy/`)

| # | Extra / broken code | Why it was removed |
| --- | --- | --- |
| 1 | Four near-identical training scripts (`Training.py`, `Untitled-1.py`, `Untitled-22.py`, `backup training.py`) | 4 copies of the same 200 lines; impossible to review or maintain |
| 2 | Hard-coded paths `C:\Users\ALI\Desktop\...` (in 5 files, two different folders) | Crashes on any other machine; replaced by `src/config.py` + `LID_*` env vars |
| 3 | `plt.show()` called ~8× per run | Blocks on headless machines/CI and saves nothing; now figures are written to `outputs/*.png` |
| 4 | `vectorizer.fit(X_train)` **after** the pipeline was already fitted | Re-fits the same vectoriser a second time for no reason |
| 5 | `subset = X_train_vec[indices].toarray()` with `indices = y_train[y_train == lang].index` | **Bug:** pandas label index used as positional index on a matrix → wrong rows or `IndexError`. Fixed with a boolean mask |
| 6 | `.toarray()` on a whole class subset | Materialises a dense (samples × ~100k n-grams) matrix; replaced by sparse `.sum(axis=0)` |
| 7 | Chart titled "Accuracy per Language" plotting `precision` | Misleading metric; now a grouped Precision/Recall/F1 chart |
| 8 | Unused imports (`numpy`, `joblib`, `cross_val_score`, `learning_curve`, `WordCloud`) | Dead weight; `wordcloud` is now an optional dependency |
| 9 | Bigram-entropy + "top 5 trigrams" panel in the GUI | Interesting statistics, but not part of language identification; GUI now shows per-language probabilities |
| 10 | `final_dataset.csv` written to the repo root | Pollutes the working tree; artefacts now go to `outputs/` (git-ignored) |
| 11 | No de-duplication of corpus lines | Duplicate lines leak between train and test and inflate accuracy; duplicates are now dropped per language |
| 12 | No cross-validation, no persisted metrics | Added `--cv` k-fold scoring and `outputs/metrics.json` |
| 13 | Chinese detection = "any CJK char in the first 300 characters of a column" | Picks the wrong column on noisy CSVs; now the column with the **highest** CJK ratio (threshold 0.3) wins |
| 14 | GUI crashes with a traceback if the model file is missing | Friendly "Model not found" dialog + non-zero exit in the CLI |
| 15 | No way to run anything except editing the script | Added argparse CLIs: `src.train`, `src.predict`, `src.gui` |

## 3. Dataset

The project now trains from the [Kaggle "Language Identification" dataset](https://www.kaggle.com/datasets/zarajamshaid/language-identification-datasst) —
a 22-language × 1,000-paragraph excerpt of the **WiLI-2018** Wikipedia benchmark
(22,000 rows). Only the English, Urdu and Chinese rows are used; the other 19
languages are filtered out automatically.

Because the column headers differ between exports (`Text`/`language`,
`text`/`Language`, …), `src/data.py::detect_labeled_columns` now **detects both
columns** instead of hard-coding names. `scripts/prepare_dataset.py` can download
it via the Kaggle CLI and/or split it into the older three-file layout.

The original three-file input still works unchanged — see
[`data/README.md`](data/README.md).

## 4. Behaviour changes you should know about

* Class order is fixed to `chinese, english, urdu` (alphabetical, as scikit-learn
  sorts them), so label indices in saved models stay stable.
* Training refuses to run with fewer than 20 samples per language
  (`MIN_SAMPLES_PER_LANG`) instead of producing a silently broken model.
* English text is still lower-cased; Urdu and Chinese are left untouched
  (case is meaningless for those scripts).
