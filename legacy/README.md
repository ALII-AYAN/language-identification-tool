# Legacy / experimental scripts (extra code)

These are the **original scripts exactly as they were before the clean-up**.
They are kept only for reference and are **not imported by the package**.

| File | Status |
| --- | --- |
| `Training.py` | duplicate of `backup training.py` (+ one indexing bug fix) |
| `Untitled-1.py` | duplicate, hard-coded path `...\old windows data 33232\...` |
| `Untitled-22.py` | duplicate + unused `wordcloud` / `learning_curve` / `cross_val_score` imports |
| `backup training.py` | duplicate, hard-coded path `...\Information retrival final project\...` |
| `import joblib.py` | the original Tkinter GUI (hard-coded model path) |

Safe to delete once you have verified the new pipeline:

```bash
rm -rf legacy
```

See [`CLEANUP.md`](../CLEANUP.md) for what was kept, what was dropped, and why.
