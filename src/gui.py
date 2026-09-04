"""Tkinter desktop app: type text, get the detected language + confidence.

Run with:  python -m src.gui            (or: python -m src.gui --model path/to/model.joblib)
"""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    import tkinter as tk
    from tkinter import messagebox, ttk
except ImportError as exc:  # pragma: no cover - environment dependent
    raise SystemExit(
        "tkinter is not available. It ships with CPython on Windows/macOS; "
        "on Debian/Ubuntu install it with:  sudo apt install python3-tk"
    ) from exc

from .config import MODEL_PATH
from .predict import predict, predict_proba

BG = "#f4f6f8"
ACCENT = "#0078D7"


class LanguageDetectorApp:
    def __init__(self, root: tk.Tk, model_path: Path):
        self.root = root
        self.model_path = Path(model_path)

        root.title("Language Detector - English / Urdu / Chinese")
        root.geometry("620x460")
        root.resizable(False, False)
        root.configure(bg=BG)

        style = ttk.Style()
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure("TButton", font=("Segoe UI", 12), padding=6,
                        background=ACCENT, foreground="white")
        style.map("TButton", background=[("active", "#005a9e")])
        style.configure("TLabel", font=("Segoe UI", 11), background=BG)

        tk.Label(root, text="Language Detector", font=("Segoe UI", 20, "bold"),
                 bg=BG, fg=ACCENT).pack(pady=(18, 5))
        tk.Label(root, text=f"Model: {self.model_path.name}", font=("Segoe UI", 9),
                 bg=BG, fg="#666").pack()

        tk.Label(root, text="Enter text:", bg=BG).pack(anchor="w", padx=30, pady=(12, 0))
        self.text_input = tk.Text(root, height=5, width=64, font=("Segoe UI", 11))
        self.text_input.pack(padx=30, pady=5)

        ttk.Button(root, text="Detect Language", command=self.detect).pack(pady=12)

        self.result_label = tk.Label(root, text="", font=("Segoe UI", 16, "bold"),
                                     bg=BG, fg=ACCENT)
        self.result_label.pack(pady=(5, 2))

        self.confidence_label = tk.Label(root, text="", font=("Segoe UI", 10),
                                         bg=BG, fg="#333")
        self.confidence_label.pack()

        self.bars_frame = tk.Frame(root, bg=BG)
        self.bars_frame.pack(fill="x", padx=40, pady=10)

    # ------------------------------------------------------------------
    def detect(self) -> None:
        text = " ".join(self.text_input.get("1.0", tk.END).split())
        if not text:
            messagebox.showwarning("Input Error", "Please enter some text to detect the language.")
            return

        try:
            lang = predict(text, self._model())
            scores = predict_proba(text, self._model())
        except FileNotFoundError as exc:
            messagebox.showerror("Model not found", str(exc))
            return

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        self.result_label.config(text=f"Detected language: {lang}")
        self.confidence_label.config(text=f"Confidence: {ranked[0][1]:.2%}")

        for widget in self.bars_frame.winfo_children():
            widget.destroy()
        for name, prob in ranked:
            row = tk.Frame(self.bars_frame, bg=BG)
            row.pack(fill="x", pady=2)
            tk.Label(row, text=name, width=9, anchor="w", bg=BG).pack(side="left")
            bar = ttk.Progressbar(row, orient="horizontal", length=320,
                                  mode="determinate", maximum=1.0)
            bar.pack(side="left", padx=6)
            bar["value"] = prob
            tk.Label(row, text=f"{prob:.3f}", bg=BG).pack(side="left")

    def _model(self):
        if not hasattr(self, "_cached_model"):
            from .model import load_model

            self._cached_model = load_model(self.model_path)
        return self._cached_model


def main() -> None:
    from .model import load_model

    parser = argparse.ArgumentParser(description="Launch the language detector GUI")
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    args = parser.parse_args()

    try:
        model = load_model(args.model)
    except FileNotFoundError as exc:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Model not found", str(exc))
        root.destroy()
        return

    root = tk.Tk()
    app = LanguageDetectorApp(root, args.model)
    app._cached_model = model
    root.mainloop()


if __name__ == "__main__":
    main()
