import tkinter as tk
from tkinter import ttk, messagebox
import joblib
from collections import Counter
import math
import numpy as np

# ===========================
# LOAD TRAINED MODEL
# ===========================
model_path = r"C:\Users\ALI\Desktop\old windows data 33232\Information retrival final project\Model\language_identifier_model.joblib"
model = joblib.load(model_path)

# Extract vectorizer from pipeline
vectorizer = model.named_steps['countvectorizer']

# ===========================
# HELPER FUNCTIONS
# ===========================
def clean_input(text):
    """Remove newlines and extra spaces."""
    return " ".join(text.replace('\n', ' ').split())

def predict_language(text):
    return model.predict([text])[0]

def ngram_entropy(text, n=2):
    """Calculate character-level n-gram entropy, fallback to smaller n if needed."""
    text = text.replace(" ", "")
    if len(text) < n:
        n = 1
    ngrams = [text[i:i+n] for i in range(len(text)-n+1)]
    counts = Counter(ngrams)
    total = sum(counts.values())
    if total == 0:
        return 0.0
    entropy = -sum((freq/total) * math.log2(freq/total) for freq in counts.values())
    return round(entropy, 3)

def top_ngrams(text, top_n=5, n=3):
    """Return top character-level ngrams, fallback to smaller n if needed."""
    text = text.replace(" ", "")
    if len(text) < n:
        n = len(text)
    if n == 0:
        return ""
    ngrams = [text[i:i+n] for i in range(len(text)-n+1)]
    counts = Counter(ngrams)
    most_common = counts.most_common(top_n)
    return ", ".join([f"{gram}({freq})" for gram, freq in most_common])

# ===========================
# GUI APP
# ===========================
class LanguageDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Language Detector")
        self.root.geometry("550x400")
        self.root.resizable(False, False)
        self.root.configure(bg="#f4f6f8")

        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TButton", font=("Segoe UI", 12), padding=6,
                        background="#0078D7", foreground="white")
        style.map("TButton", background=[('active', '#005a9e')])
        style.configure("TLabel", font=("Segoe UI", 11), background="#f4f6f8")

        # Title
        self.title_label = tk.Label(root, text="Language Detector", font=("Segoe UI Bold", 20),
                                    bg="#f4f6f8", fg="#0078D7")
        self.title_label.pack(pady=20)

        # Input text
        self.input_label = tk.Label(root, text="Enter text:", bg="#f4f6f8")
        self.input_label.pack(pady=(10,0))
        self.text_input = tk.Text(root, height=5, width=60, font=("Segoe UI", 11))
        self.text_input.pack(pady=5)

        # Detect button
        self.detect_btn = ttk.Button(root, text="Detect Language", command=self.detect_language)
        self.detect_btn.pack(pady=15)

        # Result labels
        self.result_label = tk.Label(root, text="", font=("Segoe UI Bold", 14),
                                     bg="#f4f6f8", fg="#0078D7")
        self.result_label.pack(pady=5)
        self.entropy_label = tk.Label(root, text="", font=("Segoe UI", 11),
                                      bg="#f4f6f8", fg="#333")
        self.entropy_label.pack()
        self.ngram_label = tk.Label(root, text="", font=("Segoe UI", 11),
                                    bg="#f4f6f8", fg="#333")
        self.ngram_label.pack()

    def animate_text(self, label, text):
        """Animate text appearing letter by letter."""
        label.config(text="")
        for i in range(len(text)+1):
            self.root.after(i*30, lambda t=text[:i]: label.config(text=t))

    def detect_language(self):
        user_text = clean_input(self.text_input.get("1.0", tk.END))
        if not user_text:
            messagebox.showwarning("Input Error", "Please enter some text to detect the language.")
            return

        lang = predict_language(user_text)
        entropy = ngram_entropy(user_text, n=2)
        top_ng = top_ngrams(user_text, top_n=5, n=3)

        self.animate_text(self.result_label, f"Detected Language: {lang}")
        self.entropy_label.config(text=f"Bigram Entropy: {entropy}")
        self.ngram_label.config(text=f"Top 5 Trigrams: {top_ng}")

# ===========================
# RUN APP
# ===========================
if __name__ == "__main__":
    root = tk.Tk()
    app = LanguageDetectorApp(root)
    root.mainloop()
