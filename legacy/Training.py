"""
FINAL LANGUAGE IDENTIFICATION PROJECT - IMPROVED VERSION
Languages: English, Urdu, Chinese
Model: Character n-grams + Multinomial Naive Bayes
Author: Your Name
"""

import os
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.pipeline import make_pipeline
import joblib

# ----------------------
# 1. CONFIG
# ----------------------
DATA_DIR = r"C:\Users\ALI\Desktop\Information retrival final project\dataset"
MODEL_SAVE_DIR = r"C:\Users\ALI\Desktop\Information retrival final project\Model"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# ----------------------
# 2. DATA LOADING FUNCTIONS
# ----------------------
def load_english():
    path = os.path.join(DATA_DIR, "english-corpus.txt")
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [l.strip() for l in f.read().splitlines() if len(l.strip()) > 1]
    return pd.DataFrame({"text": lines, "lang": "english"})

def load_urdu():
    path = os.path.join(DATA_DIR, "urdu-corpus.txt")
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [l.strip() for l in f.read().splitlines() if len(l.strip()) > 1]
    return pd.DataFrame({"text": lines, "lang": "urdu"})

def load_chinese_from_csv():
    path = os.path.join(DATA_DIR, "english-chinese.csv")
    df = pd.read_csv(path, encoding="utf-8", engine="python")
    chinese_col = None
    for col in df.columns:
        sample_text = "".join(df[col].astype(str))[:300]
        if any("\u4e00" <= ch <= "\u9fff" for ch in sample_text):
            chinese_col = col
            break
    if chinese_col is None:
        raise ValueError("Could not detect Chinese column in CSV")
    sentences = [s.strip() for s in df[chinese_col].dropna().astype(str) if len(s.strip()) > 1]
    return pd.DataFrame({"text": sentences, "lang": "chinese"})

# ----------------------
# 3. CLEAN TEXT
# ----------------------
def clean_text(s, lang):
    s = str(s).strip()
    s = re.sub(r"http\S+|www\S+|\S+@\S+", " ", s)
    s = re.sub(r"\s+", " ", s)
    if lang == "english":
        s = s.lower()
    return s.strip()

# ----------------------
# 4. BUILD DATASET
# ----------------------
def build_dataset():
    print("Loading English...")
    df_eng = load_english()
    print("Loading Urdu...")
    df_urdu = load_urdu()
    print("Loading Chinese...")
    df_chi = load_chinese_from_csv()

    df_eng["text"] = df_eng["text"].apply(lambda x: clean_text(x, "english"))
    df_urdu["text"] = df_urdu["text"].apply(lambda x: clean_text(x, "urdu"))
    df_chi["text"] = df_chi["text"].apply(lambda x: clean_text(x, "chinese"))

    df = pd.concat([df_eng, df_urdu, df_chi]).reset_index(drop=True)
    print("\nDataset counts:", df["lang"].value_counts().to_dict())
    df.to_csv("final_dataset.csv", index=False)
    print("Saved merged dataset → final_dataset.csv")
    return df

# ----------------------
# 5. TRAIN MODEL
# ----------------------
def train_model(df):
    X = df["text"]
    y = df["lang"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )

    vectorizer = CountVectorizer(analyzer="char", ngram_range=(1,4))
    model = MultinomialNB(alpha=0.5)
    pipeline = make_pipeline(vectorizer, model)

    print("\nTraining model...")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)

    # ----- Accuracy & Classification Report -----
    acc = accuracy_score(y_test, y_pred)
    print("\n=========== RESULTS ===========")
    print("Overall Accuracy:", acc)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # ----- Confusion Matrix -----
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(xticks_rotation=45, cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

    # ----- Accuracy per Language -----
    report = classification_report(y_test, y_pred, labels=model.classes_, output_dict=True)
    per_class_acc = {lang: report[lang]['precision'] for lang in model.classes_}
    plt.figure(figsize=(6,4))
    plt.bar(per_class_acc.keys(), per_class_acc.values(), color=["#0078D7","#E81123","#2D7D9A"])
    plt.ylim(0,1)
    plt.ylabel("Precision")
    plt.title("Precision per Language")
    plt.tight_layout()
    plt.show()

    # ----- Training Data Distribution -----
    train_counts = y_train.value_counts()
    plt.figure(figsize=(6,4))
    plt.bar(train_counts.index, train_counts.values, color=["#0078D7","#E81123","#2D7D9A"])
    plt.title("Training Data Distribution")
    plt.ylabel("Number of Samples")
    plt.tight_layout()
    plt.show()

    # ----- Top 10 N-grams per Language -----
    vectorizer.fit(X_train)
    X_train_vec = vectorizer.transform(X_train)
    vocab = np.array(vectorizer.get_feature_names_out())
    y_train_array = y_train.values  # convert to numpy
    for lang in model.classes_:
        mask = (y_train_array == lang)
        subset = X_train_vec[mask].toarray()  # fixed indexing issue
        ngram_counts = subset.sum(axis=0)
        top_indices = ngram_counts.argsort()[-10:][::-1]
        top_ngrams = vocab[top_indices]
        top_values = ngram_counts[top_indices]

        print(f"\nTop 10 n-grams for {lang}:")
        for ng, val in zip(top_ngrams, top_values):
            print(f"{ng}: {val}")

        plt.figure(figsize=(6,3))
        plt.bar(top_ngrams, top_values, color="#2D7D9A")
        plt.title(f"Top 10 N-grams for {lang}")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    # ----- Prediction Probability Distribution -----
    plt.figure(figsize=(6,4))
    for i, lang in enumerate(model.classes_):
        plt.hist(y_prob[:,i], bins=20, alpha=0.5, label=lang)
    plt.xlabel("Predicted Probability")
    plt.ylabel("Frequency")
    plt.title("Prediction Probability Distribution")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return pipeline

# ----------------------
# 6. MAIN
# ----------------------
if __name__ == "__main__":
    df = build_dataset()
    model_pipeline = train_model(df)

    model_path = os.path.join(MODEL_SAVE_DIR, "language_identifier_model.joblib")
    joblib.dump(model_pipeline, model_path)
    print(f"\nModel saved to: {model_path}")
    print("\nDONE ✔")
