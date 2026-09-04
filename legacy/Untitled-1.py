"""
FINAL LANGUAGE IDENTIFICATION PROJECT
Languages: English, Urdu, Chinese
Dataset Source: Local corpora (provided by user)
Model: Character n-grams + Multinomial Naive Bayes
Author: Your Name
"""

import os
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import numpy as np
import joblib

# -------------------------------------------------------
# 1. CONFIG — YOUR DATASET DIRECTORY
# -------------------------------------------------------
data_dir = r"C:\Users\ALI\Desktop\old windows data 33232\Information retrival final project\dataset"

# -------------------------------------------------------
# 2. LOADING LOCAL FILES
# -------------------------------------------------------
def load_english():
    path = os.path.join(data_dir, "english-corpus.txt")
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.read().splitlines()
    lines = [l.strip() for l in lines if len(l.strip()) > 1]
    return pd.DataFrame({"text": lines, "lang": "english"})

def load_urdu():
    path = os.path.join(data_dir, "urdu-corpus.txt")
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.read().splitlines()
    lines = [l.strip() for l in lines if len(l.strip()) > 1]
    return pd.DataFrame({"text": lines, "lang": "urdu"})

def load_chinese_from_csv():
    path = os.path.join(data_dir, "english-chinese.csv")
    df = pd.read_csv(path, encoding="utf-8", engine="python")

    chinese_col = None
    for col in df.columns:
        sample_text = "".join(df[col].astype(str))[:300]
        if any("\u4e00" <= ch <= "\u9fff" for ch in sample_text):
            chinese_col = col
            break

    if chinese_col is None:
        raise ValueError("Could not detect Chinese column in english-chinese.csv")

    chinese_sentences = df[chinese_col].dropna().astype(str).tolist()
    chinese_sentences = [s.strip() for s in chinese_sentences if len(s.strip()) > 1]

    return pd.DataFrame({"text": chinese_sentences, "lang": "chinese"})

# -------------------------------------------------------
# 3. TEXT CLEANING
# -------------------------------------------------------
def clean_text(s, lang):
    s = str(s).strip()
    s = re.sub(r"http\S+|www\S+|\S+@\S+", " ", s)
    s = re.sub(r"\s+", " ", s)
    if lang == "english":
        s = s.lower()
    return s.strip()

# -------------------------------------------------------
# 4. BUILD MERGED DATASET
# -------------------------------------------------------
def build_dataset():
    print("Loading English...")
    df_eng = load_english()

    print("Loading Urdu...")
    df_urdu = load_urdu()

    print("Loading Chinese from CSV...")
    df_chi = load_chinese_from_csv()

    # Clean
    df_eng["text"] = df_eng["text"].apply(lambda x: clean_text(x, "english"))
    df_urdu["text"] = df_urdu["text"].apply(lambda x: clean_text(x, "urdu"))
    df_chi["text"] = df_chi["text"].apply(lambda x: clean_text(x, "chinese"))

    # Combine
    df = pd.concat([df_eng, df_urdu, df_chi]).reset_index(drop=True)

    print("\nFinal dataset counts:")
    print(df["lang"].value_counts().to_dict())

    df.to_csv("final_dataset.csv", index=False)
    print("\nSaved merged dataset → final_dataset.csv")

    return df

# -------------------------------------------------------
# 5. TRAIN MODEL WITH VISUALIZATION
# -------------------------------------------------------
def train_model(df):
    X = df["text"]
    y = df["lang"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )

    vectorizer = CountVectorizer(analyzer="char", ngram_range=(1, 4))
    model = MultinomialNB(alpha=0.5)
    pipeline = make_pipeline(vectorizer, model)

    print("\nTraining model...")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)

    # ----------------------------
    # 1. Accuracy & Classification Report
    # ----------------------------
    acc = accuracy_score(y_test, y_pred)
    print("\n=========== RESULTS ===========")
    print("Accuracy:", acc)
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    # ----------------------------
    # 2. Confusion Matrix
    # ----------------------------
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(xticks_rotation=45, cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

    # ----------------------------
    # 3. Accuracy per Language
    # ----------------------------
    report = classification_report(y_test, y_pred, labels=model.classes_, output_dict=True)
    per_class_acc = {lang: report[lang]['precision'] for lang in model.classes_}
    plt.figure(figsize=(6,4))
    plt.bar(per_class_acc.keys(), per_class_acc.values(), color=["#0078D7","#E81123","#2D7D9A"])
    plt.ylim(0,1)
    plt.ylabel("Precision")
    plt.title("Accuracy per Language")
    plt.tight_layout()
    plt.show()

    # ----------------------------
    # 4. Training Data Distribution
    # ----------------------------
    train_counts = y_train.value_counts()
    plt.figure(figsize=(6,4))
    plt.bar(train_counts.index, train_counts.values, color=["#0078D7","#E81123","#2D7D9A"])
    plt.title("Training Data Distribution")
    plt.ylabel("Number of Samples")
    plt.tight_layout()
    plt.show()

    # ----------------------------
    # 5. Top 10 N-grams per Language
    # ----------------------------
    vectorizer.fit(X_train)
    X_train_vec = vectorizer.transform(X_train)
    vocab = np.array(vectorizer.get_feature_names_out())
    for lang in model.classes_:
        indices = y_train[y_train == lang].index
        subset = X_train_vec[indices].toarray()
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

    # ----------------------------
    # 6. Prediction Probability Distribution
    # ----------------------------
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

# -------------------------------------------------------
# 6. MAIN SCRIPT
# -------------------------------------------------------
if __name__ == "__main__":
    df = build_dataset()
    model = train_model(df)

    # Save model
    save_dir = r"C:\Users\ALI\Desktop\old windows data 33232\Information retrival final project\Model"
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, "language_identifier_model.joblib")
    joblib.dump(model, model_path)
    print(f"\nModel saved to:\n{model_path}")
    print("\nDONE ✔")
