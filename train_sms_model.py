"""
Train SMS spam/phishing detection model.

Uses TF-IDF (char n-grams) + LogisticRegression on SMS text.

Dataset expected: data/raw_sms/sms_spam.csv
  columns: [label, message]
  label: "ham" = legitimate, "spam" = phishing/spam

You can use the UCI SMS Spam Collection dataset:
  https://archive.ics.uci.edu/dataset/228/sms+spam+collection
  Download, extract, and save as data/raw_sms/sms_spam.csv
"""

import json
import os
import time

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split

# ── Paths ────────────────────────────────────────────────────────────
SMS_CSV = "data/raw_sms/sms_spam.csv"
MODEL_DIR = "model"

SMS_MODEL_PATH = os.path.join(MODEL_DIR, "sms_model.joblib")
SMS_VECTORIZER_PATH = os.path.join(MODEL_DIR, "sms_vectorizer.joblib")
SMS_META_PATH = os.path.join(MODEL_DIR, "sms_meta.json")


def load_sms_data(path: str) -> pd.DataFrame:
    """
    Tries multiple CSV formats to handle the UCI SMS Spam dataset.
    The dataset is tab-separated with columns: label, message
    """
    # Try common formats with latin-1 encoding (handles special characters)
    for sep in [",", "\t"]:
        for enc in ["latin-1", "utf-8"]:
            try:
                df = pd.read_csv(path, sep=sep, encoding=enc, on_bad_lines="skip")
                if len(df.columns) >= 2:
                    cols = list(df.columns)
                    # Rename first two columns to label and message
                    df = df.rename(columns={cols[0]: "label", cols[1]: "message"})
                    # Drop any extra unnamed columns
                    df = df[["label", "message"]]
                    if len(df) > 10:  # sanity check
                        return df
            except Exception:
                continue

    raise ValueError(f"Could not parse SMS CSV at {path}")


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    # ── Load data ────────────────────────────────────────────────────
    print("Loading SMS dataset...")
    df = load_sms_data(SMS_CSV)
    df = df.dropna(subset=["label", "message"]).copy()

    # Map labels to numeric: ham=0 (legit), spam=1 (phishing)
    label_map = {"ham": 0, "spam": 1, "legitimate": 0, "phishing": 1}
    df["label_num"] = df["label"].str.strip().str.lower().map(label_map)
    df = df.dropna(subset=["label_num"])
    df["label_num"] = df["label_num"].astype(int)

    print(f"Total rows: {len(df):,}")
    print("Label distribution:\n", df["label_num"].value_counts().to_string())

    X = df["message"].astype(str)
    y = df["label_num"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── Vectorize + Train ────────────────────────────────────────────
    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        analyzer="char_wb",
        ngram_range=(3, 5),
        max_features=100_000,
        min_df=2,
    )

    clf = LogisticRegression(max_iter=300, class_weight="balanced")

    print("\nVectorizing + training SMS model...")
    t0 = time.time()

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    clf.fit(X_train_vec, y_train)

    train_time = time.time() - t0

    # ── Evaluate ─────────────────────────────────────────────────────
    preds = clf.predict(X_test_vec)
    acc = accuracy_score(y_test, preds)
    f1_spam = f1_score(y_test, preds, pos_label=1)
    f1_legit = f1_score(y_test, preds, pos_label=0)

    print(f"\nAccuracy:      {acc:.4f}")
    print(f"Spam F1 (1):   {f1_spam:.4f}")
    print(f"Legit F1 (0):  {f1_legit:.4f}")
    print(classification_report(y_test, preds, digits=4))

    # ── Save ─────────────────────────────────────────────────────────
    joblib.dump(clf, SMS_MODEL_PATH)
    joblib.dump(vectorizer, SMS_VECTORIZER_PATH)

    meta = {
        "task": "sms_spam_phishing_detection",
        "model": "LogisticRegression",
        "vectorizer": "TF-IDF (char_wb 3-5, max_features=100k)",
        "train_time_sec": round(train_time, 3),
        "test_accuracy": round(float(acc), 6),
        "f1_spam": round(float(f1_spam), 6),
        "f1_legit": round(float(f1_legit), 6),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "labels": {"0": "legitimate/ham", "1": "spam/phishing"},
    }

    with open(SMS_META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved:\n  - {SMS_MODEL_PATH}\n  - {SMS_VECTORIZER_PATH}\n  - {SMS_META_PATH}")


if __name__ == "__main__":
    main()