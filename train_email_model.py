import os
import json
import time
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score


EMAIL_CSV = "data/raw_emails/phishing_email.csv"
MODEL_DIR = "model"

EMAIL_MODEL_PATH = os.path.join(MODEL_DIR, "email_model.joblib")
EMAIL_VECTORIZER_PATH = os.path.join(MODEL_DIR, "email_vectorizer.joblib")
EMAIL_META_PATH = os.path.join(MODEL_DIR, "email_meta.json")


def main():

    os.makedirs(MODEL_DIR, exist_ok=True)

    df = pd.read_csv(EMAIL_CSV)
    df = df.dropna(subset=["text_combined", "label"]).copy()

    df["label"] = df["label"].astype(int)

    print("Email label counts:\n", df["label"].value_counts())

    X = df["text_combined"].astype(str)
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        analyzer="char_wb",
        ngram_range=(3, 5),
        max_features=250000,
        min_df=3
    )

    clf = LogisticRegression(
        max_iter=300,
        class_weight="balanced"
    )

    print("\nVectorizing + Training Email Model...")
    t0 = time.time()

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    clf.fit(X_train_vec, y_train)

    train_time = time.time() - t0

    preds = clf.predict(X_test_vec)

    acc = accuracy_score(y_test, preds)
    f1_phish = f1_score(y_test, preds, pos_label=1)
    f1_legit = f1_score(y_test, preds, pos_label=0)

    print("\n✅ Email training finished")
    print(f"Accuracy: {acc:.4f}")
    print(f"Phishing F1 (1): {f1_phish:.4f}")
    print(f"Legit F1 (0): {f1_legit:.4f}\n")

    print("Report:\n", classification_report(y_test, preds, digits=4))

    joblib.dump(clf, EMAIL_MODEL_PATH)
    joblib.dump(vectorizer, EMAIL_VECTORIZER_PATH)

    meta = {
        "task": "email_phishing_detection",
        "model": "LogisticRegression",
        "vectorizer": "TFIDF(char_wb 3-5)",
        "train_time_sec": round(train_time, 3),
        "test_accuracy": round(float(acc), 6),
        "f1_phishing": round(float(f1_phish), 6),
        "f1_legit": round(float(f1_legit), 6),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test))
    }

    with open(EMAIL_META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("\nSaved:")
    print(f"- {EMAIL_MODEL_PATH}")
    print(f"- {EMAIL_VECTORIZER_PATH}")
    print(f"- {EMAIL_META_PATH}")


if __name__ == "__main__":
    main()