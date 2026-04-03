"""
Train URL phishing detection model.

Uses hand-crafted features from feature_extraction.py + RandomForest
(matching the best model from classification_report.txt: 93.3% accuracy, 0.981 AUC).

Dataset expected: data/urls.csv with columns [url, status]
  status: 0 = phishing, 1 = legitimate
"""

import json
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

# Add parent dir so we can import backend.feature_extraction
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backend.feature_extraction import FEATURE_NAMES, extract_features

# ── Paths ────────────────────────────────────────────────────────────
DATA_PATH = "data/urls.csv"
MODEL_DIR = "model"
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.joblib")
META_PATH = os.path.join(MODEL_DIR, "model_meta.json")

SAMPLE_SIZE = 200_000  # set to None to use full dataset


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    # ── Load data ────────────────────────────────────────────────────
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    df["url"] = df["url"].astype(str).fillna("")
    df["status"] = df["status"].astype(int)

    print(f"Total rows: {len(df):,}")
    print("Label distribution:\n", df["status"].value_counts().to_string())

    # ── Optional sampling ────────────────────────────────────────────
    if SAMPLE_SIZE and len(df) > SAMPLE_SIZE:
        frac = SAMPLE_SIZE / len(df)
        df = (
            df.groupby("status", group_keys=False)
            .apply(lambda x: x.sample(frac=frac, random_state=42))
            .reset_index(drop=True)
        )
        print(f"Sampled to {len(df):,} rows")

    # ── Extract features ─────────────────────────────────────────────
    print("\nExtracting features from URLs...")
    t0 = time.time()
    feature_rows = []
    for url in df["url"]:
        feats = extract_features(url)
        feature_rows.append([feats.get(name, 0) for name in FEATURE_NAMES])

    X = np.array(feature_rows, dtype=np.float64)
    y = df["status"].values  # 0=phishing, 1=legitimate
    print(f"Feature extraction done in {time.time() - t0:.1f}s  ({X.shape[1]} features)")

    # ── Train/test split ─────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── Train models ─────────────────────────────────────────────────
    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=25,
            min_samples_split=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="liblinear",
        ),
    }

    best_name = None
    best_model = None
    best_f1 = -1
    results = {}

    for name, clf in models.items():
        print(f"\n{'='*50}")
        print(f"Training {name}...")
        t1 = time.time()
        clf.fit(X_train, y_train)
        train_time = time.time() - t1

        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1_phish = f1_score(y_test, y_pred, pos_label=0)
        f1_legit = f1_score(y_test, y_pred, pos_label=1)

        try:
            y_proba = clf.predict_proba(X_test)[:, 1]
            roc = roc_auc_score(y_test, y_proba)
        except Exception:
            roc = None

        print(f"Accuracy: {acc:.4f}")
        print(f"Phishing F1 (0): {f1_phish:.4f}")
        print(f"Legit F1 (1):    {f1_legit:.4f}")
        if roc:
            print(f"ROC AUC:         {roc:.4f}")
        print(f"Train time:      {train_time:.2f}s")
        print(classification_report(y_test, y_pred, digits=4))

        results[name] = {
            "accuracy": float(acc),
            "f1_phishing": float(f1_phish),
            "f1_legit": float(f1_legit),
            "roc_auc": float(roc) if roc else None,
            "train_time_sec": round(train_time, 2),
        }

        if f1_phish > best_f1:
            best_f1 = f1_phish
            best_name = name
            best_model = clf

    # ── Save best model ──────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"Best model: {best_name} (Phishing F1 = {best_f1:.4f})")

    joblib.dump(best_model, BEST_MODEL_PATH)

    meta = {
        "best_model": best_name,
        "feature_names": FEATURE_NAMES,
        "feature_count": len(FEATURE_NAMES),
        "labels": {"0": "phishing", "1": "legitimate"},
        "sample_size": int(len(df)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "results": results,
        "test_accuracy": results[best_name]["accuracy"],
        "roc_auc": results[best_name].get("roc_auc"),
    }

    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved:\n  - {BEST_MODEL_PATH}\n  - {META_PATH}")


if __name__ == "__main__":
    main()
