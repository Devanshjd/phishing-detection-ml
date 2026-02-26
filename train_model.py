import json
import time
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score

DATA_PATH = "data/urls.csv"
MODEL_PATH = "model/best_model.joblib"
VEC_PATH = "model/vectorizer.joblib"
META_PATH = "model/model_meta.json"

SAMPLE_SIZE = 200000  # keep fast

def normalize_url(u: str) -> str:
    u = (u or "").strip()
    if not u:
        return ""
    # if no scheme, add http://
    if "://" not in u:
        u = "http://" + u
    return u

def main():
    df = pd.read_csv(DATA_PATH)
    df["url"] = df["url"].astype(str).fillna("")
    df["status"] = df["status"].astype(int)

    print("Status value counts:\n", df["status"].value_counts())

    # sample (keeps class balance)
    if len(df) > SAMPLE_SIZE:
        frac = SAMPLE_SIZE / len(df)
        df = (
            df.groupby("status", group_keys=False)
              .apply(lambda x: x.sample(frac=frac, random_state=42))
              .reset_index(drop=True)
        )
        print(f"Using sample of {len(df):,} rows")

    # normalize urls
    df["url_norm"] = df["url"].apply(normalize_url)

    # IMPORTANT:
    y = df["status"].values   # 0=phishing, 1=legitimate

    X_text = df["url_norm"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X_text, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\nVectorizing (char n-grams TF-IDF)...")
    t0 = time.time()
    vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_features=200000
    )
    X_train_vec = vec.fit_transform(X_train)
    X_test_vec = vec.transform(X_test)
    print(f"Vectorization done in {time.time()-t0:.2f}s")

    print("\nTraining LogisticRegression...")
    t1 = time.time()
    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="liblinear"
    )
    model.fit(X_train_vec, y_train)
    print(f"Training done in {time.time()-t1:.2f}s")

    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    f1_phish = f1_score(y_test, y_pred, pos_label=0)
    f1_legit = f1_score(y_test, y_pred, pos_label=1)

    print("\n✅ Training finished")
    print("Accuracy:", round(acc, 4))
    print("Phishing F1 (0):", round(f1_phish, 4))
    print("Legit F1 (1):", round(f1_legit, 4))
    print("\nReport:\n", classification_report(y_test, y_pred))

    joblib.dump(model, MODEL_PATH)
    joblib.dump(vec, VEC_PATH)

    meta = {
        "selected_model": "LogisticRegression + TFIDF(char_wb 3-5)",
        "test_accuracy": float(acc),
        "f1_phishing_0": float(f1_phish),
        "f1_legit_1": float(f1_legit),
        "labels": {"0": "phishing", "1": "legitimate"},
        "sample_size": int(len(df)),
        "vectorizer": "TfidfVectorizer char_wb (3,5)"
    }

    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    print("\nSaved:")
    print("-", MODEL_PATH)
    print("-", VEC_PATH)
    print("-", META_PATH)

if __name__ == "__main__":
    main()