# backend/api.py
"""
FastAPI backend for AI Threat Intelligence Platform.
Endpoints: /predict/url, /predict/email, /predict/sms, /history/recent, /stats
"""

from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.db import init_db, insert_scan, get_recent, get_stats
from backend.feature_extraction import (
    FEATURE_NAMES,
    extract_features,
    features_to_vector,
    normalize_url,
)

# ── Paths ────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "model"
FRONTEND_DIR = BASE_DIR / "frontend"

# URL model (feature-based RandomForest)
URL_MODEL_PATH = MODEL_DIR / "best_model.joblib"
URL_META_PATH = MODEL_DIR / "model_meta.json"

# Email model (TF-IDF + LogReg)
EMAIL_MODEL_PATH = MODEL_DIR / "email_model.joblib"
EMAIL_VECT_PATH = MODEL_DIR / "email_vectorizer.joblib"
EMAIL_META_PATH = MODEL_DIR / "email_meta.json"

# SMS model (TF-IDF + LogReg)
SMS_MODEL_PATH = MODEL_DIR / "sms_model.joblib"
SMS_VECT_PATH = MODEL_DIR / "sms_vectorizer.joblib"
SMS_META_PATH = MODEL_DIR / "sms_meta.json"


# ── Helpers ──────────────────────────────────────────────────────────
ALLOWLIST_HOSTS = {
    "google.com", "www.google.com", "accounts.google.com",
    "paypal.com", "www.paypal.com",
    "microsoft.com", "www.microsoft.com", "login.microsoftonline.com",
    "apple.com", "www.apple.com",
    "github.com", "www.github.com",
    "amazon.com", "www.amazon.com",
    "facebook.com", "www.facebook.com",
    "twitter.com", "www.twitter.com", "x.com",
    "linkedin.com", "www.linkedin.com",
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_host(u: str) -> str:
    from urllib.parse import urlparse
    try:
        return (urlparse(u).hostname or "").lower()
    except Exception:
        return ""


def email_indicators(subject: str, body: str) -> Dict[str, Any]:
    text = f"{subject}\n{body}".strip()
    urls_found = re.findall(r"https?://\S+", text, flags=re.I)
    return {
        "contains_link": len(urls_found) > 0,
        "link_count": len(urls_found),
        "subject_length": len(subject or ""),
        "body_length": len(body or ""),
        "text_length": len(text),
        "has_urgency_words": bool(re.search(
            r"urgent|immediately|suspend|verify|expire|click here|act now",
            text, re.I
        )),
    }


def sms_indicators(text: str) -> Dict[str, Any]:
    urls_found = re.findall(r"https?://\S+", text, flags=re.I)
    return {
        "contains_link": len(urls_found) > 0,
        "link_count": len(urls_found),
        "message_length": len(text),
        "has_urgency_words": bool(re.search(
            r"urgent|immediately|suspend|verify|expire|click|act now|won|prize|free|congratulations",
            text, re.I
        )),
        "has_phone_number": bool(re.search(r"\b\d{10,}\b", text)),
    }


# ── Load models at startup ──────────────────────────────────────────
def _load_joblib(path: Path):
    return joblib.load(path) if path.exists() else None


def _load_json(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


url_model = _load_joblib(URL_MODEL_PATH)
url_meta = _load_json(URL_META_PATH)

email_model = _load_joblib(EMAIL_MODEL_PATH)
email_vectorizer = _load_joblib(EMAIL_VECT_PATH)
email_meta = _load_json(EMAIL_META_PATH)

sms_model = _load_joblib(SMS_MODEL_PATH)
sms_vectorizer = _load_joblib(SMS_VECT_PATH)
sms_meta = _load_json(SMS_META_PATH)


# ── FastAPI app ──────────────────────────────────────────────────────
app = FastAPI(title="AI Threat Intelligence Platform", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if FRONTEND_DIR.exists():
    app.mount("/frontend", StaticFiles(directory=str(FRONTEND_DIR)), name="frontend")


@app.on_event("startup")
def on_startup():
    init_db()


@app.get("/")
def root():
    return RedirectResponse(url="/frontend/index.html")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "url_model_loaded": url_model is not None,
        "email_model_loaded": email_model is not None and email_vectorizer is not None,
        "sms_model_loaded": sms_model is not None and sms_vectorizer is not None,
    }


# ── Request schemas ──────────────────────────────────────────────────
class UrlRequest(BaseModel):
    url: str = Field(..., min_length=1, max_length=10_000)


class EmailRequest(BaseModel):
    subject: str = ""
    body: str = ""


class SmsRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5_000)


# ── URL prediction (feature-based) ──────────────────────────────────
@app.post("/predict/url")
def predict_url(req: UrlRequest):
    raw = req.url.strip()
    normalized = normalize_url(raw)
    host = get_host(normalized)

    # Extract features (always — for indicator display)
    feats = extract_features(raw)

    # Allowlist fast-pass
    if host in ALLOWLIST_HOSTS:
        payload = {
            "verdict": "legitimate",
            "confidence": 0.99,
            "legit_probability": 0.99,
            "phishing_probability": 0.01,
            "normalized_url": normalized,
            "indicators": {**feats, "allowlist_match": True},
            "model": {"selected": "Allowlist bypass", "test_accuracy": url_meta.get("test_accuracy")},
        }
    elif url_model is None:
        return {"error": "URL model not loaded. Run train_model.py first."}
    else:
        # Feature-based prediction
        X = np.array([features_to_vector(feats)], dtype=np.float64)
        proba = url_model.predict_proba(X)[0]

        # Class order: [0=phishing, 1=legitimate]
        phishing_prob = float(proba[0])
        legit_prob = float(proba[1])
        confidence = max(phishing_prob, legit_prob)

        if legit_prob >= 0.7:
            verdict = "legitimate"
        elif phishing_prob >= 0.7:
            verdict = "phishing"
        else:
            verdict = "suspicious"

        payload = {
            "verdict": verdict,
            "confidence": round(confidence, 4),
            "legit_probability": round(legit_prob, 4),
            "phishing_probability": round(phishing_prob, 4),
            "normalized_url": normalized,
            "indicators": feats,
            "model": {
                "selected": url_meta.get("best_model", "RandomForest"),
                "feature_count": len(FEATURE_NAMES),
                "test_accuracy": url_meta.get("test_accuracy"),
                "roc_auc": url_meta.get("roc_auc"),
            },
        }

    insert_scan("url", normalized, payload["verdict"], payload["confidence"],
                json.dumps({"indicators": payload["indicators"], "model": payload["model"]}),
                now_iso())
    return payload


# ── Email prediction ─────────────────────────────────────────────────
@app.post("/predict/email")
def predict_email(req: EmailRequest):
    subject = (req.subject or "").strip()
    body = (req.body or "").strip()
    text = f"SUBJECT: {subject}\nBODY: {body}".strip()

    indicators = email_indicators(subject, body)

    if not (email_model and email_vectorizer):
        return {"error": "Email model not loaded. Run train_email_model.py first."}

    X = email_vectorizer.transform([text])
    proba = email_model.predict_proba(X)[0]
    # label 0=legitimate, 1=phishing
    legit_prob = float(proba[0])
    phishing_prob = float(proba[1])
    confidence = max(phishing_prob, legit_prob)

    if phishing_prob >= 0.7:
        verdict = "phishing"
    elif phishing_prob >= 0.4:
        verdict = "suspicious"
    else:
        verdict = "legitimate"

    payload = {
        "verdict": verdict,
        "confidence": round(confidence, 4),
        "legit_probability": round(legit_prob, 4),
        "phishing_probability": round(phishing_prob, 4),
        "indicators": indicators,
        "model": {
            "selected": "LogisticRegression",
            "vectorizer": "TF-IDF (char_wb 3-5)",
            "test_accuracy": email_meta.get("test_accuracy"),
        },
    }

    insert_scan("email", text[:5000], payload["verdict"], payload["confidence"],
                json.dumps({"indicators": indicators, "model": payload["model"]}),
                now_iso())
    return payload


# ── SMS prediction (NEW) ─────────────────────────────────────────────
@app.post("/predict/sms")
def predict_sms(req: SmsRequest):
    text = req.text.strip()
    indicators = sms_indicators(text)

    # Use dedicated SMS model if available, fall back to email model
    if sms_model and sms_vectorizer:
        X = sms_vectorizer.transform([text])
        proba = sms_model.predict_proba(X)[0]
        model_name = "SMS LogisticRegression"
        model_acc = sms_meta.get("test_accuracy")
    elif email_model and email_vectorizer:
        X = email_vectorizer.transform([f"SUBJECT: [SMS]\nBODY: {text}"])
        proba = email_model.predict_proba(X)[0]
        model_name = "Email model (fallback)"
        model_acc = email_meta.get("test_accuracy")
    else:
        return {"error": "No SMS or email model loaded. Train a model first."}

    # label 0=legitimate, 1=spam/phishing
    legit_prob = float(proba[0])
    phishing_prob = float(proba[1])
    confidence = max(phishing_prob, legit_prob)

    if phishing_prob >= 0.7:
        verdict = "phishing"
    elif phishing_prob >= 0.4:
        verdict = "suspicious"
    else:
        verdict = "legitimate"

    payload = {
        "verdict": verdict,
        "confidence": round(confidence, 4),
        "legit_probability": round(legit_prob, 4),
        "phishing_probability": round(phishing_prob, 4),
        "indicators": indicators,
        "model": {
            "selected": model_name,
            "test_accuracy": model_acc,
        },
    }

    insert_scan("sms", text[:5000], payload["verdict"], payload["confidence"],
                json.dumps({"indicators": indicators, "model": payload["model"]}),
                now_iso())
    return payload


# ── History + Stats ──────────────────────────────────────────────────
@app.get("/history/recent")
def history_recent(limit: int = 20):
    items = get_recent(limit=limit)
    for it in items:
        try:
            it["meta_json"] = json.loads(it["meta_json"])
        except Exception:
            pass
    return {"items": items}


@app.get("/stats")
def stats():
    return get_stats()
