"""
Streamlit alternative frontend for the AI Threat Intelligence Platform.
Run: streamlit run app.py
(Make sure the FastAPI backend is running at the API_BASE URL)
"""

import streamlit as st
import requests
import json

st.set_page_config(page_title="AI Threat Intelligence Platform", layout="wide")

API_BASE = st.sidebar.text_input("API Base URL", "http://127.0.0.1:8000")

st.markdown("""
<style>
.block-container { padding-top: 2rem; }
.card-safe { background:#102D22; padding:18px; border-radius:12px; color:#2cffb0; }
.card-warn { background:#3B2F14; padding:18px; border-radius:12px; color:#ffb84d; }
.card-bad  { background:#3B1C1C; padding:18px; border-radius:12px; color:#ff4d6d; }
</style>
""", unsafe_allow_html=True)

st.title("AI Threat Intelligence Platform")
st.caption("ML-based URL, Email & SMS Phishing Detection")

tab1, tab2, tab3, tab4 = st.tabs(["URL Scanner", "Email Scanner", "SMS Scanner", "Model Info"])


def show_result(data):
    """Display prediction result consistently."""
    verdict = data.get("verdict", "unknown")
    confidence = data.get("confidence", 0)
    legit_prob = data.get("legit_probability", 0)
    phish_prob = data.get("phishing_probability", 0)

    col1, col2 = st.columns(2)
    with col1:
        if verdict == "legitimate":
            st.markdown(f'<div class="card-safe">LEGITIMATE — Confidence: {confidence:.1%}</div>',
                        unsafe_allow_html=True)
        elif verdict == "phishing":
            st.markdown(f'<div class="card-bad">PHISHING — Confidence: {confidence:.1%}</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="card-warn">SUSPICIOUS — Confidence: {confidence:.1%}</div>',
                        unsafe_allow_html=True)

    with col2:
        st.metric("Legitimate", f"{legit_prob:.1%}")
        st.metric("Phishing", f"{phish_prob:.1%}")

    st.progress(confidence)

    with st.expander("Technical details"):
        st.json(data)


# ── URL Scanner ──────────────────────────────────────────────────────
with tab1:
    st.subheader("Scan a URL")
    url = st.text_input("Enter URL", placeholder="e.g. https://example.com")
    if st.button("Analyze URL"):
        if not url.strip():
            st.warning("Please enter a URL.")
        else:
            try:
                r = requests.post(f"{API_BASE}/predict/url", json={"url": url}, timeout=15)
                data = r.json()
                if "error" in data:
                    st.error(data["error"])
                else:
                    show_result(data)
            except Exception as e:
                st.error(f"Connection failed: {e}")

# ── Email Scanner ────────────────────────────────────────────────────
with tab2:
    st.subheader("Scan Email Content")
    subject = st.text_input("Email Subject", placeholder="e.g. Urgent: Verify your account")
    body = st.text_area("Email Body", height=200, placeholder="Paste email content here...")
    if st.button("Analyze Email"):
        if not subject.strip() and not body.strip():
            st.warning("Please enter email subject or body.")
        else:
            try:
                r = requests.post(f"{API_BASE}/predict/email",
                                  json={"subject": subject, "body": body}, timeout=15)
                data = r.json()
                if "error" in data:
                    st.error(data["error"])
                else:
                    show_result(data)
            except Exception as e:
                st.error(f"Connection failed: {e}")

# ── SMS Scanner ──────────────────────────────────────────────────────
with tab3:
    st.subheader("Scan SMS Message")
    sms_text = st.text_area("SMS Message", height=150,
                            placeholder="Paste suspicious SMS message here...")
    if st.button("Analyze SMS"):
        if not sms_text.strip():
            st.warning("Please enter an SMS message.")
        else:
            try:
                r = requests.post(f"{API_BASE}/predict/sms",
                                  json={"text": sms_text}, timeout=15)
                data = r.json()
                if "error" in data:
                    st.error(data["error"])
                else:
                    show_result(data)
            except Exception as e:
                st.error(f"Connection failed: {e}")

# ── Model Info ───────────────────────────────────────────────────────
with tab4:
    st.subheader("Model Information")
    try:
        r = requests.get(f"{API_BASE}/health", timeout=5)
        st.json(r.json())
    except Exception:
        st.info("Could not connect to backend for health check.")

    for meta_file, label in [
        ("model/model_meta.json", "URL Model"),
        ("model/email_meta.json", "Email Model"),
        ("model/sms_meta.json", "SMS Model"),
    ]:
        try:
            with open(meta_file) as f:
                meta = json.load(f)
            st.subheader(label)
            st.json(meta)
        except FileNotFoundError:
            st.info(f"{label} metadata not found.")
