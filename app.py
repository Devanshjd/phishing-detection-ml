import streamlit as st
import requests

st.set_page_config(page_title="AI Threat Intelligence Platform", layout="wide")

API_BASE = st.sidebar.text_input("API Base URL", "http://127.0.0.1:8000")

st.markdown("""
<style>
.block-container { padding-top: 2rem; }
.title { font-size: 44px; font-weight: 800; }
.subtitle { color: #9BA; font-size: 16px; margin-bottom: 30px; }
.card-safe { background:#102D22; padding:18px; border-radius:12px; }
.card-warn { background:#3B2F14; padding:18px; border-radius:12px; }
.card-bad  { background:#3B1C1C; padding:18px; border-radius:12px; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🔐 AI Threat Intelligence Platform</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Machine Learning based URL & Email Phishing Detection System</div>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🌐 URL Scanner", "📧 Email Scanner", "📊 Model Info"])

# ---------------- URL ----------------
with tab1:
    st.subheader("Scan Website URL")
    url = st.text_input("Enter URL")

    if st.button("Analyze URL"):
        try:
            r = requests.post(f"{API_BASE}/predict/url", json={"url": url})
            data = r.json()

            verdict = data["verdict"]
            confidence = data["confidence"]
            legit_prob = data["legit_probability"]
            phish_prob = data["phishing_probability"]

            col1, col2 = st.columns(2)

            with col1:
                if verdict == "legitimate":
                    st.markdown(f'<div class="card-safe">✅ Legitimate<br>Confidence: {confidence:.2f}</div>', unsafe_allow_html=True)
                elif verdict == "phishing":
                    st.markdown(f'<div class="card-bad">⚠️ Phishing<br>Confidence: {confidence:.2f}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="card-warn">🟠 Suspicious<br>Confidence: {confidence:.2f}</div>', unsafe_allow_html=True)

            with col2:
                st.metric("Legitimate Probability", f"{legit_prob:.2f}")
                st.metric("Phishing Probability", f"{phish_prob:.2f}")

            st.progress(confidence)

            with st.expander("Technical Indicators"):
                st.json(data.get("indicators", {}))

        except Exception as e:
            st.error(str(e))

# ---------------- EMAIL ----------------
with tab2:
    st.subheader("Scan Email Content")
    subject = st.text_input("Email Subject")
    body = st.text_area("Email Body", height=200)

    if st.button("Analyze Email"):
        try:
            r = requests.post(f"{API_BASE}/predict/email", json={"subject": subject, "body": body})
            data = r.json()

            verdict = data["verdict"]
            confidence = data["confidence"]

            if verdict == "legitimate":
                st.success(f"Legitimate (Confidence: {confidence:.2f})")
            elif verdict == "phishing":
                st.error(f"Phishing (Confidence: {confidence:.2f})")
            else:
                st.warning(f"Suspicious (Confidence: {confidence:.2f})")

        except Exception as e:
            st.error(str(e))

# ---------------- MODEL INFO ----------------
with tab3:
    st.subheader("Model Information")
    try:
        import json
        with open("model/model_meta.json") as f:
            meta = json.load(f)

        st.json(meta)

    except:
        st.info("Model metadata not found.")