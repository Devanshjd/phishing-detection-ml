"""
Feature extraction for URL phishing detection.
Extracts hand-crafted numeric features from raw URLs.
Used by both train_model.py and the FastAPI prediction endpoint.
"""

import re
import math
import urllib.parse

# ── Known URL shortener domains ──────────────────────────────────────
SHORTENERS = frozenset({
    "bit.ly", "t.co", "tinyurl.com", "goo.gl", "ow.ly",
    "is.gd", "buff.ly", "adf.ly", "rb.gy", "shorturl.at",
    "cutt.ly", "tiny.cc", "lnkd.in",
})

# ── Words commonly found in phishing URLs ────────────────────────────
SUSPICIOUS_WORDS = [
    "login", "verify", "update", "free", "secure", "account", "bank",
    "confirm", "password", "paypal", "signin", "webscr", "support",
    "reward", "suspend", "alert", "expire", "urgent", "click",
    "ebay", "apple", "microsoft", "amazon", "netflix",
]

# ── Common TLDs for one-hot encoding ─────────────────────────────────
COMMON_TLDS = [
    "com", "org", "net", "edu", "gov", "uk", "in", "ac",
    "co", "io", "ru", "cn", "xyz", "top", "info", "biz",
]

# ── Ordered feature names (must match what the model was trained on) ─
FEATURE_NAMES = [
    "url_len", "host_len", "path_len", "query_len",
    "dot_count", "hyphen_count", "at_count", "qmark_count",
    "eq_count", "slash_count", "ampersand_count",
    "digit_count", "alpha_count", "special_count",
    "digit_ratio", "alpha_ratio", "special_ratio",
    "subdomain_count", "path_depth",
    "has_https", "has_ip_in_host", "has_shortener", "has_port",
    "has_at_sign", "has_double_slash_redirect",
    "suspicious_word_hits",
    "entropy_url", "entropy_host", "entropy_path",
    "avg_word_len_host", "longest_word_host",
    "num_host_tokens", "num_path_tokens",
] + [f"tld_{t}" for t in COMMON_TLDS]


def normalize_url(url: str) -> str:
    """Ensure URL has a scheme for correct parsing."""
    url = (url or "").strip()
    if not url:
        return ""
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", url):
        url = "http://" + url
    return url


def _safe_parse(url: str):
    url = normalize_url(url)
    if not url:
        return urllib.parse.urlparse("http://empty")
    url = url.replace("[", "").replace("]", "")
    return urllib.parse.urlparse(url)


def _entropy(s: str) -> float:
    """Shannon entropy of a string."""
    if not s:
        return 0.0
    freq: dict[str, int] = {}
    for ch in s:
        freq[ch] = freq.get(ch, 0) + 1
    length = len(s)
    return -sum((c / length) * math.log2(c / length) for c in freq.values())


def extract_features(url: str) -> dict:
    """
    Extract all numeric features from a URL.
    Returns a dict whose keys match FEATURE_NAMES (order-safe).
    """
    url = normalize_url(url)
    p = _safe_parse(url)

    full = url.lower()
    host = (p.hostname or "").lower()
    path = (p.path or "").lower()
    query = (p.query or "").lower()

    parts = host.split(".") if host else []
    tld = parts[-1] if len(parts) >= 2 else ""

    # ── Lengths ──
    url_len = len(full)
    host_len = len(host)
    path_len = len(path)
    query_len = len(query)

    # ── Counts ──
    dot_count = full.count(".")
    hyphen_count = full.count("-")
    at_count = full.count("@")
    qmark_count = full.count("?")
    eq_count = full.count("=")
    slash_count = full.count("/")
    ampersand_count = full.count("&")

    digit_count = sum(ch.isdigit() for ch in full)
    alpha_count = sum(ch.isalpha() for ch in full)
    special_count = url_len - digit_count - alpha_count

    # ── Ratios ──
    digit_ratio = digit_count / url_len if url_len else 0.0
    alpha_ratio = alpha_count / url_len if url_len else 0.0
    special_ratio = special_count / url_len if url_len else 0.0

    # ── Structural ──
    subdomain_count = max(0, len(parts) - 2)
    path_segments = [seg for seg in path.split("/") if seg]
    path_depth = len(path_segments)

    # ── Boolean flags ──
    has_https = 1 if p.scheme == "https" else 0
    has_ip_in_host = 1 if re.fullmatch(r"\d{1,3}(\.\d{1,3}){3}", host) else 0
    has_shortener = 1 if host in SHORTENERS else 0
    try:
        has_port = 1 if p.port and p.port not in (80, 443) else 0
    except (ValueError, TypeError):
        has_port = 1  # malformed port = suspicious
    has_at_sign = 1 if "@" in full else 0
    has_double_slash_redirect = 1 if "//" in path else 0

    # ── Suspicious word hits ──
    suspicious_word_hits = sum(1 for w in SUSPICIOUS_WORDS if w in full)

    # ── Entropy ──
    entropy_url = _entropy(full)
    entropy_host = _entropy(host)
    entropy_path = _entropy(path)

    # ── Host token stats ──
    host_words = [w for w in re.split(r"[.\-]", host) if w]
    avg_word_len_host = (sum(len(w) for w in host_words) / len(host_words)) if host_words else 0.0
    longest_word_host = max((len(w) for w in host_words), default=0)
    num_host_tokens = len(host_words)
    num_path_tokens = len(path_segments)

    feats = {
        "url_len": url_len,
        "host_len": host_len,
        "path_len": path_len,
        "query_len": query_len,
        "dot_count": dot_count,
        "hyphen_count": hyphen_count,
        "at_count": at_count,
        "qmark_count": qmark_count,
        "eq_count": eq_count,
        "slash_count": slash_count,
        "ampersand_count": ampersand_count,
        "digit_count": digit_count,
        "alpha_count": alpha_count,
        "special_count": special_count,
        "digit_ratio": round(digit_ratio, 6),
        "alpha_ratio": round(alpha_ratio, 6),
        "special_ratio": round(special_ratio, 6),
        "subdomain_count": subdomain_count,
        "path_depth": path_depth,
        "has_https": has_https,
        "has_ip_in_host": has_ip_in_host,
        "has_shortener": has_shortener,
        "has_port": has_port,
        "has_at_sign": has_at_sign,
        "has_double_slash_redirect": has_double_slash_redirect,
        "suspicious_word_hits": suspicious_word_hits,
        "entropy_url": round(entropy_url, 6),
        "entropy_host": round(entropy_host, 6),
        "entropy_path": round(entropy_path, 6),
        "avg_word_len_host": round(avg_word_len_host, 4),
        "longest_word_host": longest_word_host,
        "num_host_tokens": num_host_tokens,
        "num_path_tokens": num_path_tokens,
    }

    # TLD one-hot
    for t in COMMON_TLDS:
        feats[f"tld_{t}"] = 1 if tld == t else 0

    return feats


def features_to_vector(feats: dict) -> list:
    """Convert feature dict to ordered list matching FEATURE_NAMES."""
    return [feats.get(name, 0) for name in FEATURE_NAMES]