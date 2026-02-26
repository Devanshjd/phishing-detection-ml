import re
import math
import urllib.parse
import re
import urllib.parse
import math

def normalize_url(url: str) -> str:
    url = (url or "").strip()
    if not url:
        return ""
    # If user enters "google.com" add scheme
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", url):
        url = "http://" + url
    return url

SHORTENERS = {
    "bit.ly", "t.co", "tinyurl.com", "goo.gl", "ow.ly", "is.gd", "buff.ly", "adf.ly"
}

SUSPICIOUS_WORDS = [
    "login", "verify", "update", "free", "secure", "account", "bank",
    "confirm", "password", "paypal", "signin", "webscr", "support", "reward"
]

COMMON_TLDS = [
    "com", "org", "net", "edu", "gov", "uk", "in", "ac", "co", "io", "ru", "cn", "xyz", "top"
]


def _safe_parse(url: str):
    url = (url or "").strip()
    if not url:
        return urllib.parse.urlparse("http://")
    if not url.startswith(("http://", "https://")):
        url = "http://" + url
    # Fix invalid IPv6 bracket issues by stripping bad brackets
    url = url.replace("[", "").replace("]", "")
    return urllib.parse.urlparse(url)


def _entropy(s: str) -> float:
    if not s:
        return 0.0
    probs = {}
    for ch in s:
        probs[ch] = probs.get(ch, 0) + 1
    ent = 0.0
    length = len(s)
    for c, cnt in probs.items():
        p = cnt / length
        ent -= p * math.log2(p)
    return ent


def extract_features(url: str) -> dict:
    url = normalize_url(url)
    p = _safe_parse(url)
    full = url.strip()
    host = (p.hostname or "").lower()
    path = (p.path or "").lower()
    query = (p.query or "").lower()

    # Basic parts
    netloc = (p.netloc or "").lower()

    # Parse tld (simple)
    parts = host.split(".")
    tld = parts[-1] if len(parts) >= 2 else ""

    # Counts
    url_len = len(full)
    host_len = len(host)
    path_len = len(path)

    dot_count = full.count(".")
    hyphen_count = full.count("-")
    at_count = full.count("@")
    qmark_count = full.count("?")
    eq_count = full.count("=")
    slash_count = full.count("/")
    digit_count = sum(ch.isdigit() for ch in full)
    alpha_count = sum(ch.isalpha() for ch in full)
    special_count = url_len - digit_count - alpha_count

    digit_ratio = (digit_count / url_len) if url_len else 0.0
    special_ratio = (special_count / url_len) if url_len else 0.0

    # Subdomain depth
    subdomain_count = max(0, len(parts) - 2)  # a.b.c.com -> subdomain_count=2

    # IP in host?
    has_ip_in_host = 1 if re.fullmatch(r"\d{1,3}(\.\d{1,3}){3}", host or "") else 0

    # Shortener?
    has_shortener = 1 if host in SHORTENERS else 0

    # Suspicious word hits
    suspicious_hits = sum(1 for w in SUSPICIOUS_WORDS if w in full.lower())

    # HTTPS?
    has_https = 1 if p.scheme == "https" else 0

    feats = {
        # Lengths
        "url_len": url_len,
        "host_len": host_len,
        "path_len": path_len,

        # Counts
        "dot_count": dot_count,
        "hyphen_count": hyphen_count,
        "at_count": at_count,
        "qmark_count": qmark_count,
        "eq_count": eq_count,
        "slash_count": slash_count,

        # Ratios
        "digit_ratio": digit_ratio,
        "special_ratio": special_ratio,
        "subdomain_count": subdomain_count,

        # Flags
        "has_https": has_https,
        "has_ip_in_host": has_ip_in_host,
        "has_shortener": has_shortener,
        "suspicious_word_hits": suspicious_hits,

        # Entropy
        "entropy_url": _entropy(full.lower()),
        "entropy_host": _entropy(host.lower()),
    }

    # TLD one-hot
    for t in COMMON_TLDS:
        feats[f"tld_{t}"] = 1 if tld == t else 0

    return feats