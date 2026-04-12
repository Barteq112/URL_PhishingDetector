import re
import math
import os
import pandas as pd
from urllib.parse import urlparse
from tqdm import tqdm


# =========================
# KONFIGURACJA
# =========================
INPUT_CSV = r"C:/Users/Julian/PycharmProjects/PythonProject/URL_PhishingDetector/data/normalized/data_bal-20000.csv_normalized.csv"
OUTPUT_CSV = r"C:/Users/Julian/PycharmProjects/PythonProject/URL_PhishingDetector/data/features.csv"


# =========================
# POMOCNICZE
# =========================
def normalize_url(url: str) -> str:
    url = str(url).strip()

    if not url:
        return ""

    url = re.sub(r"\s+", "", url)

    if not re.match(r"^https?://", url, re.IGNORECASE):
        url = "http://" + url

    return url


def safe_urlparse(url: str):
    """
    Bezpieczne parsowanie URL.
    Jeśli urlparse rzuci np. ValueError: Invalid IPv6 URL,
    zwracamy obiekt zastępczy oraz flagę błędu.
    """
    try:
        return urlparse(url), 0
    except ValueError:
        return urlparse("http://invalid.local"), 1


def safe_hostname(parsed) -> str:
    try:
        return parsed.hostname or ""
    except ValueError:
        return ""


def safe_port(parsed) -> int:
    try:
        return parsed.port or 0
    except ValueError:
        return 0


def entropy(text: str) -> float:
    if not text:
        return 0.0
    probs = [text.count(c) / len(text) for c in set(text)]
    return -sum(p * math.log2(p) for p in probs if p > 0)


def find_column(df: pd.DataFrame, keywords: list[str]):
    for col in df.columns:
        col_lower = str(col).lower().strip()
        if any(k in col_lower for k in keywords):
            return col
    return None


# =========================
# EKSTRAKCJA CECH
# =========================
def extract_features(raw_url: str) -> dict:
    url = normalize_url(raw_url)
    parsed, parse_error = safe_urlparse(url)

    host = safe_hostname(parsed)
    if not host and url:
        # jeśli hostname nie dało się odczytać, traktujemy to jako błąd parsowania
        # tylko gdy URL wygląda podejrzanie pod kątem nawiasów
        if "[" in url or "]" in url:
            parse_error = 1

    path = parsed.path or ""
    query = parsed.query or ""
    fragment = parsed.fragment or ""
    port = safe_port(parsed)

    digit_count = sum(c.isdigit() for c in url)
    letter_count = sum(c.isalpha() for c in url)
    special_char_count = sum(not c.isalnum() for c in url)

    url_lower = url.lower()

    features = {
        "input_url": raw_url,
        "normalized_url": url,

        "url_parse_error": parse_error,

        "scheme": parsed.scheme if parse_error == 0 else "",
        "domain": host,
        "path": path,
        "query": query,
        "fragment": fragment,
        "port": port,

        "url_length": len(url),
        "domain_length": len(host),
        "path_length": len(path),
        "query_length": len(query),

        "digit_count": digit_count,
        "letter_count": letter_count,
        "special_char_count": special_char_count,
        "dot_count": url.count("."),
        "hyphen_count": url.count("-"),
        "underscore_count": url.count("_"),
        "at_count": url.count("@"),
        "slash_count": url.count("/"),
        "question_mark_count": url.count("?"),
        "equal_count": url.count("="),
        "ampersand_count": url.count("&"),
        "percent_count": url.count("%"),
        "open_bracket_count": url.count("["),
        "close_bracket_count": url.count("]"),

        "is_https": int(url.startswith("https://")),
        "is_domain_ip": int(bool(re.fullmatch(r"\d{1,3}(\.\d{1,3}){3}", host))),
        "contains_punycode": int("xn--" in host.lower()),

        "has_brackets": int("[" in url or "]" in url),
        "has_invalid_ipv6_pattern": int(("[" in url or "]" in url) and parse_error == 1),

        "subdomain_count": max(len(host.split(".")) - 2, 0) if host else 0,
        "tld_length": len(host.split(".")[-1]) if "." in host else 0,

        "url_entropy": entropy(url),

        "contains_login_keyword": int("login" in url_lower),
        "contains_secure_keyword": int("secure" in url_lower),
        "contains_token_keyword": int("token" in url_lower),
        "contains_bank_keyword": int("bank" in url_lower),
        "contains_verify_keyword": int("verify" in url_lower),
        "contains_update_keyword": int("update" in url_lower),
        "contains_account_keyword": int("account" in url_lower),
        "contains_signin_keyword": int("signin" in url_lower),
        "contains_password_keyword": int("password" in url_lower),
    }

    features["digit_ratio"] = digit_count / len(url) if len(url) > 0 else 0.0
    features["special_char_ratio"] = special_char_count / len(url) if len(url) > 0 else 0.0

    return features


# =========================
# GŁÓWNA FUNKCJA
# =========================
def build_features_csv(input_csv: str, output_csv: str):
    print(f"Wczytywanie: {input_csv}")
    df = pd.read_csv(input_csv)

    print("Kolumny w pliku:")
    print(list(df.columns))

    url_col = find_column(df, ["url", "link", "domain"])
    label_col = find_column(df, ["label", "status", "class", "phishing", "target"])

    if url_col is None:
        raise ValueError("Nie znalazłem kolumny z URL.")

    print(f"Użyta kolumna URL: {url_col}")
    if label_col is not None:
        print(f"Użyta kolumna label: {label_col}")
    else:
        print("Nie znalazłem kolumny label — zapiszę samo features bez label.")

    df = df[df[url_col].notna()].copy()
    df[url_col] = df[url_col].astype(str).str.strip()
    df = df[df[url_col] != ""].reset_index(drop=True)

    rows = []
    error_count = 0

    for row in tqdm(df.itertuples(index=False), total=len(df), desc="Feature extraction"):
        url_value = getattr(row, url_col)

        try:
            feats = extract_features(url_value)
            error_count += int(feats["url_parse_error"])
        except Exception as e:
            raw = str(url_value)

            feats = {
                "input_url": raw,
                "normalized_url": raw,
                "url_parse_error": 1,
                "scheme": "",
                "domain": "",
                "path": "",
                "query": "",
                "fragment": "",
                "port": 0,

                "url_length": len(raw),
                "domain_length": 0,
                "path_length": 0,
                "query_length": 0,

                "digit_count": sum(c.isdigit() for c in raw),
                "letter_count": sum(c.isalpha() for c in raw),
                "special_char_count": sum(not c.isalnum() for c in raw),
                "dot_count": raw.count("."),
                "hyphen_count": raw.count("-"),
                "underscore_count": raw.count("_"),
                "at_count": raw.count("@"),
                "slash_count": raw.count("/"),
                "question_mark_count": raw.count("?"),
                "equal_count": raw.count("="),
                "ampersand_count": raw.count("&"),
                "percent_count": raw.count("%"),
                "open_bracket_count": raw.count("["),
                "close_bracket_count": raw.count("]"),

                "is_https": 0,
                "is_domain_ip": 0,
                "contains_punycode": 0,

                "has_brackets": int("[" in raw or "]" in raw),
                "has_invalid_ipv6_pattern": int("[" in raw or "]" in raw),

                "subdomain_count": 0,
                "tld_length": 0,

                "url_entropy": entropy(raw),

                "contains_login_keyword": int("login" in raw.lower()),
                "contains_secure_keyword": int("secure" in raw.lower()),
                "contains_token_keyword": int("token" in raw.lower()),
                "contains_bank_keyword": int("bank" in raw.lower()),
                "contains_verify_keyword": int("verify" in raw.lower()),
                "contains_update_keyword": int("update" in raw.lower()),
                "contains_account_keyword": int("account" in raw.lower()),
                "contains_signin_keyword": int("signin" in raw.lower()),
                "contains_password_keyword": int("password" in raw.lower()),

                "digit_ratio": (sum(c.isdigit() for c in raw) / len(raw)) if len(raw) > 0 else 0.0,
                "special_char_ratio": (sum(not c.isalnum() for c in raw) / len(raw)) if len(raw) > 0 else 0.0,
            }

            error_count += 1
            print(f"\n[WARN] Nie udało się sparsować URL: {raw}")
            print(f"[WARN] Szczegóły: {e}")

        if label_col is not None:
            feats["label"] = getattr(row, label_col)

        rows.append(feats)

    out_df = pd.DataFrame(rows)

    if "label" in out_df.columns:
        cols = [c for c in out_df.columns if c != "label"] + ["label"]
        out_df = out_df[cols]

    output_dir = os.path.dirname(output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    out_df.to_csv(output_csv, index=False, encoding="utf-8")

    print("\nGotowe.")
    print(f"Zapisano plik: {output_csv}")
    print(f"Liczba rekordów: {len(out_df)}")
    print(f"Liczba kolumn: {len(out_df.columns)}")
    print(f"Liczba URL-i z błędem parsowania: {error_count}")
    print("\nPodgląd:")
    print(out_df.head())


if __name__ == "__main__":
    build_features_csv(INPUT_CSV, OUTPUT_CSV)