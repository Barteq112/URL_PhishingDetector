from __future__ import annotations

import ipaddress
import math
import re
from urllib.parse import parse_qs, urlparse

import tldextract

COMMON_TLDS = {
    "com",
    "org",
    "net",
    "edu",
    "gov",
    "pl",
    "de",
    "uk",
}

SUSPICIOUS_TLDS = {
    "zip",
    "review",
    "country",
    "kim",
    "cricket",
    "science",
    "work",
    "click",
    "gq",
    "ml",
    "cf",
    "tk",
    "ga",
}

SHORTENER_DOMAINS = {
    "bit.ly",
    "tinyurl.com",
    "t.co",
    "goo.gl",
    "ow.ly",
    "is.gd",
    "buff.ly",
    "rebrand.ly",
    "cutt.ly",
    "shorturl.at",
}

SUSPICIOUS_KEYWORDS = {
    "login",
    "signin",
    "secure",
    "verify",
    "update",
    "token",
    "account",
    "bank",
    "paypal",
    "wallet",
    "password",
    "confirm",
    "reset",
    "auth",
}

BRAND_KEYWORDS = {
    "paypal",
    "google",
    "microsoft",
    "apple",
    "amazon",
    "facebook",
    "instagram",
    "whatsapp",
    "netflix",
    "steam",
    "bank",
}

AUTH_SUBDOMAIN_KEYWORDS = {
    "auth",
    "login",
    "signin",
    "secure",
    "verify",
    "account",
    "token",
}


def normalize_url(url: str) -> str:
    value = str(url).strip()
    if not value.startswith(("http://", "https://")):
        return f"http://{value}"
    return value


def _is_ip(host: str | None) -> int:
    if not host:
        return 0
    try:
        ipaddress.ip_address(host)
        return 1
    except ValueError:
        return 0


def parse_url_features(url: str) -> dict[str, object]:
    normalized = normalize_url(url)
    parsed = urlparse(normalized)
    host = parsed.hostname or ""
    extracted = tldextract.extract(host)

    query_map = parse_qs(parsed.query, keep_blank_values=True)
    query_key_count = len(query_map)
    query_value_count = sum(len(values) for values in query_map.values())
    subdomain_count = len([part for part in extracted.subdomain.split(".") if part])

    return {
        "input_url": url,
        "normalized_url": normalized,
        "scheme": parsed.scheme,
        "domain": host,
        "path": parsed.path or "",
        "query": parsed.query or "",
        "fragment": parsed.fragment or "",
        "port": parsed.port or 0,
        "subdomain": extracted.subdomain or "",
        "registered_domain": extracted.domain or "",
        "tld": extracted.suffix or "",
        "subdomain_count": subdomain_count,
        "query_param_count": query_key_count,
        "query_value_count": query_value_count,
        "is_https": int(parsed.scheme == "https"),
        "is_domain_ip": _is_ip(host),
        "has_userinfo": int("@" in parsed.netloc),
    }


def _entropy(value: str) -> float:
    if not value:
        return 0.0
    probs = [value.count(char) / len(value) for char in set(value)]
    return -sum(prob * math.log2(prob) for prob in probs)


def _keyword_count(value: str) -> int:
    lowered = value.lower()
    return sum(1 for keyword in SUSPICIOUS_KEYWORDS if keyword in lowered)


def get_character_pattern_features(url: str, domain: str, path: str, query: str) -> dict[str, float]:
    special_count = len(re.findall(r"[^a-zA-Z0-9]", url))
    digit_count = sum(ch.isdigit() for ch in url)
    letter_count = sum(ch.isalpha() for ch in url)
    base_len = max(len(url), 1)

    return {
        "url_length": len(url),
        "length_url": len(url),
        "domain_length": len(domain),
        "path_length": len(path),
        "query_length": len(query),
        "digit_count": digit_count,
        "letter_count": letter_count,
        "special_char_count": special_count,
        "dot_count": url.count("."),
        "qty_dot_url": url.count("."),
        "hyphen_count": url.count("-"),
        "qty_hyphen_path": path.count("-"),
        "underscore_count": url.count("_"),
        "at_count": url.count("@"),
        "slash_count": url.count("/"),
        "question_mark_count": url.count("?"),
        "equal_count": url.count("="),
        "ampersand_count": url.count("&"),
        "percent_count": url.count("%"),
        "digit_ratio": digit_count / base_len,
        "special_char_ratio": special_count / base_len,
        "contains_punycode": int("xn--" in url.lower()),
        "has_long_repeated_char_sequence": int(bool(re.search(r"(.)\1{3,}", url))),
        "url_entropy": _entropy(url.lower()),
    }


def _subdomain_auth_count(subdomain: str) -> int:
    parts = [part for part in re.split(r"[^a-z0-9]+", subdomain.lower()) if part]
    return sum(1 for part in parts if part in AUTH_SUBDOMAIN_KEYWORDS)


def get_auth_and_keyword_features(
    url: str,
    domain: str,
    path: str,
    query: str,
    has_userinfo: int,
    subdomain: str,
) -> dict[str, int]:
    text = f"{domain} {path} {query} {url}".lower()
    return {
        "has_auth_section": has_userinfo,
        "suspicious_keyword_count": _keyword_count(text),
        "contains_login_keyword": int("login" in text or "signin" in text),
        "contains_secure_keyword": int("secure" in text or "verify" in text),
        "contains_token_keyword": int("token" in text or "auth" in text),
        "has_brand_keyword": int(any(keyword in text for keyword in BRAND_KEYWORDS)),
        "subdomain_auth_count": _subdomain_auth_count(subdomain),
    }


def get_tld_features(tld: str) -> dict[str, int]:
    tld_clean = tld.lower().strip(".")
    return {
        "tld_length": len(tld_clean),
        "tld_is_common": int(tld_clean in COMMON_TLDS),
        "tld_is_suspicious": int(tld_clean in SUSPICIOUS_TLDS),
    }


def get_extra_features(domain: str, subdomain_count: int, is_domain_ip: int) -> dict[str, int]:
    lowered_domain = domain.lower()
    return {
        "is_shortened_url": int(lowered_domain in SHORTENER_DOMAINS),
        "subdomain_count_feature": int(subdomain_count),
        "is_domain_ip_feature": int(is_domain_ip),
    }


def extract_features(url: str) -> dict[str, object]:
    parsed = parse_url_features(url)

    domain = str(parsed["domain"])
    path = str(parsed["path"])
    query = str(parsed["query"])
    tld = str(parsed["tld"])
    has_userinfo = int(parsed["has_userinfo"])
    subdomain_count = int(parsed["subdomain_count"])
    subdomain = str(parsed["subdomain"])
    is_domain_ip = int(parsed["is_domain_ip"])
    normalized_url = str(parsed["normalized_url"])

    features: dict[str, object] = {}
    features.update(parsed)

    features.update(
        get_character_pattern_features(
            url=normalized_url,
            domain=domain,
            path=path,
            query=query,
        )
    )
    features.update(
        get_auth_and_keyword_features(
            url=normalized_url,
            domain=domain,
            path=path,
            query=query,
            has_userinfo=has_userinfo,
            subdomain=subdomain,
        )
    )
    features.update(get_tld_features(tld))
    features.update(
        get_extra_features(
            domain=domain,
            subdomain_count=subdomain_count,
            is_domain_ip=is_domain_ip,
        )
    )

    return features
