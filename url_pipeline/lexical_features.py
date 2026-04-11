from __future__ import annotations

import math
import re

from url_pipeline.config import COMMON_TLDS, SHORTENER_DOMAINS, SUSPICIOUS_KEYWORDS, SUSPICIOUS_TLDS


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
        "domain_length": len(domain),
        "path_length": len(path),
        "query_length": len(query),
        "digit_count": digit_count,
        "letter_count": letter_count,
        "special_char_count": special_count,
        "dot_count": url.count("."),
        "hyphen_count": url.count("-"),
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


def get_auth_and_keyword_features(url: str, domain: str, path: str, query: str, has_userinfo: int) -> dict[str, int]:
    text = f"{domain} {path} {query} {url}".lower()
    return {
        "has_auth_section": has_userinfo,
        "suspicious_keyword_count": _keyword_count(text),
        "contains_login_keyword": int("login" in text or "signin" in text),
        "contains_secure_keyword": int("secure" in text or "verify" in text),
        "contains_token_keyword": int("token" in text or "auth" in text),
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
