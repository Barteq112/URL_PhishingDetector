from __future__ import annotations

import ipaddress
import math
import re
import socket
from datetime import datetime, timezone
from urllib.parse import parse_qs, urlparse

import dns.exception
import dns.resolver
import requests
import tldextract
import whois
from whois.exceptions import WhoisError

HTTP_TIMEOUT_SECONDS = 8
DNS_TIMEOUT_SECONDS = 3

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


def get_redirect_features(url: str) -> dict[str, int]:
    try:
        response = requests.get(
            url,
            timeout=HTTP_TIMEOUT_SECONDS,
            allow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 URLFeatureExtractor/1.0"},
        )
    except requests.exceptions.RequestException:
        return {
            "http_request_ok": 0,
            "http_status_code": 0,
            "redirect_count": -1,
            "redirect_to_other_domain": -1,
            "final_url_is_https": -1,
        }

    source_domain = urlparse(url).hostname or ""
    final_domain = urlparse(response.url).hostname or ""
    redirect_to_other = int(final_domain and source_domain and final_domain != source_domain)

    return {
        "http_request_ok": 1,
        "http_status_code": int(response.status_code),
        "redirect_count": len(response.history),
        "redirect_to_other_domain": redirect_to_other,
        "final_url_is_https": int(urlparse(response.url).scheme == "https"),
    }


def get_ip_resolution_features(domain: str) -> dict[str, int]:
    if not domain:
        return {"resolved_ip_count": 0, "has_multiple_ips": 0}

    try:
        addresses = socket.getaddrinfo(domain, None)
    except (socket.gaierror, OSError):
        return {"resolved_ip_count": 0, "has_multiple_ips": 0}

    ips = {row[4][0] for row in addresses if row and row[4]}
    ip_count = len(ips)
    return {"resolved_ip_count": ip_count, "has_multiple_ips": int(ip_count > 1)}


def _pick_first_date(value):
    if isinstance(value, list):
        return value[0] if value else None
    return value


def _days_between(earlier: datetime | None, later: datetime | None) -> int:
    if not earlier or not later:
        return -1
    return max((later - earlier).days, -1)


def get_domain_details(domain: str) -> dict[str, int]:
    if not domain:
        return {
            "whois_available": 0,
            "domain_age_days": -1,
            "domain_validity_days": -1,
        }

    try:
        record = whois.whois(domain)
    except (WhoisError, TimeoutError, ConnectionError, OSError, ValueError):
        return {
            "whois_available": 0,
            "domain_age_days": -1,
            "domain_validity_days": -1,
        }

    created_at = _pick_first_date(record.creation_date)
    expires_at = _pick_first_date(record.expiration_date)
    now = datetime.now(timezone.utc)

    if created_at and created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)
    if expires_at and expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)

    return {
        "whois_available": 1,
        "domain_age_days": _days_between(created_at, now),
        "domain_validity_days": _days_between(now, expires_at),
    }


def _resolver() -> dns.resolver.Resolver:
    resolver = dns.resolver.Resolver()
    resolver.timeout = DNS_TIMEOUT_SECONDS
    resolver.lifetime = DNS_TIMEOUT_SECONDS
    return resolver


def _safe_count(domain: str, rtype: str) -> int:
    resolver = _resolver()
    try:
        answers = resolver.resolve(domain, rtype)
        return len(answers)
    except (
        dns.resolver.NXDOMAIN,
        dns.resolver.NoAnswer,
        dns.resolver.NoNameservers,
        dns.resolver.LifetimeTimeout,
        dns.exception.DNSException,
    ):
        return 0


def _has_spf(domain: str) -> int:
    resolver = _resolver()
    try:
        answers = resolver.resolve(domain, "TXT")
    except (
        dns.resolver.NXDOMAIN,
        dns.resolver.NoAnswer,
        dns.resolver.NoNameservers,
        dns.resolver.LifetimeTimeout,
        dns.exception.DNSException,
    ):
        return 0

    for answer in answers:
        txt = answer.to_text().lower()
        if "v=spf1" in txt:
            return 1
    return 0


def get_dns_features(domain: str) -> dict[str, int]:
    if not domain:
        return {
            "dns_mx_count": 0,
            "dns_ns_count": 0,
            "dns_txt_count": 0,
            "dns_has_spf": 0,
        }

    return {
        "dns_mx_count": _safe_count(domain, "MX"),
        "dns_ns_count": _safe_count(domain, "NS"),
        "dns_txt_count": _safe_count(domain, "TXT"),
        "dns_has_spf": _has_spf(domain),
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


def extract_features(url: str, enable_network: bool = True) -> dict[str, object]:
    parsed = parse_url_features(url)

    domain = str(parsed["domain"])
    path = str(parsed["path"])
    query = str(parsed["query"])
    tld = str(parsed["tld"])
    has_userinfo = int(parsed["has_userinfo"])
    subdomain_count = int(parsed["subdomain_count"])
    is_domain_ip = int(parsed["is_domain_ip"])
    normalized_url = str(parsed["normalized_url"])

    features: dict[str, object] = {}
    features.update(parsed)

    if enable_network:
        features.update(get_domain_details(domain))
        features.update(get_dns_features(domain))
        features.update(get_redirect_features(normalized_url))
        features.update(get_ip_resolution_features(domain))
    else:
        features.update({"whois_available": -1, "domain_age_days": -1, "domain_validity_days": -1})
        features.update({"dns_mx_count": -1, "dns_ns_count": -1, "dns_txt_count": -1, "dns_has_spf": -1})
        features.update(
            {
                "http_request_ok": -1,
                "http_status_code": -1,
                "redirect_count": -1,
                "redirect_to_other_domain": -1,
                "final_url_is_https": -1,
            }
        )
        features.update({"resolved_ip_count": -1, "has_multiple_ips": -1})

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
