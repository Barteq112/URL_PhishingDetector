from __future__ import annotations

import ipaddress
from urllib.parse import parse_qs, urlparse

import tldextract


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
