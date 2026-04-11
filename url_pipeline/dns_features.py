from __future__ import annotations

import dns.resolver

from url_pipeline.config import DNS_TIMEOUT_SECONDS


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
