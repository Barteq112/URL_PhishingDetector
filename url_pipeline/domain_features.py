from __future__ import annotations

from datetime import datetime, timezone

import whois
from whois.parser import PywhoisError


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
    except (PywhoisError, TimeoutError, ConnectionError, OSError, ValueError):
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
