from __future__ import annotations

import socket
from urllib.parse import urlparse

import requests

from url_pipeline.config import HTTP_TIMEOUT_SECONDS


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
