from __future__ import annotations

from url_pipeline.dns_features import get_dns_features
from url_pipeline.domain_features import get_domain_details
from url_pipeline.lexical_features import (
    get_auth_and_keyword_features,
    get_character_pattern_features,
    get_extra_features,
    get_tld_features,
)
from url_pipeline.network_features import get_ip_resolution_features, get_redirect_features
from url_pipeline.parsing import parse_url_features


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
