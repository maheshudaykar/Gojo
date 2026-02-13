from __future__ import annotations

import json
import os
import socket
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None  # type: ignore[assignment]

try:
    import dns.resolver
except ImportError:  # pragma: no cover
    dns = None  # type: ignore[assignment]

from phish_detector.parsing import get_registrable_domain

RDAP_DOMAIN_URL = "https://rdap.org/domain/{domain}"
RDAP_IP_URL = "https://rdap.org/ip/{ip}"
DEFAULT_TIMEOUT = 2.0
DNS_RESOLVERS = ["1.1.1.1", "8.8.8.8", "9.9.9.9"]


@dataclass(frozen=True)
class DomainEnrichment:
    registrable_domain: str
    age_days: int | None
    age_trust: float
    asn: str | None
    asn_org: str | None
    reputation_trust: float
    reputation_reasons: list[str]
    volatility_score: float
    ip_addresses: list[str]


@lru_cache(maxsize=1)
def _load_list(path: str) -> set[str]:
    root = Path(__file__).resolve().parents[1]
    file_path = root / "configs" / path
    entries: set[str] = set()
    if file_path.exists():
        for line in file_path.read_text(encoding="utf-8").splitlines():
            entry = line.strip().lower()
            if entry and not entry.startswith("#"):
                entries.add(entry)
    return entries


def _rdap_domain(domain: str) -> dict[str, Any] | None:
    if requests is None:
        return None
    try:
        response = requests.get(RDAP_DOMAIN_URL.format(domain=domain), timeout=DEFAULT_TIMEOUT)
        if response.status_code != 200:
            return None
        return json.loads(response.text)
    except (requests.RequestException, json.JSONDecodeError):
        return None


def _rdap_ip(ip: str) -> dict[str, Any] | None:
    if requests is None:
        return None
    try:
        response = requests.get(RDAP_IP_URL.format(ip=ip), timeout=DEFAULT_TIMEOUT)
        if response.status_code != 200:
            return None
        return json.loads(response.text)
    except (requests.RequestException, json.JSONDecodeError):
        return None


def _extract_registration_date(data: dict[str, Any] | None) -> datetime | None:
    if not data:
        return None
    events = data.get("events", [])
    for event in events:
        if event.get("eventAction") == "registration" and event.get("eventDate"):
            try:
                return datetime.fromisoformat(event["eventDate"].replace("Z", "+00:00"))
            except ValueError:
                continue
    return None


def _resolve_ips(domain: str) -> list[str]:
    try:
        return [socket.gethostbyname(domain)]
    except socket.gaierror:
        return []


def _dns_volatility(domain: str) -> tuple[float, list[str]]:
    if dns is None:
        return 0.0, []
    unique_ips: set[str] = set()
    for resolver_ip in DNS_RESOLVERS:
        resolver = dns.resolver.Resolver(configure=False)
        resolver.nameservers = [resolver_ip]
        resolver.lifetime = DEFAULT_TIMEOUT
        try:
            answers = resolver.resolve(domain, "A")
        except (
            dns.resolver.NXDOMAIN,
            dns.resolver.NoAnswer,
            dns.resolver.NoNameservers,
            dns.exception.Timeout,
            dns.exception.DNSException,
        ):
            continue
        for answer in answers:
            unique_ips.add(str(answer))
    count = len(unique_ips)
    if count >= 4:
        score = 1.0
    elif count == 3:
        score = 0.7
    elif count == 2:
        score = 0.4
    else:
        score = 0.0
    return score, sorted(unique_ips)


def _age_trust_score(age_days: int | None) -> float:
    if age_days is None:
        return 0.5
    if age_days <= 30:
        return 0.0
    if age_days >= 730:
        return 1.0
    return (age_days - 30) / (730 - 30)


def _google_safe_browsing(url: str) -> bool | None:
    if requests is None:
        return None
    api_key = os.getenv("GOJO_GSB_API_KEY")
    if not api_key:
        return None
    endpoint = f"https://safebrowsing.googleapis.com/v4/threatMatches:find?key={api_key}"
    payload: dict[str, Any] = {
        "client": {"clientId": "gojo", "clientVersion": "2.0.0"},
        "threatInfo": {
            "threatTypes": ["MALWARE", "SOCIAL_ENGINEERING", "UNWANTED_SOFTWARE"],
            "platformTypes": ["ANY_PLATFORM"],
            "threatEntryTypes": ["URL"],
            "threatEntries": [{"url": url}],
        },
    }
    try:
        response = requests.post(endpoint, json=payload, timeout=DEFAULT_TIMEOUT)
        if response.status_code != 200:
            return None
        data = response.json()
        return bool(data.get("matches"))
    except requests.RequestException:
        return None


def get_domain_enrichment(
    host: str,
    url: str,
    enable_age: bool = True,
    enable_reputation: bool = True,
    enable_volatility: bool = True,
) -> DomainEnrichment:
    if os.getenv("GOJO_DISABLE_ENRICHMENT"):
        return DomainEnrichment(
            registrable_domain=host,
            age_days=None,
            age_trust=0.5,
            asn=None,
            asn_org=None,
            reputation_trust=0.5,
            reputation_reasons=["enrichment_disabled"],
            volatility_score=0.0,
            ip_addresses=[],
        )

    registrable = get_registrable_domain(host)
    allowlist = _load_list("allowlist_domains.txt")
    blocklist = _load_list("blocklist_domains.txt")
    asn_allowlist = _load_list("asn_allowlist.txt")
    asn_denylist = _load_list("asn_denylist.txt")

    reasons: list[str] = []
    reputation = 0.5

    if registrable in allowlist:
        reputation = 1.0
        reasons.append("allowlist")
    if registrable in blocklist:
        reputation = 0.0
        reasons.append("blocklist")

    if enable_reputation:
        gsb_match = _google_safe_browsing(url)
        if gsb_match is True:
            reputation = 0.0
            reasons.append("gsb_match")
        elif gsb_match is False:
            reasons.append("gsb_clear")

    age_days = None
    if enable_age:
        rdap_data = _rdap_domain(registrable)
        registration = _extract_registration_date(rdap_data)
        if registration:
            age_days = (datetime.now(timezone.utc) - registration).days

    ips = _resolve_ips(registrable)
    asn = None
    asn_org = None
    if ips and enable_reputation:
        ip_data = _rdap_ip(ips[0])
        if ip_data:
            asn = str(ip_data.get("asn")) if ip_data.get("asn") is not None else None
            asn_org = ip_data.get("name")
            if asn and asn in asn_allowlist and reputation < 0.9:
                reputation = 0.9
                reasons.append("asn_allowlist")
            if asn and asn in asn_denylist:
                reputation = 0.1
                reasons.append("asn_denylist")

    volatility_score = 0.0
    dns_ips: list[str] = []
    if enable_volatility:
        volatility_score, dns_ips = _dns_volatility(registrable)
        if dns_ips:
            ips = dns_ips

    return DomainEnrichment(
        registrable_domain=registrable,
        age_days=age_days,
        age_trust=_age_trust_score(age_days),
        asn=asn,
        asn_org=asn_org,
        reputation_trust=reputation,
        reputation_reasons=reasons,
        volatility_score=volatility_score,
        ip_addresses=ips,
    )


def default_domain_enrichment(host: str) -> DomainEnrichment:
    return DomainEnrichment(
        registrable_domain=host,
        age_days=None,
        age_trust=0.5,
        asn=None,
        asn_org=None,
        reputation_trust=0.5,
        reputation_reasons=["enrichment_disabled"],
        volatility_score=0.0,
        ip_addresses=[],
    )
