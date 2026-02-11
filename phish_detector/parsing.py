from __future__ import annotations

import ipaddress
import json
import re
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Optional
from urllib.parse import unquote, urlsplit, urlunsplit

_ENCODED_RE = re.compile(r"%[0-9A-Fa-f]{2}")


@dataclass(frozen=True)
class ParsedURL:
    original: str
    normalized: str
    scheme: str
    netloc: str
    host: str
    port: Optional[int]
    path: str
    query: str
    fragment: str
    is_ip: bool
    is_shortener: bool
    has_encoded_chars: bool
    decoded_url: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)


@lru_cache(maxsize=1)
def _load_shortener_hosts() -> set[str]:
    root = Path(__file__).resolve().parents[1]
    shorteners_path = root / "configs" / "shorteners.txt"
    hosts: set[str] = set()
    if shorteners_path.exists():
        for line in shorteners_path.read_text(encoding="utf-8").splitlines():
            entry = line.strip().lower()
            if entry and not entry.startswith("#"):
                hosts.add(entry)
    return hosts


def load_shortener_hosts(extra_hosts: Optional[Iterable[str]] = None) -> set[str]:
    """Load known URL shortener hosts from configs/shorteners.txt."""
    hosts = set(_load_shortener_hosts())
    if extra_hosts:
        hosts.update(h.strip().lower() for h in extra_hosts if h.strip())
    return hosts


def _ensure_scheme(url: str) -> str:
    if "://" in url:
        return url
    return f"http://{url}"


def _normalize_host(host: str) -> str:
    if not host:
        return host
    host = host.strip().lower()
    if host.endswith("."):
        host = host[:-1]
    # Convert unicode domains to punycode for consistent comparison.
    try:
        host = host.encode("idna").decode("ascii")
    except UnicodeError:
        pass
    return host


def _is_ip_address(host: str) -> bool:
    if not host:
        return False
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return False


def parse_url(url: str, shortener_hosts: Optional[Iterable[str]] = None) -> ParsedURL:
    """Parse and normalize a URL for downstream phishing analysis."""
    if not url or not url.strip():
        raise ValueError("URL must be a non-empty string.")

    original = url.strip()
    normalized_input = _ensure_scheme(original)
    split = urlsplit(normalized_input)

    host = _normalize_host(split.hostname or "")
    port = split.port
    scheme = split.scheme.lower()
    netloc = split.netloc

    normalized = urlunsplit((scheme, split.netloc, split.path, split.query, split.fragment))
    decoded_url = unquote(original)

    shorteners = load_shortener_hosts(shortener_hosts)
    is_shortener = host in shorteners

    return ParsedURL(
        original=original,
        normalized=normalized,
        scheme=scheme,
        netloc=netloc,
        host=host,
        port=port,
        path=split.path,
        query=split.query,
        fragment=split.fragment,
        is_ip=_is_ip_address(host),
        is_shortener=is_shortener,
        has_encoded_chars=bool(_ENCODED_RE.search(original)),
        decoded_url=decoded_url,
    )
