from __future__ import annotations

import math
import re
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Mapping

from phish_detector.parsing import ParsedURL

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_HOMOGLYPH_RE = re.compile(r"rn|[01357@$!]")

FeatureValue = float | int | bool | str

FEATURE_SCHEMA = [
    "url_length",
    "host_length",
    "path_length",
    "query_length",
    "num_digits",
    "num_specials",
    "num_tokens",
    "num_subdomains",
    "num_dots",
    "has_at_symbol",
    "has_ip",
    "has_port",
    "is_shortener",
    "has_encoded_chars",
    "is_suspicious_tld",
    "host_entropy",
    "path_entropy",
    "uppercase_ratio",
    "max_repeat_run",
    "has_homoglyph",
]


@lru_cache(maxsize=1)
def _load_suspicious_tlds() -> set[str]:
    root = Path(__file__).resolve().parents[1]
    tld_path = root / "configs" / "suspicious_tlds.txt"
    tlds: set[str] = set()
    if tld_path.exists():
        for line in tld_path.read_text(encoding="utf-8").splitlines():
            entry = line.strip().lower()
            if not entry or entry.startswith("#"):
                continue
            tlds.add(entry.lstrip("."))
    return tlds


def load_suspicious_tlds(extra_tlds: Iterable[str] | None = None) -> set[str]:
    """Load suspicious TLDs from configs/suspicious_tlds.txt."""
    tlds = set(_load_suspicious_tlds())
    if extra_tlds:
        tlds.update(t.strip().lower().lstrip(".") for t in extra_tlds if t.strip())
    return tlds


def shannon_entropy(text: str) -> float:
    if not text:
        return 0.0
    counts: dict[str, int] = {}
    for char in text:
        counts[char] = counts.get(char, 0) + 1
    length = len(text)
    entropy = 0.0
    for count in counts.values():
        p = count / length
        entropy -= p * math.log2(p)
    return entropy


def _uppercase_ratio(text: str) -> float:
    letters = [char for char in text if char.isalpha()]
    if not letters:
        return 0.0
    upper = sum(char.isupper() for char in letters)
    return round(upper / len(letters), 4)


def _max_repeat_run(text: str) -> int:
    if not text:
        return 0
    max_run = 1
    current = 1
    for prev, curr in zip(text, text[1:]):
        if curr == prev:
            current += 1
            max_run = max(max_run, current)
        else:
            current = 1
    return max_run


def extract_features(parsed: ParsedURL, suspicious_tlds: set[str]) -> dict[str, FeatureValue]:
    host = parsed.host
    path = parsed.path
    query = parsed.query
    url_text = parsed.original

    host_parts = [part for part in host.split(".") if part]
    tld = host_parts[-1] if host_parts else ""

    tokens = _TOKEN_RE.findall(host + path + query)
    num_digits = sum(char.isdigit() for char in url_text)
    num_specials = sum(not char.isalnum() for char in url_text)
    uppercase_ratio = _uppercase_ratio(url_text)
    max_repeat_run = _max_repeat_run(url_text)
    has_homoglyph = bool(_HOMOGLYPH_RE.search(host))

    return {
        "url_length": len(url_text),
        "host_length": len(host),
        "path_length": len(path),
        "query_length": len(query),
        "num_digits": num_digits,
        "num_specials": num_specials,
        "num_tokens": len(tokens),
        "num_subdomains": max(len(host_parts) - 2, 0),
        "num_dots": host.count("."),
        "has_at_symbol": "@" in url_text,
        "has_ip": parsed.is_ip,
        "has_port": parsed.port is not None,
        "is_shortener": parsed.is_shortener,
        "has_encoded_chars": parsed.has_encoded_chars,
        "tld": tld,
        "is_suspicious_tld": tld in suspicious_tlds,
        "host_entropy": round(shannon_entropy(host), 4),
        "path_entropy": round(shannon_entropy(path), 4),
        "uppercase_ratio": uppercase_ratio,
        "max_repeat_run": max_repeat_run,
        "has_homoglyph": has_homoglyph,
    }


def vectorize_features(features: Mapping[str, FeatureValue]) -> list[float]:
    vector: list[float] = []
    for name in FEATURE_SCHEMA:
        value = features.get(name, 0)
        if isinstance(value, bool):
            vector.append(1.0 if value else 0.0)
        elif isinstance(value, (int, float)):
            vector.append(float(value))
        else:
            vector.append(0.0)
    return vector


def get_feature_schema() -> list[str]:
    return list(FEATURE_SCHEMA)
