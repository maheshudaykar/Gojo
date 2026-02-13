from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable
from urllib.parse import quote, urlsplit, urlunsplit

from phish_detector.parsing import parse_url


@dataclass(frozen=True)
class Perturbation:
    name: str
    url: str


_CONFUSABLES = {
    "a": "а",
    "c": "с",
    "e": "е",
    "i": "і",
    "o": "о",
    "p": "р",
    "y": "у",
    "x": "х",
}


def _replace_host(original_url: str, new_host: str) -> str:
    parsed = urlsplit(_ensure_scheme(original_url))
    netloc = new_host
    if parsed.port:
        netloc = f"{new_host}:{parsed.port}"
    return urlunsplit((parsed.scheme, netloc, parsed.path, parsed.query, parsed.fragment))


def _ensure_scheme(url: str) -> str:
    if "://" in url:
        return url
    return f"http://{url}"


def _random_typo(host: str) -> str | None:
    if len(host) < 4:
        return None
    idx = random.randint(1, len(host) - 2)
    choice = random.choice(["delete", "swap", "replace"])
    if choice == "delete":
        return host[:idx] + host[idx + 1:]
    if choice == "swap" and idx < len(host) - 1:
        return host[:idx] + host[idx + 1] + host[idx] + host[idx + 2:]
    letters = "abcdefghijklmnopqrstuvwxyz"
    return host[:idx] + random.choice(letters) + host[idx + 1:]


def _homoglyph(host: str) -> str | None:
    for idx, ch in enumerate(host):
        repl = _CONFUSABLES.get(ch)
        if repl:
            return host[:idx] + repl + host[idx + 1:]
    return None


def _encode_path(url: str) -> str | None:
    parsed = urlsplit(_ensure_scheme(url))
    if not parsed.path:
        return None
    encoded = quote(parsed.path, safe="/=")
    return urlunsplit((parsed.scheme, parsed.netloc, encoded, parsed.query, parsed.fragment))


def _deep_subdomain(host: str, depth: int = 3) -> str:
    tokens = [f"secure{idx}" for idx in range(depth)]
    return ".".join(tokens + [host])


def generate_perturbations(url: str, max_variants: int = 6) -> list[Perturbation]:
    parsed = parse_url(url)
    host = parsed.host
    if not host:
        return []

    variants: list[Perturbation] = []

    typo = _random_typo(host)
    if typo:
        variants.append(Perturbation("typo", _replace_host(url, typo)))

    transposition = _random_typo(host)
    if transposition and transposition != typo:
        variants.append(Perturbation("transposition", _replace_host(url, transposition)))

    homoglyph = _homoglyph(host)
    if homoglyph:
        variants.append(Perturbation("homoglyph", _replace_host(url, homoglyph)))

    encoded = _encode_path(url)
    if encoded:
        variants.append(Perturbation("url_encoding", encoded))

    deep = _deep_subdomain(host, depth=4)
    variants.append(Perturbation("deep_subdomain", _replace_host(url, deep)))

    if len(variants) > max_variants:
        variants = variants[:max_variants]

    return variants


def iter_perturbations(urls: Iterable[str]) -> list[Perturbation]:
    all_variants: list[Perturbation] = []
    for url in urls:
        all_variants.extend(generate_perturbations(url))
    return all_variants
