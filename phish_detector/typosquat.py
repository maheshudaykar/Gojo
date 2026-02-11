from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from phish_detector.parsing import ParsedURL

_COMMON_SLD_SUFFIXES = {"co", "com", "net", "org", "gov", "edu"}
_HOMOGLYPH_REPLACEMENTS = (
    ("rn", "m"),
    ("0", "o"),
    ("1", "l"),
    ("3", "e"),
    ("5", "s"),
    ("7", "t"),
    ("@", "a"),
    ("$", "s"),
    ("!", "i"),
)

_KEYBOARD_ADJACENT = {
    "a": set("qwsz"),
    "b": set("vghn"),
    "c": set("xdfv"),
    "d": set("ersfxc"),
    "e": set("wsdr"),
    "f": set("rtgdvc"),
    "g": set("tyfhvb"),
    "h": set("yugjbn"),
    "i": set("ujko"),
    "j": set("uikhnm"),
    "k": set("ioljmn"),
    "l": set("opk"),
    "m": set("njk"),
    "n": set("bhjm"),
    "o": set("iklp"),
    "p": set("ol"),
    "q": set("wa"),
    "r": set("edft"),
    "s": set("wedxza"),
    "t": set("rfgy"),
    "u": set("yhji"),
    "v": set("cfgb"),
    "w": set("qase"),
    "x": set("zsdc"),
    "y": set("tghu"),
    "z": set("asx"),
}


@dataclass(frozen=True)
class TyposquatMatch:
    brand: str
    candidate: str
    distance: int
    method: str


def load_brand_list(extra_brands: Iterable[str] | None = None) -> set[str]:
    root = Path(__file__).resolve().parents[1]
    brand_path = root / "configs" / "brands.txt"
    brands: set[str] = set()
    if brand_path.exists():
        for line in brand_path.read_text(encoding="utf-8").splitlines():
            entry = line.strip().lower()
            if entry and not entry.startswith("#"):
                brands.add(entry)
    if extra_brands:
        brands.update(b.strip().lower() for b in extra_brands if b.strip())
    return brands


def _registrable_label(host: str) -> str:
    parts = [part for part in host.split(".") if part]
    if len(parts) < 2:
        return host
    tld = parts[-1]
    sld = parts[-2]
    if len(tld) == 2 and sld in _COMMON_SLD_SUFFIXES and len(parts) >= 3:
        return parts[-3]
    return sld


def _normalize_homoglyphs(text: str) -> str:
    normalized = text
    for src, dest in _HOMOGLYPH_REPLACEMENTS:
        normalized = normalized.replace(src, dest)
    return normalized


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current = [i]
        for j, char_b in enumerate(b, start=1):
            insert = current[j - 1] + 1
            delete = previous[j] + 1
            substitute = previous[j - 1] + (char_a != char_b)
            current.append(min(insert, delete, substitute))
        previous = current
    return previous[-1]


def _damerau_levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    d: dict[tuple[int, int], int] = {}
    len_a = len(a)
    len_b = len(b)
    max_dist = len_a + len_b
    d[(0, 0)] = max_dist
    for i in range(0, len_a + 1):
        d[(i + 1, 1)] = i
        d[(i + 1, 0)] = max_dist
    for j in range(0, len_b + 1):
        d[(1, j + 1)] = j
        d[(0, j + 1)] = max_dist

    last_row: dict[str, int] = {}
    for i in range(1, len_a + 1):
        db = 0
        for j in range(1, len_b + 1):
            i1 = last_row.get(b[j - 1], 0)
            j1 = db
            cost = 0 if a[i - 1] == b[j - 1] else 1
            if cost == 0:
                db = j

            d[(i + 1, j + 1)] = min(
                d[(i, j)] + cost,
                d[(i + 1, j)] + 1,
                d[(i, j + 1)] + 1,
                d[(i1, j1)] + (i - i1 - 1) + 1 + (j - j1 - 1),
            )
        last_row[a[i - 1]] = i
    return d[(len_a + 1, len_b + 1)]


def keyboard_adjacent_score(a: str, b: str) -> float:
    if len(a) != len(b):
        return 0.0
    diff = [(ca, cb) for ca, cb in zip(a, b) if ca != cb]
    if len(diff) != 1:
        return 0.0
    src, dest = diff[0]
    if src in _KEYBOARD_ADJACENT and dest in _KEYBOARD_ADJACENT[src]:
        return 0.8
    return 0.0


def detect_typosquatting(parsed: ParsedURL) -> TyposquatMatch | None:
    brands = load_brand_list()
    if not brands:
        return None

    host = parsed.host.lower()
    if not host:
        return None

    labels = [label for label in host.split(".") if label]
    candidates = {label for label in labels if len(label) >= 3}
    for label in labels:
        for part in label.replace("_", "-").split("-"):
            if len(part) >= 3:
                candidates.add(part)
    candidates.add(_registrable_label(host))
    candidates = {candidate for candidate in candidates if candidate}
    best: TyposquatMatch | None = None

    for brand in brands:
        if len(brand) < 3:
            continue
        for candidate in candidates:
            if brand == candidate:
                continue

            if len(brand) >= 4 and brand in candidate and brand != candidate:
                match = TyposquatMatch(
                    brand=brand,
                    candidate=candidate,
                    distance=0,
                    method="substring",
                )
                if best is None or match.distance < best.distance:
                    best = match
                continue

            normalized = _normalize_homoglyphs(candidate)
            distance = _levenshtein(candidate, brand)
            norm_distance = _levenshtein(normalized, brand)
            trans_distance = _damerau_levenshtein(candidate, brand)
            effective_distance = min(distance, norm_distance, trans_distance)
            threshold = 1 if len(brand) <= 5 else 2

            if effective_distance <= threshold:
                if trans_distance < min(distance, norm_distance):
                    method = "transposition"
                elif norm_distance < distance:
                    method = "homoglyph"
                else:
                    method = "edit_distance"
                match = TyposquatMatch(
                    brand=brand,
                    candidate=candidate,
                    distance=effective_distance,
                    method=method,
                )
                if best is None or match.distance < best.distance:
                    best = match

    return best
