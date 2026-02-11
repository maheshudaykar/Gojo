from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from phish_detector.enrichment import DomainEnrichment
from phish_detector.intent import find_intent_tokens
from phish_detector.parsing import ParsedURL
from phish_detector.typosquat import TyposquatMatch, keyboard_adjacent_score


@dataclass(frozen=True)
class BrandRiskResult:
    score: float
    components: dict[str, float]
    corroborating: list[str]
    typo_match: TyposquatMatch | None


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def _typo_score(match: TyposquatMatch | None) -> float:
    if match is None:
        return 0.0
    brand_len = max(1, len(match.brand))
    base = max(0.0, 1.0 - (match.distance / brand_len))
    if match.method == "substring":
        base = max(base, 0.8)
    if match.method == "transposition":
        base = max(base, 0.9)
    keyboard_score = keyboard_adjacent_score(match.candidate, match.brand)
    return max(base, keyboard_score)


def compute_brand_typo_risk(
    parsed: ParsedURL,
    features: dict[str, Any],
    match: TyposquatMatch | None,
    enrichment: DomainEnrichment,
) -> BrandRiskResult:
    intent_tokens = find_intent_tokens(f"{parsed.path}{parsed.query}")
    intent_score = 1.0 if intent_tokens else 0.0

    entropy = float(features.get("host_entropy", 0.0))
    if entropy >= 3.8:
        entropy_score = 1.0
    elif entropy >= 3.2:
        entropy_score = 0.5
    else:
        entropy_score = 0.0

    suspicious_tld = 1.0 if features.get("is_suspicious_tld") else 0.0
    homoglyph_score = 1.0 if features.get("has_homoglyph") else 0.0

    obfuscation_flags = [
        bool(features.get("has_encoded_chars")),
        bool(features.get("has_at_symbol")),
        bool(features.get("has_ip")),
        bool(features.get("has_port")),
    ]
    obfuscation_score = min(sum(1.0 for flag in obfuscation_flags if flag) / 3.0, 1.0)
    obfuscation_score = max(obfuscation_score, enrichment.volatility_score)

    typo_score = _typo_score(match)

    age_trust = enrichment.age_trust
    reputation_trust = enrichment.reputation_trust

    value = (
        1.8 * typo_score
        + 1.4 * intent_score
        + 1.2 * homoglyph_score
        + 1.1 * suspicious_tld
        + 0.9 * entropy_score
        + 0.7 * obfuscation_score
        - 1.3 * age_trust
        - 0.8 * reputation_trust
    )
    score = 100.0 * _sigmoid(value)

    corroborating: list[str] = []
    if intent_score:
        corroborating.append("intent")
    if suspicious_tld:
        corroborating.append("suspicious_tld")
    if entropy_score >= 0.5:
        corroborating.append("high_entropy")
    if obfuscation_score >= 0.5:
        corroborating.append("obfuscation")
    if homoglyph_score:
        corroborating.append("homoglyph")

    components = {
        "T": typo_score,
        "I": intent_score,
        "H": homoglyph_score,
        "S": suspicious_tld,
        "E": entropy_score,
        "U": obfuscation_score,
        "A": age_trust,
        "R": reputation_trust,
    }

    return BrandRiskResult(
        score=score,
        components=components,
        corroborating=corroborating,
        typo_match=match,
    )
