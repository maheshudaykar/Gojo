from __future__ import annotations

from dataclasses import dataclass

from phish_detector.features import FeatureValue, extract_features, load_suspicious_tlds
from phish_detector.intent import find_intent_tokens
from phish_detector.parsing import ParsedURL
from phish_detector.typosquat import detect_typosquatting

_SAFE_PORTS = {80, 443}


@dataclass(frozen=True)
class RuleHit:
    name: str
    weight: int
    details: str


def _as_int(value: FeatureValue) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    return 0


def _as_float(value: FeatureValue) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def evaluate_rules(parsed: ParsedURL) -> tuple[dict[str, FeatureValue], list[RuleHit]]:
    suspicious_tlds = load_suspicious_tlds()
    features = extract_features(parsed, suspicious_tlds)
    hits: list[RuleHit] = []

    if features["is_suspicious_tld"]:
        hits.append(RuleHit("suspicious_tld", 90, f"TLD '{features['tld']}' flagged"))

    if _as_int(features["num_subdomains"]) >= 3:
        hits.append(RuleHit("deep_subdomains", 10, "Excessive subdomain depth"))

    if _as_int(features["url_length"]) >= 75:
        hits.append(RuleHit("long_url", 8, "Unusually long URL"))

    if _as_int(features["num_dots"]) >= 4:
        hits.append(RuleHit("many_dots", 6, "Many dots in host"))

    if features["has_encoded_chars"]:
        hits.append(RuleHit("encoded_chars", 6, "Contains encoded characters"))

    if features["has_at_symbol"]:
        hits.append(RuleHit("at_symbol", 8, "Contains '@' symbol"))

    if features["has_ip"]:
        hits.append(RuleHit("ip_host", 12, "Host is an IP address"))

    if features["is_shortener"]:
        hits.append(RuleHit("shortener", 8, "Known URL shortener"))

    if parsed.port and parsed.port not in _SAFE_PORTS:
        hits.append(RuleHit("uncommon_port", 6, f"Port {parsed.port} is uncommon"))

    text_blob = f"{parsed.host}{parsed.path}{parsed.query}"
    matched_tokens = find_intent_tokens(text_blob)
    if matched_tokens:
        hits.append(
            RuleHit(
                "suspicious_tokens",
                10,
                f"Suspicious tokens: {', '.join(matched_tokens)}",
            )
        )

    if _as_float(features["host_entropy"]) >= 3.8:
        hits.append(RuleHit("high_entropy_host", 8, "Host looks random"))

    typosquat = detect_typosquatting(parsed)
    if typosquat:
        hits.append(
            RuleHit(
                "typosquat",
                15,
                (
                    f"Looks like '{typosquat.brand}' (candidate '{typosquat.candidate}', "
                    f"distance {typosquat.distance}, {typosquat.method})"
                ),
            )
        )

    return features, hits
