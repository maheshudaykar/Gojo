from __future__ import annotations

from dataclasses import dataclass

from phish_detector.rules import RuleHit

GREEN_MAX_SCORE = 25
YELLOW_MAX_SCORE = 60


@dataclass(frozen=True)
class ScoreResult:
    score: int
    label: str
    hits: list[RuleHit]


def label_for_score(score: int) -> str:
    if score <= GREEN_MAX_SCORE:
        return "green"
    if score <= YELLOW_MAX_SCORE:
        return "yellow"
    return "red"


def binary_label_for_score(score: int) -> str:
    return "phish" if score > YELLOW_MAX_SCORE else "legit"


def compute_score(hits: list[RuleHit]) -> ScoreResult:
    raw_score = sum(hit.weight for hit in hits)
    score = max(0, min(100, raw_score))

    return ScoreResult(score=score, label=label_for_score(score), hits=hits)


def combine_scores(rule_score: int, ml_score: int, weight: float, hits: list[RuleHit]) -> ScoreResult:
    weight = min(max(weight, 0.0), 1.0)
    combined = int(round((weight * ml_score) + ((1.0 - weight) * rule_score)))
    combined = max(0, min(100, combined))
    return ScoreResult(score=combined, label=label_for_score(combined), hits=hits)
