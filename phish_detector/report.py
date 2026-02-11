from __future__ import annotations

from dataclasses import asdict
from typing import Any

from phish_detector.parsing import ParsedURL
from phish_detector.scoring import ScoreResult


def build_report(
    parsed: ParsedURL,
    features: dict[str, Any],
    score: ScoreResult,
    ml_info: dict[str, Any] | None = None,
    policy_info: dict[str, Any] | None = None,
    feedback_info: dict[str, Any] | None = None,
    context_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "summary": {
            "score": score.score,
            "label": score.label,
        },
        "parsed": asdict(parsed),
        "features": features,
        "signals": [asdict(hit) for hit in score.hits],
    }
    if ml_info is not None:
        report["ml"] = ml_info
    if policy_info is not None:
        report["policy"] = policy_info
    if feedback_info is not None:
        report["feedback"] = feedback_info
    if context_info is not None:
        report["context"] = context_info
    return report
