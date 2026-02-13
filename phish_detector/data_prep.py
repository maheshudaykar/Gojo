from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from phish_detector.analyze import AnalysisConfig, analyze_url, load_ml_context
from phish_detector.parsing import get_registrable_domain, parse_url
from phish_detector.scoring import binary_label_for_score


@dataclass(frozen=True)
class QualityResult:
    score: float
    flags: list[str]
    needs_review: bool


def _normalize_label(value: str) -> int:
    label = value.strip().lower()
    if label in {"1", "phish", "phishing", "malicious", "bad"}:
        return 1
    if label in {"0", "legit", "benign", "good", "safe"}:
        return 0
    raise ValueError(f"Unsupported label: {value}")


def load_rows(path: Path, url_col: str, label_col: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for idx, row in enumerate(reader):
            if url_col not in row or label_col not in row:
                raise KeyError("Dataset missing required columns.")
            url = (row.get(url_col) or "").strip()
            if not url:
                continue
            label = _normalize_label(str(row[label_col]))
            rows.append({"idx": idx, "url": url, "label": label})
    return rows


def dedup_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    seen: dict[str, dict[str, Any]] = {}
    conflicts = 0
    for row in rows:
        parsed = parse_url(row["url"])
        key = parsed.normalized
        if key in seen:
            if seen[key]["label"] != row["label"]:
                conflicts += 1
            continue
        seen[key] = row
    return list(seen.values()), conflicts


def family_split(
    rows: list[dict[str, Any]],
    test_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    families: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        host = parse_url(row["url"]).host
        family = get_registrable_domain(host)
        families.setdefault(family, []).append(row)

    rng = random.Random(seed)
    family_keys = list(families.keys())
    rng.shuffle(family_keys)

    test_count = max(1, int(len(family_keys) * test_ratio))
    test_keys = set(family_keys[:test_count])

    train_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    for family, items in families.items():
        if family in test_keys:
            test_rows.extend(items)
        else:
            train_rows.extend(items)
    return train_rows, test_rows


def score_label_quality(
    row: dict[str, Any],
    config: AnalysisConfig,
    ml_context: dict[str, Any] | None,
    mismatch_confidence: float,
    hard_negative_threshold: float,
) -> QualityResult:
    report, extra = analyze_url(row["url"], config, ml_context=ml_context, policy=None)
    predicted = binary_label_for_score(report["summary"]["score"])
    true_label = "phish" if row["label"] == 1 else "legit"
    match = predicted == true_label

    rule_score = float(extra.get("rule_score", 0.0))
    ml_confidence = float(extra.get("ml_confidence", 0.0))
    confidence = ml_confidence if extra.get("ml_score") is not None else min(rule_score / 100.0, 1.0)

    flags: list[str] = []
    if not match and confidence >= mismatch_confidence:
        flags.append("mismatch_high_confidence")
    if not match and confidence < mismatch_confidence:
        flags.append("mismatch_low_confidence")
    if confidence < 0.55:
        flags.append("low_confidence")

    brand_risk = float(extra.get("brand_typo_risk", 0.0))
    if row["label"] == 0 and brand_risk >= hard_negative_threshold:
        flags.append("benign_high_brand_risk")
    if row["label"] == 1 and report["summary"]["score"] < 40:
        flags.append("phish_low_score")

    if match:
        quality = 0.7 + 0.3 * confidence
    else:
        quality = 0.2 * (1.0 - confidence)

    quality = max(0.0, min(1.0, quality))
    needs_review = (not match and confidence >= mismatch_confidence) or quality < 0.4

    return QualityResult(score=quality, flags=flags, needs_review=needs_review)


def mine_hard_negatives(
    rows: list[dict[str, Any]],
    config: AnalysisConfig,
    ml_context: dict[str, Any] | None,
    threshold: float,
) -> list[dict[str, Any]]:
    hard: list[dict[str, Any]] = []
    for row in rows:
        if row["label"] != 0:
            continue
        report, extra = analyze_url(row["url"], config, ml_context=ml_context, policy=None)
        score = float(report["summary"]["score"])
        brand_risk = float(extra.get("brand_typo_risk", 0.0))
        if brand_risk >= threshold or score >= threshold:
            hard.append({"url": row["url"], "score": score, "brand_risk": brand_risk})
    return hard


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Gojo data prep and quality tooling")
    parser.add_argument("--input", required=True, help="Input CSV path")
    parser.add_argument("--url-col", default="url", help="URL column name")
    parser.add_argument("--label-col", default="label", help="Label column name")
    parser.add_argument("--output-dir", default="results/data_prep", help="Output directory")
    parser.add_argument("--dedup", action="store_true", help="Remove duplicate URLs")
    parser.add_argument("--split-mode", choices=["none", "family"], default="family")
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hard-negative-threshold", type=float, default=70.0)
    parser.add_argument("--mismatch-confidence", type=float, default=0.7)
    parser.add_argument("--quality-threshold", type=float, default=0.4)
    parser.add_argument("--ml-mode", choices=["none", "lexical", "char", "ensemble"], default="ensemble")
    parser.add_argument("--lexical-model", default="models/lexical_model.joblib")
    parser.add_argument("--char-model", default="models/char_model.joblib")
    parser.add_argument("--enable-enrichment", action="store_true", help="Allow enrichment lookups")
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    input_path = Path(args.input)
    rows = load_rows(input_path, args.url_col, args.label_col)

    conflicts = 0
    original_count = len(rows)
    if args.dedup:
        rows, conflicts = dedup_rows(rows)

    config = AnalysisConfig(
        ml_mode=args.ml_mode,
        lexical_model=args.lexical_model,
        char_model=args.char_model,
        policy_path="models/policy.json",
        feedback_store="models/feedback.json",
        enable_feedback=False,
        enable_enrichment=args.enable_enrichment,
        enable_brand_risk=True,
        enable_policy=False,
    )
    ml_context = load_ml_context(config) if args.ml_mode != "none" else None

    hard_negatives = mine_hard_negatives(
        rows,
        config,
        ml_context,
        threshold=args.hard_negative_threshold,
    )
    _write_rows(output_dir / "hard_negatives.csv", hard_negatives)

    quality_rows: list[dict[str, Any]] = []
    review_rows: list[dict[str, Any]] = []
    for row in rows:
        quality = score_label_quality(
            row,
            config,
            ml_context,
            mismatch_confidence=args.mismatch_confidence,
            hard_negative_threshold=args.hard_negative_threshold,
        )
        quality_rows.append(
            {
                "url": row["url"],
                "label": row["label"],
                "quality_score": f"{quality.score:.3f}",
                "flags": ";".join(quality.flags),
                "needs_review": quality.needs_review,
            }
        )
        if quality.needs_review or quality.score < args.quality_threshold:
            review_rows.append(quality_rows[-1])

    _write_rows(output_dir / "label_quality.csv", quality_rows)
    _write_rows(output_dir / "label_review.csv", review_rows)

    split_report: dict[str, Any] = {
        "dataset": {
            "path": str(input_path),
            "sha256": _sha256_file(input_path),
            "rows_original": original_count,
            "rows_after": len(rows),
        },
        "conflicts": conflicts,
        "hard_negatives": len(hard_negatives),
        "quality_review": len(review_rows),
        "split_recipe": {
            "mode": args.split_mode,
            "seed": args.seed,
            "test_ratio": args.test_ratio,
            "dedup": args.dedup,
            "url_col": args.url_col,
            "label_col": args.label_col,
        },
        "preprocessing": {
            "normalize_labels": True,
            "family_split": args.split_mode == "family",
            "enable_enrichment": args.enable_enrichment,
            "ml_mode": args.ml_mode,
        },
    }

    if args.split_mode == "family":
        train_rows, test_rows = family_split(rows, args.test_ratio, args.seed)
        _write_rows(output_dir / "train.csv", train_rows)
        _write_rows(output_dir / "test.csv", test_rows)
        split_report.update({"train": len(train_rows), "test": len(test_rows)})

    (output_dir / "prep_report.json").write_text(json.dumps(split_report, indent=2), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
