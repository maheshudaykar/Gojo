from __future__ import annotations

import argparse
import csv
import json
import logging
from typing import Any

from phish_detector.analyze import AnalysisConfig, analyze_url, load_ml_context
from phish_detector.policy import BanditPolicy
from phish_detector.scoring import label_for_score


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gojo phishing URL detector (bulk + polish)")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--url", help="URL to analyze")
    target.add_argument("--input-csv", help="CSV file containing URLs")
    parser.add_argument("--url-col", default="url", help="CSV column with URLs")
    parser.add_argument(
        "--format",
        choices=["json", "text"],
        default="json",
        help="Output format",
    )
    parser.add_argument(
        "--output",
        help="Write results to a file (for bulk mode)",
    )
    parser.add_argument(
        "--output-format",
        choices=["csv", "json", "jsonl"],
        default="csv",
        help="Bulk output format",
    )
    parser.add_argument(
        "--ml-mode",
        choices=["none", "lexical", "char", "ensemble"],
        default="ensemble",
        help="ML inference mode",
    )
    parser.add_argument("--lexical-model", default="models/lexical_model.joblib")
    parser.add_argument("--char-model", default="models/char_model.joblib")
    parser.add_argument("--policy-path", default="models/policy.json")
    parser.add_argument("--feedback-store", default="models/feedback.json")
    parser.add_argument("--label", choices=["phish", "legit"], help="Feedback label")
    parser.add_argument("--resolve-feedback", help="Resolve a pending feedback id")
    parser.add_argument("--explain", action="store_true", help="Show full scoring breakdown")
    parser.add_argument("--shadow-learn", action="store_true", help="Update policy without affecting output")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser


def _build_config(args: argparse.Namespace, enable_feedback: bool) -> AnalysisConfig:
    return AnalysisConfig(
        ml_mode=args.ml_mode,
        lexical_model=args.lexical_model,
        char_model=args.char_model,
        policy_path=args.policy_path,
        feedback_store=args.feedback_store,
        shadow_learn=args.shadow_learn,
        label=args.label,
        resolve_feedback=args.resolve_feedback,
        enable_feedback=enable_feedback,
    )


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    config = _build_config(args, enable_feedback=True)
    ml_context = load_ml_context(config) if args.ml_mode != "none" else None
    policy = BanditPolicy(args.policy_path) if args.ml_mode != "none" else None

    if args.input_csv:
        if not args.output:
            raise SystemExit("Bulk mode requires --output")
        reports: list[dict[str, Any]] = []
        with open(args.input_csv, "r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            with open(args.output, "w", encoding="utf-8", newline="") as out_handle:
                writer: csv.DictWriter[str] | None = None
                if args.output_format == "csv":
                    writer = csv.DictWriter(
                        out_handle,
                        fieldnames=[
                            "url",
                            "score",
                            "label",
                            "rule_score",
                            "ml_score",
                            "ml_confidence",
                            "policy_weight",
                            "signals",
                            "brand_typo_risk",
                        ],
                    )
                    writer.writeheader()
                for row in reader:
                    url = (row.get(args.url_col) or "").strip()
                    if not url:
                        logging.warning("Skipping row without URL")
                        continue
                    report, extra = analyze_url(
                        url,
                        _build_config(args, enable_feedback=False),
                        ml_context=ml_context,
                        policy=policy,
                    )
                    summary = report["summary"]
                    if args.output_format == "csv":
                        if writer is None:
                            raise RuntimeError("CSV writer was not initialized.")
                        writer.writerow(
                            {
                                "url": url,
                                "score": summary["score"],
                                "label": summary["label"],
                                "rule_score": extra["rule_score"],
                                "ml_score": extra["ml_score"],
                                "ml_confidence": f"{extra['ml_confidence']:.4f}",
                                "policy_weight": extra["policy_weight"],
                                "signals": ";".join(hit["name"] for hit in extra["signals"]),
                                "brand_typo_risk": f"{extra.get('brand_typo_risk', 0.0):.2f}",
                            }
                        )
                    elif args.output_format == "jsonl":
                        out_handle.write(json.dumps(report) + "\n")
                    else:
                        reports.append(report)
                if args.output_format == "json":
                    out_handle.write(json.dumps(reports, indent=2))
    else:
        report, extra = analyze_url(
            args.url,
            config,
            ml_context=ml_context,
            policy=policy,
        )
        if args.format == "json":
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            print(f"Score: {report['summary']['score']} ({report['summary']['label']})")
            if args.explain:
                print(f"Rule score: {extra['rule_score']} ({label_for_score(extra['rule_score'])})")
                if extra["ml_score"] is not None:
                    print(f"ML score: {extra['ml_score']} (confidence {extra['ml_confidence']:.2f})")
                if extra["policy_weight"] is not None:
                    print(f"Policy weight: {extra['policy_weight']} (shadow={args.shadow_learn})")
            signals = extra["signals"]
            if signals:
                print("Signals:")
                for hit in signals:
                    print(f"- {hit['name']}: {hit['details']} (weight {hit['weight']})")
            else:
                print("Signals: none")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
