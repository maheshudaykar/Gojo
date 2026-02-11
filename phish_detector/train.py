from __future__ import annotations

import argparse

from phish_detector.ml_char_ngram import train_char_model
from phish_detector.ml_lexical import train_lexical_model


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train ML models for phishing URL detection")
    parser.add_argument("--data", required=True, help="Path to CSV dataset")
    parser.add_argument("--url-col", default="url", help="URL column name")
    parser.add_argument("--label-col", default="label", help="Label column name")
    parser.add_argument("--out-dir", default="models", help="Output directory")
    parser.add_argument(
        "--mode",
        choices=["lexical", "char", "both"],
        default="both",
        help="Which model(s) to train",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    if args.mode in {"lexical", "both"}:
        train_lexical_model(
            dataset_path=args.data,
            url_col=args.url_col,
            label_col=args.label_col,
            output_path=f"{args.out_dir}/lexical_model.joblib",
            metadata_path=f"{args.out_dir}/lexical_metadata.json",
        )
    if args.mode in {"char", "both"}:
        train_char_model(
            dataset_path=args.data,
            url_col=args.url_col,
            label_col=args.label_col,
            output_path=f"{args.out_dir}/char_model.joblib",
            metadata_path=f"{args.out_dir}/char_metadata.json",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
