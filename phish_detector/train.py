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
    parser.add_argument(
        "--benchmark-data",
        help="Run benchmark after training using this dataset",
    )
    parser.add_argument("--benchmark-url-col", default="url", help="Benchmark URL column")
    parser.add_argument("--benchmark-label-col", default="label", help="Benchmark label column")
    parser.add_argument("--benchmark-time-col", default=None, help="Benchmark time column")
    parser.add_argument("--benchmark-output", default="results", help="Benchmark output directory")
    parser.add_argument("--benchmark-seeds", type=int, default=5, help="Benchmark seeds")
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
    if args.benchmark_data:
        from phish_detector.benchmark import main as bench_main

        bench_args: list[str] = [
            "--data",
            args.benchmark_data,
            "--url-col",
            args.benchmark_url_col,
            "--label-col",
            args.benchmark_label_col,
            "--output-dir",
            args.benchmark_output,
            "--seeds",
            str(args.benchmark_seeds),
        ]
        if args.benchmark_time_col:
            bench_args.extend(["--time-col", args.benchmark_time_col])
        bench_main(bench_args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
