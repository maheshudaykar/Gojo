from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
import time
import uuid
from datetime import datetime
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Literal, cast

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split as _train_test_split  # type: ignore[import-untyped]
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from phish_detector.adversarial import generate_perturbations
from phish_detector.analyze import AnalysisConfig, analyze_url
from phish_detector.brand_risk import BrandRiskConfig
from phish_detector.experiment_manifest import create_experiment_manifest
from phish_detector.features import extract_features, load_suspicious_tlds, vectorize_features
from phish_detector.feedback import FeedbackEntry, load_entries
from phish_detector.offpolicy import OffPolicyResult, evaluate_offpolicy, evaluate_offpolicy_entries
from phish_detector.parsing import parse_url
from phish_detector.policy import BanditPolicy
from phish_detector.policy_v2 import ThompsonSamplingPolicy
from phish_detector.scoring import binary_label_for_score
from phish_detector.typosquat import detect_typosquatting

TrainTestSplit = Any
train_test_split: TrainTestSplit = cast(Any, _train_test_split)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class BenchmarkRun:
    split: str
    baseline: str
    seed: int
    auroc: float
    auprc: float
    f1: float
    ece: float
    mce: float
    brier: float
    latency_p50_ms: float
    latency_p95_ms: float
    count: int
    positive_rate: float
    review_rate: float
    brand_count: int
    brand_auroc: float | None
    brand_auprc: float | None
    brand_f1: float | None


def _normalize_label(value: str) -> int:
    label = value.strip().lower()
    if label in {"1", "phish", "phishing", "malicious", "bad"}:
        return 1
    if label in {"0", "legit", "benign", "good", "safe"}:
        return 0
    raise ValueError(f"Unsupported label: {value}")


def _parse_time(value: str) -> float | None:
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        pass
    try:
        return datetime_from_iso(value).timestamp()
    except ValueError:
        return None


def datetime_from_iso(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def load_dataset(path: Path, url_col: str, label_col: str, time_col: str | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for idx, row in enumerate(reader):
            if url_col not in row or label_col not in row:
                raise KeyError("Dataset missing required columns.")
            url = row[url_col].strip()
            if not url:
                continue
            label = _normalize_label(row[label_col])
            timestamp = None
            if time_col:
                timestamp = _parse_time(row.get(time_col, ""))
            rows.append({"idx": idx, "url": url, "label": label, "time": timestamp})
    return rows


def time_split(rows: list[dict[str, Any]], test_ratio: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sorted_rows = sorted(
        rows,
        key=lambda row: row["time"] if row["time"] is not None else row["idx"],
    )
    cutoff = max(1, int(len(sorted_rows) * (1.0 - test_ratio)))
    return sorted_rows[:cutoff], sorted_rows[cutoff:]


def random_split(
    rows: list[dict[str, Any]],
    test_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    labels = [row["label"] for row in rows]
    split_rows = cast(
        tuple[list[dict[str, Any]], list[dict[str, Any]]],
        train_test_split(
            rows,
            test_size=test_ratio,
            random_state=seed,
            stratify=labels,
        ),
    )
    train_rows, test_rows = split_rows
    return list(train_rows), list(test_rows)


def expected_calibration_error(y_true: list[int], y_prob: list[float], bins: int = 10) -> float:
    if not y_true:
        return 0.0
    bin_edges = [i / bins for i in range(bins + 1)]
    ece = 0.0
    total = len(y_true)
    for i in range(bins):
        lower, upper = bin_edges[i], bin_edges[i + 1]
        indices = [idx for idx, prob in enumerate(y_prob) if lower <= prob < upper]
        if not indices:
            continue
        acc = sum(y_true[idx] for idx in indices) / len(indices)
        conf = sum(y_prob[idx] for idx in indices) / len(indices)
        ece += abs(acc - conf) * (len(indices) / total)
    return ece


def max_calibration_error(y_true: list[int], y_prob: list[float], bins: int = 10) -> float:
    if not y_true:
        return 0.0
    bin_edges = [i / bins for i in range(bins + 1)]
    max_gap = 0.0
    for i in range(bins):
        lower, upper = bin_edges[i], bin_edges[i + 1]
        indices = [idx for idx, prob in enumerate(y_prob) if lower <= prob < upper]
        if not indices:
            continue
        acc = sum(y_true[idx] for idx in indices) / len(indices)
        conf = sum(y_prob[idx] for idx in indices) / len(indices)
        max_gap = max(max_gap, abs(acc - conf))
    return max_gap


def reliability_bins(y_true: list[int], y_prob: list[float], bins: int = 10) -> list[dict[str, float]]:
    if not y_true:
        return []
    bin_edges = [i / bins for i in range(bins + 1)]
    results: list[dict[str, float]] = []
    for i in range(bins):
        lower, upper = bin_edges[i], bin_edges[i + 1]
        indices = [idx for idx, prob in enumerate(y_prob) if lower <= prob < upper]
        if not indices:
            results.append({"bin": float(i), "count": 0.0, "acc": 0.0, "conf": 0.0})
            continue
        acc = sum(y_true[idx] for idx in indices) / len(indices)
        conf = sum(y_prob[idx] for idx in indices) / len(indices)
        results.append({"bin": float(i), "count": float(len(indices)), "acc": float(acc), "conf": float(conf)})
    return results


def train_lexical_pipeline(
    urls: list[str],
    labels: list[int],
    seed: int,
    method: Literal["sigmoid", "isotonic"] = "sigmoid",
) -> Pipeline:
    suspicious_tlds = load_suspicious_tlds()
    feature_rows: list[list[float]] = []
    for url in urls:
        parsed = parse_url(url)
        features = extract_features(parsed, suspicious_tlds)
        feature_rows.append(vectorize_features(features))

    from sklearn.ensemble import RandomForestClassifier
    base_model = RandomForestClassifier(n_estimators=200, max_depth=30, class_weight="balanced", random_state=seed)

    pipeline: Pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", base_model),
        ]
    )
    cast(Any, pipeline).fit(feature_rows, labels)
    return pipeline


def train_char_pipeline(
    urls: list[str],
    labels: list[int],
    seed: int,
    method: Literal["sigmoid", "isotonic"] = "sigmoid",
) -> Pipeline:
    try:
        calibrated = CalibratedClassifierCV(estimator=LinearSVC(random_state=seed), cv=3, method=method)
    except TypeError:
        calibrated = CalibratedClassifierCV(base_estimator=LinearSVC(random_state=seed), cv=3, method=method)

    pipeline: Pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char",
                    ngram_range=(3, 5),
                    min_df=2,
                ),
            ),
            ("clf", calibrated),
        ]
    )
    cast(Any, pipeline).fit(urls, labels)
    return pipeline


def train_xgboost_pipeline(
    urls: list[str],
    labels: list[int],
    seed: int,
) -> Pipeline:
    suspicious_tlds = load_suspicious_tlds()
    feature_rows: list[list[float]] = []
    for url in urls:
        parsed = parse_url(url)
        features = extract_features(parsed, suspicious_tlds)
        feature_rows.append(vectorize_features(features))

    import xgboost as xgb
    base_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        random_state=seed,
        eval_metric='logloss'
    )

    pipeline: Pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", base_model),
        ]
    )
    cast(Any, pipeline).fit(feature_rows, labels)
    return pipeline


def build_config(
    baseline: str,
    brand_cfg: BrandRiskConfig | None = None,
    enable_enrichment: bool = True,
    enable_context_enrichment: bool = True,
    enable_brand_risk: bool = True,
) -> tuple[AnalysisConfig, BanditPolicy | ThompsonSamplingPolicy | None]:
    policy: BanditPolicy | ThompsonSamplingPolicy | None = None
    score_mode = "fusion"
    static_weight = 0.6
    enable_policy = True
    ml_mode = "ensemble"

    if baseline == "rules-only":
        ml_mode = "none"
        score_mode = "rules_only"
        enable_policy = False
        enable_enrichment = False
        enable_brand_risk = False
    elif baseline == "lexical-only":
        ml_mode = "lexical"
        score_mode = "ml_only"
        enable_policy = False
        enable_enrichment = False
        enable_brand_risk = False
    elif baseline == "char-only":
        ml_mode = "char"
        score_mode = "ml_only"
        enable_policy = False
        enable_enrichment = False
        enable_brand_risk = False
    elif baseline == "ml-xgboost":
        ml_mode = "xgboost"
        score_mode = "ml_only"
        enable_policy = False
        enable_enrichment = False
        enable_brand_risk = False
    elif baseline == "static-fusion":
        ml_mode = "ensemble"
        score_mode = "fusion"
        static_weight = 0.5
        enable_policy = False
    elif baseline == "fusion-no-enrichment":
        ml_mode = "ensemble"
        score_mode = "fusion"
        enable_enrichment = False
        enable_context_enrichment = False
        enable_brand_risk = False
    elif baseline == "fusion-no-rl":
        ml_mode = "ensemble"
        score_mode = "fusion"
        enable_policy = False
    elif baseline == "rl-v1":
        policy = BanditPolicy("models/policy.json")
    elif baseline == "rl-v2":
        policy = ThompsonSamplingPolicy("models/policy.json", seed=0)
    else:
        raise ValueError(f"Unknown baseline: {baseline}")

    config = AnalysisConfig(
        ml_mode=ml_mode,
        lexical_model="",
        char_model="",
        policy_path="models/policy.json",
        feedback_store="models/feedback.json",
        enable_feedback=False,
        enable_enrichment=enable_enrichment,
        enable_brand_risk=enable_brand_risk,
        enable_context_enrichment=enable_context_enrichment,
        score_mode=score_mode,
        static_weight=static_weight,
        enable_policy=enable_policy,
        brand_risk=brand_cfg or BrandRiskConfig(),
    )
    return config, policy


def evaluate_baseline(
    urls: list[str],
    labels: list[int],
    baseline: str,
    ml_context: dict[str, Any] | None,
    seed: int,
    brand_cfg: BrandRiskConfig | None = None,
    enable_enrichment: bool = True,
    enable_context_enrichment: bool = True,
) -> BenchmarkRun:
    config, policy = build_config(
        baseline,
        brand_cfg=brand_cfg,
        enable_enrichment=enable_enrichment,
        enable_context_enrichment=enable_context_enrichment,
    )

    if isinstance(policy, ThompsonSamplingPolicy):
        policy = ThompsonSamplingPolicy("models/policy.json", seed=seed)

    y_scores: list[float] = []
    y_pred: list[int] = []
    latencies: list[float] = []
    brand_mask: list[bool] = []
    review_count = 0

    for url in urls:
        start = time.perf_counter()
        report, _extra = analyze_url(url, config, ml_context=ml_context, policy=policy)
        latency_ms = (time.perf_counter() - start) * 1000
        latencies.append(latency_ms)
        score = report["summary"]["score"]
        if report["summary"].get("review"):
            review_count += 1
        y_scores.append(score / 100.0)
        y_pred.append(1 if binary_label_for_score(score) == "phish" else 0)
        brand_mask.append(detect_typosquatting(parse_url(url)) is not None)

    auroc = float(roc_auc_score(labels, y_scores)) if len(set(labels)) > 1 else 0.0
    auprc = float(average_precision_score(labels, y_scores)) if len(set(labels)) > 1 else 0.0
    f1 = float(f1_score(labels, y_pred, zero_division=0))
    ece = float(expected_calibration_error(labels, y_scores))
    mce = float(max_calibration_error(labels, y_scores))
    brier = float(brier_score_loss(labels, y_scores))
    latency_p50 = float(np.percentile(latencies, 50)) if latencies else 0.0
    latency_p95 = float(np.percentile(latencies, 95)) if latencies else 0.0
    positive_rate = sum(labels) / max(len(labels), 1)
    review_rate = review_count / max(len(labels), 1)

    brand_labels = [label for label, mask in zip(labels, brand_mask) if mask]
    brand_scores = [score for score, mask in zip(y_scores, brand_mask) if mask]
    brand_preds = [pred for pred, mask in zip(y_pred, brand_mask) if mask]
    if brand_labels:
        brand_auroc = float(roc_auc_score(brand_labels, brand_scores)) if len(set(brand_labels)) > 1 else 0.0
        brand_auprc = (
            float(average_precision_score(brand_labels, brand_scores))
            if len(set(brand_labels)) > 1
            else 0.0
        )
        brand_f1 = float(f1_score(brand_labels, brand_preds, zero_division=0))
    else:
        brand_auroc = None
        brand_auprc = None
        brand_f1 = None

    return BenchmarkRun(
        split="",
        baseline=baseline,
        seed=seed,
        auroc=auroc,
        auprc=auprc,
        f1=f1,
        ece=ece,
        mce=mce,
        brier=brier,
        latency_p50_ms=latency_p50,
        latency_p95_ms=latency_p95,
        count=len(labels),
        positive_rate=positive_rate,
        review_rate=review_rate,
        brand_count=len(brand_labels),
        brand_auroc=brand_auroc,
        brand_auprc=brand_auprc,
        brand_f1=brand_f1,
    )


def evaluate_baseline_scores(
    urls: list[str],
    labels: list[int],
    baseline: str,
    ml_context: dict[str, Any] | None,
    seed: int,
    enable_enrichment: bool = True,
    enable_context_enrichment: bool = True,
) -> tuple[list[float], list[int]]:
    config, policy = build_config(
        baseline,
        enable_enrichment=enable_enrichment,
        enable_context_enrichment=enable_context_enrichment,
    )
    if isinstance(policy, ThompsonSamplingPolicy):
        policy = ThompsonSamplingPolicy("models/policy.json", seed=seed)

    y_scores: list[float] = []
    y_pred: list[int] = []
    for url in urls:
        report, _extra = analyze_url(url, config, ml_context=ml_context, policy=policy)
        score = report["summary"]["score"]
        y_scores.append(score / 100.0)
        y_pred.append(1 if binary_label_for_score(score) == "phish" else 0)
    return y_scores, y_pred


def select_ml_context(baseline: str, ml_contexts: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    if baseline == "rules-only":
        return None
    if baseline == "lexical-only":
        return ml_contexts.get("lexical")
    if baseline == "char-only":
        return ml_contexts.get("char")
    if baseline == "ml-xgboost":
        return ml_contexts.get("xgboost")
    return ml_contexts.get("ensemble")


def _metric_auroc(y_true: list[int], scores: list[float]) -> float:
    return float(roc_auc_score(y_true, scores)) if len(set(y_true)) > 1 else 0.0


def _metric_auprc(y_true: list[int], scores: list[float]) -> float:
    return float(average_precision_score(y_true, scores)) if len(set(y_true)) > 1 else 0.0


def _metric_f1(y_true: list[int], preds: list[int]) -> float:
    return float(f1_score(y_true, preds, zero_division=0))


def evaluate_error_buckets(
    urls: list[str],
    labels: list[int],
    baseline: str,
    ml_context: dict[str, Any] | None,
    seed: int,
    enable_enrichment: bool = True,
    enable_context_enrichment: bool = True,
) -> dict[str, Any]:
    config, policy = build_config(
        baseline,
        enable_enrichment=enable_enrichment,
        enable_context_enrichment=enable_context_enrichment,
    )
    if isinstance(policy, ThompsonSamplingPolicy):
        policy = ThompsonSamplingPolicy("models/policy.json", seed=seed)

    false_positive: dict[str, int] = {}
    false_negative: dict[str, int] = {}
    counts: dict[str, int] = {"fp": 0, "fn": 0, "total": len(labels)}
    buckets: dict[str, Any] = {
        "false_positive": false_positive,
        "false_negative": false_negative,
        "counts": counts,
    }

    for url, label in zip(urls, labels):
        report, extra = analyze_url(url, config, ml_context=ml_context, policy=policy)
        predicted = binary_label_for_score(int(report["summary"]["score"]))
        true_label = "phish" if label == 1 else "legit"

        if predicted == true_label:
            continue

        signals = {hit.get("name") for hit in extra.get("signals", [])}
        brand_risk = float(extra.get("brand_typo_risk", 0.0))
        age_days = extra.get("domain_enrichment", {}).get("age_days")

        brand_typo = "typosquat" in signals or brand_risk >= 45
        benign_lookalike = "typosquat" in signals or brand_risk >= 30
        obfuscation = bool(
            signals.intersection({"encoded_chars", "at_symbol", "ip_host", "uncommon_port", "shortener"})
        )
        new_domain = age_days is not None and age_days <= 30

        if true_label == "legit":
            counts["fp"] += 1
            _bucket = false_positive
            if benign_lookalike:
                _bucket["benign_lookalike"] = _bucket.get("benign_lookalike", 0) + 1
            if obfuscation:
                _bucket["obfuscation"] = _bucket.get("obfuscation", 0) + 1
            if brand_typo:
                _bucket["brand_typo"] = _bucket.get("brand_typo", 0) + 1
        else:
            counts["fn"] += 1
            _bucket = false_negative
            if brand_typo:
                _bucket["brand_typo"] = _bucket.get("brand_typo", 0) + 1
            if obfuscation:
                _bucket["obfuscation"] = _bucket.get("obfuscation", 0) + 1
            if new_domain:
                _bucket["new_domain_abuse"] = _bucket.get("new_domain_abuse", 0) + 1

    return buckets


def bootstrap_diff(
    y_true: list[int],
    a: list[float],
    b: list[float],
    metric_fn: Callable[[list[int], list[float]], float],
    iters: int,
    seed: int,
) -> dict[str, float]:
    rng = random.Random(seed)
    n = len(y_true)
    if n == 0:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0}

    diffs: list[float] = []
    for _ in range(iters):
        indices = [rng.randrange(n) for _ in range(n)]
        sample_true = [y_true[i] for i in indices]
        sample_a = [a[i] for i in indices]
        sample_b = [b[i] for i in indices]
        diffs.append(metric_fn(sample_true, sample_a) - metric_fn(sample_true, sample_b))

    diffs.sort()
    mean = sum(diffs) / max(len(diffs), 1)
    low_idx = int(0.025 * len(diffs))
    high_idx = int(0.975 * len(diffs))
    return {"mean": mean, "ci_low": diffs[low_idx], "ci_high": diffs[min(high_idx, len(diffs) - 1)]}


def permutation_test(
    y_true: list[int],
    a: list[float],
    b: list[float],
    metric_fn: Callable[[list[int], list[float]], float],
    iters: int,
    seed: int,
) -> float:
    rng = random.Random(seed)
    observed = metric_fn(y_true, a) - metric_fn(y_true, b)
    count = 0
    for _ in range(iters):
        swapped_a: list[float] = []
        swapped_b: list[float] = []
        for ai, bi in zip(a, b):
            if rng.random() < 0.5:
                swapped_a.append(bi)
                swapped_b.append(ai)
            else:
                swapped_a.append(ai)
                swapped_b.append(bi)
        diff = metric_fn(y_true, swapped_a) - metric_fn(y_true, swapped_b)
        if abs(diff) >= abs(observed):
            count += 1
    return (count + 1) / (iters + 1)


def summarize_runs(runs: list[BenchmarkRun]) -> dict[str, Any]:
    def _stats(values: list[float]) -> dict[str, float]:
        if not values:
            return {"mean": 0.0, "std": 0.0, "ci95": 0.0}
        mean = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        ci95 = 1.96 * std / math.sqrt(len(values)) if len(values) > 1 else 0.0
        return {"mean": mean, "std": std, "ci95": ci95}

    metrics = {
        "auroc": _stats([run.auroc for run in runs]),
        "auprc": _stats([run.auprc for run in runs]),
        "f1": _stats([run.f1 for run in runs]),
        "ece": _stats([run.ece for run in runs]),
        "mce": _stats([run.mce for run in runs]),
        "brier": _stats([run.brier for run in runs]),
        "latency_p50_ms": _stats([run.latency_p50_ms for run in runs]),
        "latency_p95_ms": _stats([run.latency_p95_ms for run in runs]),
        "review_rate": _stats([run.review_rate for run in runs]),
    }
    return metrics


def write_runs_csv(path: Path, runs: list[BenchmarkRun]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "split",
                "baseline",
                "seed",
                "auroc",
                "auprc",
                "f1",
                "ece",
                "mce",
                "brier",
                "latency_p50_ms",
                "latency_p95_ms",
                "count",
                "positive_rate",
                "review_rate",
                "brand_count",
                "brand_auroc",
                "brand_auprc",
                "brand_f1",
            ],
        )
        writer.writeheader()
        for run in runs:
            writer.writerow(
                {
                    "split": run.split,
                    "baseline": run.baseline,
                    "seed": run.seed,
                    "auroc": f"{run.auroc:.4f}",
                    "auprc": f"{run.auprc:.4f}",
                    "f1": f"{run.f1:.4f}",
                    "ece": f"{run.ece:.4f}",
                    "mce": f"{run.mce:.4f}",
                    "brier": f"{run.brier:.4f}",
                    "latency_p50_ms": f"{run.latency_p50_ms:.2f}",
                    "latency_p95_ms": f"{run.latency_p95_ms:.2f}",
                    "count": run.count,
                    "positive_rate": f"{run.positive_rate:.4f}",
                    "review_rate": f"{run.review_rate:.4f}",
                    "brand_count": run.brand_count,
                    "brand_auroc": "" if run.brand_auroc is None else f"{run.brand_auroc:.4f}",
                    "brand_auprc": "" if run.brand_auprc is None else f"{run.brand_auprc:.4f}",
                    "brand_f1": "" if run.brand_f1 is None else f"{run.brand_f1:.4f}",
                }
            )


def plot_summary(summary: dict[str, Any], output_dir: Path, split: str) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore[import-not-found]
        import seaborn as sns  # type: ignore[import-not-found]
    except ImportError:
        return

    plt = cast(Any, plt)
    sns = cast(Any, sns)

    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    baselines = list(summary.keys())
    auroc_means = [summary[name]["auroc"]["mean"] for name in baselines]
    f1_means = [summary[name]["f1"]["mean"] for name in baselines]

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(x=baselines, y=auroc_means, ax=ax)
    ax.set_title(f"AUROC ({split})")
    ax.set_ylabel("AUROC")
    ax.set_xlabel("Baseline")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(str(output_dir / f"auroc_{split}.png"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(x=baselines, y=f1_means, ax=ax)
    ax.set_title(f"F1 ({split})")
    ax.set_ylabel("F1")
    ax.set_xlabel("Baseline")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(str(output_dir / f"f1_{split}.png"))
    plt.close(fig)


def _lexical_feature_rows(urls: list[str]) -> list[list[float]]:
    suspicious_tlds = load_suspicious_tlds()
    rows: list[list[float]] = []
    for url in urls:
        parsed = parse_url(url)
        features = extract_features(parsed, suspicious_tlds)
        rows.append(vectorize_features(features))
    return rows


def _predict_lexical_proba(pipeline: Pipeline, urls: list[str]) -> list[float]:
    features = _lexical_feature_rows(urls)
    proba = cast(Any, pipeline).predict_proba(features)
    return [float(row[1]) for row in proba]


def _predict_char_proba(pipeline: Pipeline, urls: list[str]) -> list[float]:
    proba = cast(Any, pipeline).predict_proba(urls)
    return [float(row[1]) for row in proba]


def _calibration_metrics(y_true: list[int], y_prob: list[float]) -> dict[str, float]:
    auroc = float(roc_auc_score(y_true, y_prob)) if len(set(y_true)) > 1 else 0.0
    auprc = float(average_precision_score(y_true, y_prob)) if len(set(y_true)) > 1 else 0.0
    ece = float(expected_calibration_error(y_true, y_prob))
    mce = float(max_calibration_error(y_true, y_prob))
    brier = float(brier_score_loss(y_true, y_prob))
    return {"auroc": auroc, "auprc": auprc, "ece": ece, "mce": mce, "brier": brier}


def evaluate_calibration_compare(
    train_urls: list[str],
    train_labels: list[int],
    test_urls: list[str],
    test_labels: list[int],
    seed: int,
) -> dict[str, Any]:
    results: dict[str, Any] = {"lexical": {}, "char": {}}

    lexical_sigmoid = train_lexical_pipeline(train_urls, train_labels, seed, method="sigmoid")
    lexical_isotonic = train_lexical_pipeline(train_urls, train_labels, seed, method="isotonic")
    char_sigmoid = train_char_pipeline(train_urls, train_labels, seed, method="sigmoid")
    char_isotonic = train_char_pipeline(train_urls, train_labels, seed, method="isotonic")

    results["lexical"]["sigmoid"] = _calibration_metrics(
        test_labels, _predict_lexical_proba(lexical_sigmoid, test_urls)
    )
    results["lexical"]["isotonic"] = _calibration_metrics(
        test_labels, _predict_lexical_proba(lexical_isotonic, test_urls)
    )
    results["char"]["sigmoid"] = _calibration_metrics(
        test_labels, _predict_char_proba(char_sigmoid, test_urls)
    )
    results["char"]["isotonic"] = _calibration_metrics(
        test_labels, _predict_char_proba(char_isotonic, test_urls)
    )

    return results


def evaluate_reliability(
    urls: list[str],
    labels: list[int],
    ml_context: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    config, policy = build_config("rl-v2")
    policy = ThompsonSamplingPolicy("models/policy.json", seed=seed)

    scores: list[float] = []
    for url in urls:
        report, _extra = analyze_url(url, config, ml_context=ml_context, policy=policy)
        scores.append(report["summary"]["score"] / 100.0)

    return {
        "ece": float(expected_calibration_error(labels, scores)),
        "mce": float(max_calibration_error(labels, scores)),
        "bins": reliability_bins(labels, scores),
        "abstain": {
            "min_confidence": config.abstain_min_confidence,
            "min_score": config.abstain_min_score,
        },
    }


def evaluate_review_impact(
    urls: list[str],
    labels: list[int],
    baseline: str,
    ml_context: dict[str, Any] | None,
    seed: int,
    enable_enrichment: bool = True,
    enable_context_enrichment: bool = True,
) -> dict[str, Any]:
    config, policy = build_config(
        baseline,
        enable_enrichment=enable_enrichment,
        enable_context_enrichment=enable_context_enrichment,
    )
    if isinstance(policy, ThompsonSamplingPolicy):
        policy = ThompsonSamplingPolicy("models/policy.json", seed=seed)

    auto_labels: list[int] = []
    auto_preds: list[int] = []
    review_count = 0

    for url, label in zip(urls, labels):
        report, _extra = analyze_url(url, config, ml_context=ml_context, policy=policy)
        if report["summary"].get("review"):
            review_count += 1
            continue
        predicted = 1 if binary_label_for_score(int(report["summary"]["score"])) == "phish" else 0
        auto_labels.append(label)
        auto_preds.append(predicted)

    coverage = len(auto_labels) / max(len(labels), 1)
    precision = float(precision_score(auto_labels, auto_preds, zero_division=0)) if auto_labels else 0.0
    recall = float(recall_score(auto_labels, auto_preds, zero_division=0)) if auto_labels else 0.0
    f1 = float(f1_score(auto_labels, auto_preds, zero_division=0)) if auto_labels else 0.0

    return {
        "auto_precision": precision,
        "auto_recall": recall,
        "auto_f1": f1,
        "auto_coverage": coverage,
        "review_rate": review_count / max(len(labels), 1),
        "review_count": review_count,
        "total": len(labels),
        "abstain": {
            "min_confidence": config.abstain_min_confidence,
            "min_score": config.abstain_min_score,
        },
    }


def _resolved_entries(path: Path) -> list[FeedbackEntry]:
    entries: list[FeedbackEntry] = []
    for entry in load_entries(str(path)):
        if entry.status == "resolved" and entry.true_label:
            entries.append(entry)
    return entries


def _poison_entries(entries: list[FeedbackEntry], rate: float, rng: random.Random) -> list[FeedbackEntry]:
    if rate <= 0:
        return entries
    poisoned: list[FeedbackEntry] = []
    for entry in entries:
        if rng.random() < rate:
            flipped = "phish" if entry.true_label == "legit" else "legit"
            poisoned.append(
                FeedbackEntry(
                    id=entry.id,
                    url=entry.url,
                    predicted_label=entry.predicted_label,
                    confidence=entry.confidence,
                    context=entry.context,
                    action=entry.action,
                    rule_score=entry.rule_score,
                    ml_score=entry.ml_score,
                    final_score=entry.final_score,
                    propensity=entry.propensity,
                    policy_strategy=entry.policy_strategy,
                    status=entry.status,
                    created_at=entry.created_at,
                    resolved_at=entry.resolved_at,
                    true_label=flipped,
                )
            )
        else:
            poisoned.append(entry)
    return poisoned


def evaluate_policy_stability(
    feedback_path: str,
    policy: Any,
    seeds: list[int],
    sparse_rates: list[float],
    poison_rate: float,
    fn_cost: float,
    fp_cost: float,
    min_policy_confidence: float,
    max_weight_shift: float,
) -> dict[str, Any]:
    entries = _resolved_entries(Path(feedback_path))
    if not entries:
        return {}

    summary: dict[str, Any] = {}
    for rate in sparse_rates:
        runs: list[OffPolicyResult] = []
        for seed in seeds:
            rng = random.Random(seed)
            if rate >= 1.0:
                sampled = list(entries)
            else:
                sampled = [entry for entry in entries if rng.random() <= rate]
            sampled = _poison_entries(sampled, poison_rate, rng)
            runs.append(
                evaluate_offpolicy_entries(
                    sampled,
                    policy,
                    fn_cost,
                    fp_cost,
                    min_policy_confidence,
                    max_weight_shift,
                )
            )

        if not runs:
            continue
        guardrail_rates = [run.guardrail_violations / max(run.count, 1) for run in runs]
        summary[f"rate_{rate:.2f}"] = {
            "ips": statistics.mean([run.ips for run in runs]),
            "snips": statistics.mean([run.snips for run in runs]),
            "dr": statistics.mean([run.dr for run in runs]),
            "guardrail_violation_rate": statistics.mean(guardrail_rates),
            "count": statistics.mean([run.count for run in runs]),
            "poison_rate": poison_rate,
        }

    return summary


def evaluate_calibration_drift(
    rows: list[dict[str, Any]],
    baseline: str,
    ml_context: dict[str, Any] | None,
    seed: int,
    buckets: int = 4,
) -> dict[str, Any]:
    timed = [row for row in rows if row.get("time") is not None]
    if len(timed) < buckets:
        return {}
    timed.sort(key=lambda row: row["time"])

    config, policy = build_config(baseline)
    if isinstance(policy, ThompsonSamplingPolicy):
        policy = ThompsonSamplingPolicy("models/policy.json", seed=seed)

    bucket_size = max(1, len(timed) // buckets)
    results: list[dict[str, Any]] = []
    for idx in range(buckets):
        start = idx * bucket_size
        end = len(timed) if idx == buckets - 1 else (idx + 1) * bucket_size
        bucket_rows = timed[start:end]
        labels = [row["label"] for row in bucket_rows]
        scores: list[float] = []
        review_count = 0
        for row in bucket_rows:
            report, _extra = analyze_url(row["url"], config, ml_context=ml_context, policy=policy)
            scores.append(report["summary"]["score"] / 100.0)
            if report["summary"].get("review"):
                review_count += 1
        results.append(
            {
                "start_time": bucket_rows[0]["time"],
                "end_time": bucket_rows[-1]["time"],
                "count": len(bucket_rows),
                "ece": float(expected_calibration_error(labels, scores)),
                "mce": float(max_calibration_error(labels, scores)),
                "review_rate": review_count / max(len(bucket_rows), 1),
                "abstain": {
                    "min_confidence": config.abstain_min_confidence,
                    "min_score": config.abstain_min_score,
                },
            }
        )
    return {"buckets": results}


def evaluate_adversarial_suite(
    urls: list[str],
    labels: list[int],
    ml_context: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    config, policy = build_config("rl-v2")
    policy = ThompsonSamplingPolicy("models/policy.json", seed=seed)

    stats: dict[str, dict[str, float]] = {}
    for url, label in zip(urls, labels):
        report, _extra = analyze_url(url, config, ml_context=ml_context, policy=policy)
        base_score = float(report["summary"]["score"])
        if label != 1:
            continue
        for perturbation in generate_perturbations(url):
            report_adv, _extra_adv = analyze_url(
                perturbation.url,
                config,
                ml_context=ml_context,
                policy=policy,
            )
            adv_score = float(report_adv["summary"]["score"])
            adv_label = binary_label_for_score(int(round(adv_score)))
            entry = stats.setdefault(perturbation.name, {"count": 0.0, "success": 0.0, "delta_sum": 0.0})
            entry["count"] += 1.0
            if adv_label == "legit":
                entry["success"] += 1.0
            entry["delta_sum"] += (base_score - adv_score)

    summary: dict[str, Any] = {}
    for name, entry in stats.items():
        count = entry["count"]
        summary[name] = {
            "count": int(count),
            "attack_success_rate": float(entry["success"] / count) if count else 0.0,
            "avg_score_delta": float(entry["delta_sum"] / count) if count else 0.0,
        }
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Gojo benchmarking runner")
    parser.add_argument("--data", required=True, help="Path to CSV dataset")
    parser.add_argument("--url-col", default="url", help="URL column name")
    parser.add_argument("--label-col", default="label", help="Label column name")
    parser.add_argument("--time-col", default=None, help="Optional time column for time split")
    parser.add_argument("--test-ratio", type=float, default=0.2, help="Test split ratio")
    parser.add_argument("--seeds", type=int, default=5, help="Number of seeds")
    parser.add_argument("--seed", type=int, default=42, help="Base seed")
    parser.add_argument("--output-dir", default="results", help="Output directory")
    parser.add_argument("--plots", action="store_true", help="Generate plots")
    parser.add_argument("--max-rows", type=int, default=0, help="Limit dataset rows for faster runs")
    parser.add_argument("--feedback-store", default="models/feedback.json", help="Feedback log path")
    parser.add_argument("--skip-calibration-compare", action="store_true", help="Skip calibration method compare")
    parser.add_argument("--skip-adversarial", action="store_true", help="Skip adversarial suite")
    parser.add_argument("--skip-offpolicy", action="store_true", help="Skip off-policy evaluation")
    parser.add_argument("--skip-significance", action="store_true", help="Skip significance tests")
    parser.add_argument("--significance-iters", type=int, default=500, help="Bootstrap/permutation iterations")
    parser.add_argument("--ood-data", help="OOD dataset CSV path")
    parser.add_argument("--ood-url-col", default="url", help="OOD URL column name")
    parser.add_argument("--ood-label-col", default="label", help="OOD label column name")
    parser.add_argument("--ood-time-col", default=None, help="OOD time column name")
    parser.add_argument("--ood-time-cutoff", default=None, help="Use rows >= cutoff as OOD")
    parser.add_argument("--ood-limit", type=int, default=0, help="Limit OOD rows for faster runs")
    parser.add_argument("--stability-sparse", default="1.0,0.2,0.05", help="Sparse rates CSV")
    parser.add_argument("--stability-poison", type=float, default=0.0, help="Poison rate for feedback")
    args = parser.parse_args(argv)

    rows = load_dataset(Path(args.data), args.url_col, args.label_col, args.time_col)
    if args.max_rows and len(rows) > args.max_rows:
        rng = random.Random(args.seed)
        rows = rng.sample(rows, args.max_rows)
    output_dir = Path(args.output_dir)

    baselines = [
        "rules-only",
        "lexical-only",
        "char-only",
        "static-fusion",
        "fusion-no-enrichment",
        "fusion-no-rl",
        "rl-v1",
        "rl-v2",
    ]

    ablations: dict[str, BrandRiskConfig | None] = {
        "no-typo": BrandRiskConfig(enable_typo=False),
        "no-intent": BrandRiskConfig(enable_intent=False),
        "no-age": BrandRiskConfig(enable_age=False),
        "no-reputation": BrandRiskConfig(enable_reputation=False),
        "no-volatility": BrandRiskConfig(enable_volatility=False),
        "no-context": None,
    }

    all_runs: list[BenchmarkRun] = []
    summary: dict[str, Any] = {
        "time": {},
        "random": {},
        "ablations": {"time": {}, "random": {}},
        "calibration_compare": {},
        "reliability": {},
        "calibration_drift": {},
        "adversarial": {},
        "significance": {},
        "error_buckets": {},
        "review_impact": {},
        "ood": {},
        "offpolicy_stability": {},
        "dataset": {
            "path": args.data,
            "sha256": _sha256_file(Path(args.data)),
            "rows": len(rows),
            "max_rows": args.max_rows,
            "seed": args.seed,
            "test_ratio": args.test_ratio,
        },
    }

    split_eval_data: dict[str, dict[str, Any]] = {}

    for split_name in ["time", "random"]:
        split_summary: dict[str, Any] = {}
        ablation_summary: dict[str, Any] = {}
        for seed_offset in range(args.seeds):
            seed = args.seed + seed_offset
            random.seed(seed)
            np.random.seed(seed)

            if split_name == "time":
                train_rows, test_rows = time_split(rows, args.test_ratio)
            else:
                train_rows, test_rows = random_split(rows, args.test_ratio, seed)

            train_urls = [row["url"] for row in train_rows]
            train_labels = [row["label"] for row in train_rows]
            test_urls = [row["url"] for row in test_rows]
            test_labels = [row["label"] for row in test_rows]

            lexical_model = train_lexical_pipeline(train_urls, train_labels, seed)
            char_model = train_char_pipeline(train_urls, train_labels, seed)

            ml_contexts: dict[str, dict[str, Any]] = {
                "lexical": {"mode": "lexical", "model": lexical_model, "available": True},
                "char": {"mode": "char", "model": char_model, "available": True},
                "ensemble": {
                    "mode": "ensemble",
                    "lexical_model": lexical_model,
                    "char_model": char_model,
                    "available": True,
                },
            }

            for baseline in baselines:
                ml_context = None
                if baseline in {"lexical-only"}:
                    ml_context = ml_contexts["lexical"]
                elif baseline in {"char-only"}:
                    ml_context = ml_contexts["char"]
                elif baseline in {"static-fusion", "fusion-no-enrichment", "fusion-no-rl", "rl-v1", "rl-v2"}:
                    ml_context = ml_contexts["ensemble"]

                run = evaluate_baseline(
                    test_urls,
                    test_labels,
                    baseline,
                    ml_context,
                    seed,
                    enable_enrichment=baseline != "fusion-no-enrichment",
                    enable_context_enrichment=baseline not in {"fusion-no-enrichment", "fusion-no-rl"},
                )
                run = replace(run, split=split_name)
                all_runs.append(run)

            for name, cfg in ablations.items():
                enable_context = True
                if name == "no-context":
                    enable_context = False
                run = evaluate_baseline(
                    test_urls,
                    test_labels,
                    "rl-v2",
                    ml_contexts["ensemble"],
                    seed,
                    brand_cfg=cfg,
                    enable_enrichment=True,
                    enable_context_enrichment=enable_context,
                )
                run = replace(run, split=split_name, baseline=f"ablation-{name}")
                all_runs.append(run)

            if seed_offset == 0:
                split_eval_data[split_name] = {
                    "test_urls": test_urls,
                    "test_labels": test_labels,
                    "ml_contexts": ml_contexts,
                    "seed": seed,
                }

            if seed_offset == 0 and not args.skip_calibration_compare:
                summary["calibration_compare"][split_name] = evaluate_calibration_compare(
                    train_urls,
                    train_labels,
                    test_urls,
                    test_labels,
                    seed,
                )

        for baseline in baselines:
            runs = [run for run in all_runs if run.split == split_name and run.baseline == baseline]
            split_summary[baseline] = summarize_runs(runs)
        for name in ablations:
            runs = [run for run in all_runs if run.split == split_name and run.baseline == f"ablation-{name}"]
            ablation_summary[name] = summarize_runs(runs)

        summary[split_name] = split_summary
        summary["ablations"][split_name] = ablation_summary

        if split_name in split_eval_data:
            eval_data = split_eval_data[split_name]
            test_urls = eval_data["test_urls"]
            test_labels = eval_data["test_labels"]
            ml_contexts = eval_data["ml_contexts"]
            if not args.skip_adversarial:
                summary["adversarial"][split_name] = evaluate_adversarial_suite(
                    test_urls,
                    test_labels,
                    ml_contexts["ensemble"],
                    args.seed,
                )
            summary["reliability"][split_name] = evaluate_reliability(
                test_urls,
                test_labels,
                ml_contexts["ensemble"],
                args.seed,
            )
            summary["error_buckets"][split_name] = {
                "static-fusion": evaluate_error_buckets(
                    test_urls,
                    test_labels,
                    "static-fusion",
                    ml_contexts["ensemble"],
                    args.seed,
                ),
                "rl-v2": evaluate_error_buckets(
                    test_urls,
                    test_labels,
                    "rl-v2",
                    ml_contexts["ensemble"],
                    args.seed,
                ),
            }
            summary["review_impact"][split_name] = {
                "static-fusion": evaluate_review_impact(
                    test_urls,
                    test_labels,
                    "static-fusion",
                    ml_contexts["ensemble"],
                    args.seed,
                ),
                "rl-v2": evaluate_review_impact(
                    test_urls,
                    test_labels,
                    "rl-v2",
                    ml_contexts["ensemble"],
                    args.seed,
                ),
            }

        if args.plots:
            plot_summary(split_summary, output_dir / "plots", split_name)

        if not args.skip_significance and split_name in split_eval_data:
            eval_data = split_eval_data[split_name]
            test_urls = eval_data["test_urls"]
            test_labels = eval_data["test_labels"]
            ml_contexts = eval_data["ml_contexts"]
            seed = eval_data["seed"]

            base_scores, base_preds = evaluate_baseline_scores(
                test_urls,
                test_labels,
                "static-fusion",
                ml_contexts["ensemble"],
                seed,
            )
            target_scores, target_preds = evaluate_baseline_scores(
                test_urls,
                test_labels,
                "rl-v2",
                ml_contexts["ensemble"],
                seed,
            )

            baseline_scores_cache: dict[str, tuple[list[float], list[int]]] = {
                "static-fusion": (base_scores, base_preds),
                "rl-v2": (target_scores, target_preds),
            }

            baseline_deltas: dict[str, Any] = {}
            for baseline in baselines:
                if baseline == "static-fusion":
                    continue
                if baseline in baseline_scores_cache:
                    scores, preds = baseline_scores_cache[baseline]
                else:
                    baseline_context = select_ml_context(baseline, ml_contexts)
                    scores, preds = evaluate_baseline_scores(
                        test_urls,
                        test_labels,
                        baseline,
                        baseline_context,
                        seed,
                        enable_enrichment=baseline != "fusion-no-enrichment",
                        enable_context_enrichment=baseline not in {"fusion-no-enrichment", "fusion-no-rl"},
                    )
                    baseline_scores_cache[baseline] = (scores, preds)

                baseline_deltas[baseline] = {
                    "auroc": {
                        "bootstrap": bootstrap_diff(
                            test_labels,
                            scores,
                            base_scores,
                            _metric_auroc,
                            args.significance_iters,
                            args.seed,
                        ),
                        "p_value": permutation_test(
                            test_labels,
                            scores,
                            base_scores,
                            _metric_auroc,
                            args.significance_iters,
                            args.seed,
                        ),
                    },
                    "auprc": {
                        "bootstrap": bootstrap_diff(
                            test_labels,
                            scores,
                            base_scores,
                            _metric_auprc,
                            args.significance_iters,
                            args.seed,
                        ),
                        "p_value": permutation_test(
                            test_labels,
                            scores,
                            base_scores,
                            _metric_auprc,
                            args.significance_iters,
                            args.seed,
                        ),
                    },
                    "f1": {
                        "bootstrap": bootstrap_diff(
                            test_labels,
                            cast(list[float], preds),
                            cast(list[float], base_preds),
                            cast(Callable[[list[int], list[float]], float], _metric_f1),
                            args.significance_iters,
                            args.seed,
                        ),
                        "p_value": permutation_test(
                            test_labels,
                            cast(list[float], preds),
                            cast(list[float], base_preds),
                            cast(Callable[[list[int], list[float]], float], _metric_f1),
                            args.significance_iters,
                            args.seed,
                        ),
                    },
                }

            summary["significance"][split_name] = {
                "baseline": "static-fusion",
                "deltas": baseline_deltas,
            }

    write_runs_csv(output_dir / "benchmark_runs.csv", all_runs)
    if not args.skip_offpolicy and Path(args.feedback_store).exists():
        summary["offpolicy"] = {
            "rl-v1": evaluate_offpolicy(
                args.feedback_store,
                BanditPolicy("models/policy.json"),
                fn_cost=3.0,
                fp_cost=1.0,
                min_policy_confidence=0.55,
                max_weight_shift=0.25,
            ).__dict__,
            "rl-v2": evaluate_offpolicy(
                args.feedback_store,
                ThompsonSamplingPolicy("models/policy.json", seed=args.seed),
                fn_cost=3.0,
                fp_cost=1.0,
                min_policy_confidence=0.55,
                max_weight_shift=0.25,
            ).__dict__,
        }
        for entry in summary["offpolicy"].values():
            count = entry.get("count", 0) or 0
            violations = entry.get("guardrail_violations", 0) or 0
            entry["guardrail_violation_rate"] = violations / max(count, 1)

        sparse_rates = [float(value) for value in args.stability_sparse.split(",") if value.strip()]
        seeds = [args.seed + offset for offset in range(args.seeds)]
        summary["offpolicy_stability"] = {
            "rl-v1": evaluate_policy_stability(
                args.feedback_store,
                BanditPolicy("models/policy.json"),
                seeds,
                sparse_rates,
                args.stability_poison,
                fn_cost=3.0,
                fp_cost=1.0,
                min_policy_confidence=0.55,
                max_weight_shift=0.25,
            ),
            "rl-v2": evaluate_policy_stability(
                args.feedback_store,
                ThompsonSamplingPolicy("models/policy.json", seed=args.seed),
                seeds,
                sparse_rates,
                args.stability_poison,
                fn_cost=3.0,
                fp_cost=1.0,
                min_policy_confidence=0.55,
                max_weight_shift=0.25,
            ),
        }

    if args.ood_data or args.ood_time_cutoff:
        if args.ood_data:
            ood_rows = load_dataset(
                Path(args.ood_data),
                args.ood_url_col,
                args.ood_label_col,
                args.ood_time_col,
            )
            ood_meta: dict[str, Any] = {
                "path": args.ood_data,
                "sha256": _sha256_file(Path(args.ood_data)),
            }
        else:
            cutoff = _parse_time(args.ood_time_cutoff or "")
            ood_rows = [row for row in rows if cutoff is not None and row.get("time") and row["time"] >= cutoff]
            ood_meta = {
                "time_cutoff": args.ood_time_cutoff,
                "count": len(ood_rows),
            }

        if args.ood_limit and len(ood_rows) > args.ood_limit:
            rng = random.Random(args.seed)
            ood_rows = rng.sample(ood_rows, args.ood_limit)
            ood_meta["ood_limit"] = args.ood_limit

        ood_urls = [row["url"] for row in ood_rows]
        ood_labels = [row["label"] for row in ood_rows]

        model_source = "time" if "time" in split_eval_data else "random"
        ood_contexts = split_eval_data.get(model_source, {}).get("ml_contexts")

        if ood_urls and ood_contexts:
            summary["ood"] = {
                "meta": ood_meta | {"model_source_split": model_source, "count": len(ood_urls)},
                "static-fusion": evaluate_baseline(
                    ood_urls,
                    ood_labels,
                    "static-fusion",
                    ood_contexts["ensemble"],
                    args.seed,
                ).__dict__,
                "rl-v2": evaluate_baseline(
                    ood_urls,
                    ood_labels,
                    "rl-v2",
                    ood_contexts["ensemble"],
                    args.seed,
                ).__dict__,
                "reliability": evaluate_reliability(
                    ood_urls,
                    ood_labels,
                    ood_contexts["ensemble"],
                    args.seed,
                ),
                "review_impact": evaluate_review_impact(
                    ood_urls,
                    ood_labels,
                    "rl-v2",
                    ood_contexts["ensemble"],
                    args.seed,
                ),
            }

    if "time" in split_eval_data:
        drift_contexts = split_eval_data["time"]["ml_contexts"]
        summary["calibration_drift"] = {
            "static-fusion": evaluate_calibration_drift(rows, "static-fusion", drift_contexts["ensemble"], args.seed),
            "rl-v2": evaluate_calibration_drift(rows, "rl-v2", drift_contexts["ensemble"], args.seed),
        }

    (output_dir / "benchmark_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Generate experiment manifest for reproducibility
    run_id = f"gojo-benchmark-{uuid.uuid4().hex[:8]}"
    cli_args_dict = vars(args)

    manifest = create_experiment_manifest(
        run_id=run_id,
        train_data_path=args.data,
        cli_args=cli_args_dict,
        ood_data_path=args.ood_data if args.ood_data else None,
        results_path=output_dir,
        runtime_seconds=time.time(),  # This should be start_time captured earlier
    )

    manifest.to_json(output_dir / f"manifest_{run_id}.json")
    print(f"Experiment manifest: {output_dir / f'manifest_{run_id}.json'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
