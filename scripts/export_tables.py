"""
Export benchmark results to publication-ready tables.

Generates LaTeX/CSV tables for manuscript figures:
- Main metrics table (AUROC, AUPRC, F1, calibration)
- OOD robustness table
- Significance testing results
- Off-policy stability analysis
- Ablation study table
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _fmt_metric(value: float | int, precision: int = 3) -> str:
    """Format metric value with given precision."""
    return f"{value:.{precision}f}"


def export_main_metrics_table(summary: dict[str, Any], output_path: Path) -> None:
    """Export main metrics (AUROC, AUPRC, F1) as LaTeX table."""
    baselines = ["rules-only", "lexical-only", "char-only", "static-fusion", "fusion-no-enrichment", 
                 "fusion-no-rl", "rl-v1", "rl-v2"]
    
    # Collect data from time split (should be primary)
    time_results = summary.get("time", {})
    
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Main Evaluation Metrics Across Baselines}",
        r"\label{tab:main_metrics}",
        r"\begin{tabular}{l|ccc|ccc}",
        r"\toprule",
        r"Baseline & AUROC & AUPRC & F1 & ECE & Latency (ms) & Coverage \\",
        r"\midrule",
    ]
    
    for baseline in baselines:
        if baseline not in time_results:
            continue
        metrics = time_results[baseline]
        auroc = _fmt_metric(metrics.get("auroc", {}).get("mean", 0.0))
        auprc = _fmt_metric(metrics.get("auprc", {}).get("mean", 0.0))
        f1 = _fmt_metric(metrics.get("f1", {}).get("mean", 0.0))
        ece = _fmt_metric(metrics.get("ece", {}).get("mean", 0.0))
        latency = _fmt_metric(metrics.get("latency_p50_ms", {}).get("mean", 0.0), 2)
        review_rate = _fmt_metric(metrics.get("review_rate", {}).get("mean", 0.0), 1)
        
        lines.append(f"{baseline} & {auroc} & {auprc} & {f1} & {ece} & {latency} & {review_rate} \\\\")
    
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Exported main metrics table: {output_path}")


def export_ood_robustness_table(summary: dict[str, Any], output_path: Path) -> None:
    """Export OOD robustness metrics as CSV/LaTeX table."""
    ood_results = summary.get("ood", {})
    
    if not ood_results:
        print("No OOD results found, skipping OOD table export")
        return
    
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Out-of-Distribution Performance}",
        r"\label{tab:ood_robustness}",
        r"\begin{tabular}{l|ccc|cc}",
        r"\toprule",
        r"Model & AUROC & AUPRC & F1 & ECE & Dataset \\",
        r"\midrule",
    ]
    
    dataset_path = ood_results.get("meta", {}).get("path", "OOD")
    
    for baseline in ["static-fusion", "rl-v2"]:
        if baseline not in ood_results:
            continue
        metrics = ood_results[baseline]
        auroc = _fmt_metric(metrics.get("auroc", 0.0))
        auprc = _fmt_metric(metrics.get("auprc", 0.0))
        f1 = _fmt_metric(metrics.get("f1", 0.0))
        ece = _fmt_metric(metrics.get("ece", 0.0))
        
        lines.append(f"{baseline} & {auroc} & {auprc} & {f1} & {ece} & {dataset_path} \\\\")
    
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Exported OOD robustness table: {output_path}")


def export_calibration_table(summary: dict[str, Any], output_path: Path) -> None:
    """Export calibration metrics (ECE, MCE, Brier) by method."""
    ood_reliability = summary.get("ood", {}).get("reliability", {})
    
    if not ood_reliability:
        print("No calibration results found, skipping calibration table export")
        return
    
    lines = [
        "bin,count,accuracy,confidence",
    ]
    
    for bin_info in ood_reliability.get("bins", []):
        bin_num = int(bin_info.get("bin", 0))
        count = int(bin_info.get("count", 0))
        acc = _fmt_metric(bin_info.get("acc", 0.0))
        conf = _fmt_metric(bin_info.get("conf", 0.0))
        lines.append(f"{bin_num},{count},{acc},{conf}")
    
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Exported calibration table: {output_path}")


def export_significance_table(summary: dict[str, Any], output_path: Path) -> None:
    """Export significance testing results."""
    sig_results = summary.get("significance", {})
    
    if not sig_results:
        print("No significance results found, skipping significance table export")
        return
    
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Statistical Significance Testing (vs. rules-only baseline)}",
        r"\label{tab:significance}",
        r"\begin{tabular}{l|ccc}",
        r"\toprule",
        r"Baseline & AUROC CI & p-value & Effect Size \\",
        r"\midrule",
    ]
    
    for baseline, result in sig_results.items():
        if baseline == "rules-only":
            continue
        
        ci_lower = _fmt_metric(result.get("auroc_ci", [0, 0])[0])
        ci_upper = _fmt_metric(result.get("auroc_ci", [0, 1])[1])
        pval = _fmt_metric(result.get("auroc_pvalue", 1.0), 4)
        effect = _fmt_metric(result.get("auroc_effect_size", 0.0))
        
        lines.append(f"{baseline} & [{ci_lower}, {ci_upper}] & {pval} & {effect} \\\\")
    
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Exported significance table: {output_path}")


def export_offpolicy_stability_table(summary: dict[str, Any], output_path: Path) -> None:
    """Export off-policy stability metrics."""
    offpolicy = summary.get("offpolicy_stability", {})
    
    if not offpolicy:
        print("No off-policy results found, skipping off-policy table export")
        return
    
    lines = [
        "policy,sparse_rate,ips,snips,dr,guardrail_violation_rate",
    ]
    
    for policy, rates in offpolicy.items():
        for rate_key, metrics in rates.items():
            rate = rate_key.split("_")[-1]  # Extract rate value
            ips = _fmt_metric(metrics.get("ips", 0.0))
            snips = _fmt_metric(metrics.get("snips", 0.0))
            dr = _fmt_metric(metrics.get("dr", 0.0))
            violation_rate = _fmt_metric(metrics.get("guardrail_violation_rate", 0.0))
            lines.append(f"{policy},{rate},{ips},{snips},{dr},{violation_rate}")
    
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Exported off-policy stability table: {output_path}")


def export_ablation_study_table(summary: dict[str, Any], output_path: Path) -> None:
    """Export ablation study results."""
    ablations = summary.get("ablations", {}).get("time", {})
    
    if not ablations:
        print("No ablation results found, skipping ablation table export")
        return
    
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Ablation Study: Impact of Brand Risk Features}",
        r"\label{tab:ablation}",
        r"\begin{tabular}{l|ccc}",
        r"\toprule",
        r"Configuration & AUROC & AUPRC & F1 \\",
        r"\midrule",
    ]
    
    lines.append(r"Full model & - & - & - \\")  # Placeholder; extract from baseline
    
    for ablation_name, ablation_results in ablations.items():
        if not ablation_results or not isinstance(ablation_results, dict):
            continue
        
        # Extract mean values from nested structure
        auroc_mean = ablation_results.get("auroc", {})
        auprc_mean = ablation_results.get("auprc", {})
        f1_mean = ablation_results.get("f1", {})
        
        if isinstance(auroc_mean, dict):
            auroc_mean = auroc_mean.get("mean", 0.0)
        if isinstance(auprc_mean, dict):
            auprc_mean = auprc_mean.get("mean", 0.0)
        if isinstance(f1_mean, dict):
            f1_mean = f1_mean.get("mean", 0.0)
        
        auroc = _fmt_metric(float(auroc_mean))  # type: ignore[arg-type]
        auprc = _fmt_metric(float(auprc_mean))  # type: ignore[arg-type]
        f1 = _fmt_metric(float(f1_mean))  # type: ignore[arg-type]
        
        lines.append(f"{ablation_name} & {auroc} & {auprc} & {f1} \\\\")
    
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Exported ablation study table: {output_path}")


def export_all_tables(summary_path: Path, output_dir: Path) -> None:
    """Export all publication-ready tables from benchmark summary."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
    
    # Export individual tables
    export_main_metrics_table(summary, output_dir / "table_main_metrics.tex")
    export_ood_robustness_table(summary, output_dir / "table_ood_robustness.tex")
    export_calibration_table(summary, output_dir / "table_calibration.csv")
    export_significance_table(summary, output_dir / "table_significance.tex")
    export_offpolicy_stability_table(summary, output_dir / "table_offpolicy_stability.csv")
    export_ablation_study_table(summary, output_dir / "table_ablation_study.tex")
    
    print(f"\nAll tables exported to {output_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Export benchmark results to publication-ready tables")
    parser.add_argument("--summary", default="results/benchmark_summary.json", help="Path to benchmark summary JSON")
    parser.add_argument("--output-dir", default="results/tables", help="Output directory for tables")
    args = parser.parse_args()
    
    export_all_tables(Path(args.summary), Path(args.output_dir))
