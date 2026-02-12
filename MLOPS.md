# Operational MLOps Evidence

## Drift Monitoring Dashboard
- Inputs: lexical features, rule scores, ML confidence, final score.
- Metrics: PSI, feature mean/variance shifts, label rate changes.
- Cadence: daily batch, weekly summary.
- Alerting: PSI > 0.2 or label rate delta > 2x baseline.

## Canary Rollout Policy
- Stage 1: shadow mode (no decision impact).
- Stage 2: 5% canary traffic, monitor key metrics and guardrails.
- Stage 3: 25% ramp if no regressions.
- Stage 4: full rollout with continued monitoring.

## Rollback Policy
- Trigger if FNR/precision degradation > 5% or IPS/SNIPS drop > 0.05.
- Revert policy file and model artifacts to last stable snapshot.
- Preserve incident logs and investigation notes.

## Evidence Artifacts
- Benchmark summaries in results/benchmark_summary.json.
- Off-policy reports (IPS/SNIPS/DR) in results/benchmark_summary.json.
- Drift snapshots (future: results/drift_report.json).
- Policy change log (POLICY_LOG.md).
