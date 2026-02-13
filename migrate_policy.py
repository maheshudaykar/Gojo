"""
Migrate policy from v1 (epsilon-greedy) to v2 (Thompson Sampling).

This script converts existing policy.json to the new v2 format while preserving
learned statistics and converting them to Beta distribution parameters.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def migrate_v1_to_v2(v1_path: str, v2_path: str) -> None:
    """
    Migrate v1 policy to v2 format.

    Args:
        v1_path: Path to existing v1 policy.json
        v2_path: Path for new v2 policy.json
    """
    v1_file = Path(v1_path)
    if not v1_file.exists():
        print(f"❌ V1 policy not found: {v1_path}")
        print("   Nothing to migrate.")
        return

    print(f"📖 Reading v1 policy from {v1_path}")
    v1_data = json.loads(v1_file.read_text(encoding="utf-8"))

    # Create v2 structure
    v2_data: dict[str, Any] = {
        "version": "policy_v2_thompson",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "migrated_from": v1_data.get("version", "policy_v1"),
        "migration_date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "strategy": "thompson",
        "actions": v1_data.get("actions", [0.8, 0.6, 0.4, 0.2]),
        "contexts": {},
        "global_stats": {
            "total_updates": 0,
            "total_reward": 0.0,
            "cumulative_regret": 0.0,
            "optimal_action_rate": 0.0,
            "avg_confidence": 0.0,
            "context_distribution": {},
            "action_distribution": {},
            "best_action_per_context": {},
            "last_evaluation": None,
        },
    }

    # Migrate context data
    v1_contexts = v1_data.get("contexts", {})
    total_conversions = 0

    for context, context_data in v1_contexts.items():
        v2_context_data: dict[str, dict[str, Any]] = {}

        for action_str, stats in context_data.items():
            if not isinstance(stats, dict):
                continue

            # Type narrowing doesn't work perfectly with dict[Unknown, Unknown]
            n = int(stats.get("n", 0))  # type: ignore[arg-type]
            value = float(stats.get("value", 0.0))  # type: ignore[arg-type]

            # Convert value (expected reward) to Beta distribution parameters
            # Assume value is in [-1, 1], normalize to [0, 1]
            normalized_value = (value + 1) / 2

            # Estimate alpha and beta based on n and normalized_value
            # alpha / (alpha + beta) ≈ normalized_value
            # alpha + beta = pseudocount (use n as proxy)
            pseudocount = max(n, 2)  # At least 2 for Beta distribution
            alpha = float(normalized_value * pseudocount + 1)  # Add prior
            beta = float((1 - normalized_value) * pseudocount + 1)

            v2_context_data[action_str] = {
                "n": n,
                "value": value,
                "alpha": alpha,
                "beta": beta,
            }

            # Update global stats
            v2_data["global_stats"]["total_updates"] += n
            v2_data["global_stats"]["context_distribution"][context] = \
                v2_data["global_stats"]["context_distribution"].get(context, 0) + n
            v2_data["global_stats"]["action_distribution"][action_str] = \
                v2_data["global_stats"]["action_distribution"].get(action_str, 0) + n

            total_conversions += 1

        v2_data["contexts"][context] = v2_context_data

    # Estimate total reward (rough approximation)
    total_updates: int = v2_data["global_stats"]["total_updates"]  # type: ignore[assignment]
    if total_updates > 0:
        # Estimate average reward from context statistics
        total_weighted_value = sum(
            sum(
                stats.get("n", 0) * stats.get("value", 0.0)  # type: ignore[arg-type]
                for stats in ctx_data.values()  # type: ignore[union-attr]
                if isinstance(stats, dict)
            )
            for ctx_data in v2_data["contexts"].values()  # type: ignore[union-attr]
        )
        v2_data["global_stats"]["total_reward"] = total_weighted_value

    # Save v2 policy
    v2_file = Path(v2_path)
    v2_file.parent.mkdir(parents=True, exist_ok=True)
    v2_file.write_text(json.dumps(v2_data, indent=2), encoding="utf-8")

    print("✅ Migration complete!")
    print(f"   Migrated {len(v1_contexts)} contexts")
    print(f"   Converted {total_conversions} action statistics")
    print(f"   Total updates preserved: {total_updates}")
    print(f"   Saved to: {v2_path}")
    print()
    print("📊 Summary:")
    print("   - Strategy: Thompson Sampling (Bayesian)")
    print(f"   - Actions: {v2_data['actions']}")
    print(f"   - Contexts: {len(v2_data['contexts'])}")

    # Create backup of v1
    backup_ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    backup_path = v1_file.parent / f"policy_v1_backup_{backup_ts}.json"
    backup_path.write_text(v1_file.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"   - V1 backup: {backup_path}")


def main() -> None:
    """Main migration script."""
    print("=" * 60)
    print(" Policy Migration: v1 (Epsilon-Greedy) → v2 (Thompson Sampling)")
    print("=" * 60)
    print()

    v1_path = "models/policy.json"
    v2_path = "models/policy_v2.json"

    if len(sys.argv) > 1:
        v1_path = sys.argv[1]
    if len(sys.argv) > 2:
        v2_path = sys.argv[2]

    migrate_v1_to_v2(v1_path, v2_path)

    print()
    print("🎯 Next Steps:")
    print("   1. Review migrated policy: models/policy_v2.json")
    print("   2. Update config to use v2:")
    print("      from phish_detector.policy_v2 import ThompsonSamplingPolicy")
    print("      policy = ThompsonSamplingPolicy('models/policy_v2.json')")
    print("   3. Or rename policy_v2.json to policy.json to auto-use v2")
    print()


if __name__ == "__main__":
    main()
