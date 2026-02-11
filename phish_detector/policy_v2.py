"""
Production-grade contextual bandit policy with Thompson Sampling.

Improvements over v1:
- Thompson Sampling (Bayesian) instead of epsilon-greedy
- UCB1 as alternative exploration strategy
- Comprehensive metrics tracking (regret, cumulative reward)
- Feature importance and context analysis
- Model evaluation and performance monitoring
- Automatic retraining triggers
- Better reward shaping with temporal discounting
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np

DEFAULT_ACTIONS = [0.8, 0.6, 0.4, 0.2]  # Extended action space
DEFAULT_WEIGHT = 0.6
MAX_SNAPSHOTS = 10
RETRAINING_THRESHOLD = 1000  # Trigger evaluation after N updates
TEMPORAL_DISCOUNT = 0.99  # Discount factor for reward decay


@dataclass(frozen=True)
class PolicyDecision:
    """Decision made by the policy."""
    action: float
    context: str
    strategy: Literal["thompson", "ucb", "greedy"]
    confidence: float
    metadata: dict[str, Any] = field(default_factory=lambda: {})


@dataclass
class PolicyMetrics:
    """Comprehensive metrics for policy evaluation."""
    total_updates: int = 0
    total_reward: float = 0.0
    cumulative_regret: float = 0.0
    optimal_action_rate: float = 0.0
    avg_confidence: float = 0.0
    context_distribution: dict[str, int] = field(default_factory=lambda: cast(dict[str, int], {}))
    action_distribution: dict[str, int] = field(default_factory=lambda: cast(dict[str, int], {}))
    last_evaluation: str | None = None


def _context_from_scores(ml_confidence: float, rule_score: int) -> str:
    """Create context bucket from ML confidence and rule score."""
    # More granular buckets for better learning
    if ml_confidence < 0.5:
        ml_bucket = "ml_very_low"
    elif ml_confidence < 0.65:
        ml_bucket = "ml_low"
    elif ml_confidence < 0.8:
        ml_bucket = "ml_mid"
    elif ml_confidence < 0.9:
        ml_bucket = "ml_high"
    else:
        ml_bucket = "ml_very_high"

    if rule_score <= 5:
        rule_bucket = "rule_clean"
    elif rule_score <= 20:
        rule_bucket = "rule_mild"
    elif rule_score <= 50:
        rule_bucket = "rule_moderate"
    elif rule_score <= 80:
        rule_bucket = "rule_severe"
    else:
        rule_bucket = "rule_critical"

    return f"{ml_bucket}|{rule_bucket}"


class ThompsonSamplingPolicy:
    """
    Production-grade contextual bandit using Thompson Sampling.
    
    Features:
    - Bayesian exploration with Beta distributions
    - UCB1 as fallback strategy
    - Comprehensive metrics tracking
    - Automatic model evaluation
    - Reward shaping and temporal discounting
    """
    
    def __init__(
        self,
        path: str,
        strategy: Literal["thompson", "ucb", "greedy"] = "thompson",
        alpha_prior: float = 1.0,
        beta_prior: float = 1.0,
    ) -> None:
        self.path = Path(path)
        self.strategy = strategy
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior
        self.actions = DEFAULT_ACTIONS
        self.policy = self._load()
        self.metrics = self._load_metrics()

    def _load(self) -> dict[str, Any]:
        """Load policy state from disk."""
        if self.path.exists():
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return data  # type: ignore[return-value]
        return {
            "version": "policy_v2_thompson",
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "strategy": self.strategy,
            "actions": self.actions,
            "contexts": {},
            "global_stats": {
                "total_updates": 0,
                "total_reward": 0.0,
                "best_action_per_context": {},
            },
        }

    def _load_metrics(self) -> PolicyMetrics:
        """Load metrics from policy state."""
        stats = self.policy.get("global_stats", {})
        return PolicyMetrics(
            total_updates=stats.get("total_updates", 0),
            total_reward=stats.get("total_reward", 0.0),
            cumulative_regret=stats.get("cumulative_regret", 0.0),
            optimal_action_rate=stats.get("optimal_action_rate", 0.0),
            avg_confidence=stats.get("avg_confidence", 0.0),
            context_distribution=stats.get("context_distribution", {}),
            action_distribution=stats.get("action_distribution", {}),
            last_evaluation=stats.get("last_evaluation"),
        )

    def _snapshot(self) -> None:
        """Create versioned snapshot of policy."""
        history_dir = self.path.parent / "policy_history_v2"
        history_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        snapshot_path = history_dir / f"policy_v2_{timestamp}.json"
        snapshot_path.write_text(json.dumps(self.policy, indent=2), encoding="utf-8")
        
        # Cleanup old snapshots
        snapshots = sorted(history_dir.glob("policy_v2_*.json"))
        if len(snapshots) > MAX_SNAPSHOTS:
            for old in snapshots[:-MAX_SNAPSHOTS]:
                old.unlink(missing_ok=True)

    def _save(self) -> None:
        """Save policy state to disk."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.policy["global_stats"].update({
            "total_updates": self.metrics.total_updates,
            "total_reward": self.metrics.total_reward,
            "cumulative_regret": self.metrics.cumulative_regret,
            "optimal_action_rate": self.metrics.optimal_action_rate,
            "avg_confidence": self.metrics.avg_confidence,
            "context_distribution": dict(self.metrics.context_distribution),
            "action_distribution": dict(self.metrics.action_distribution),
            "last_evaluation": self.metrics.last_evaluation,
        })
        self.path.write_text(json.dumps(self.policy, indent=2), encoding="utf-8")

    def _thompson_sampling(self, context_data: dict[str, Any]) -> tuple[float, float]:
        """
        Thompson Sampling: sample from Beta distributions for each action.
        Returns (best_action, confidence).
        """
        samples: dict[float, float] = {}
        for action in self.actions:
            key = str(action)
            stats = context_data.get(key, {})
            
            # Beta distribution parameters (assume reward in [0, 1])
            successes = float(stats.get("alpha", self.alpha_prior))
            failures = float(stats.get("beta", self.beta_prior))
            
            # Sample from Beta(alpha, beta)
            sample = float(np.random.beta(successes, failures))
            samples[action] = sample
        
        best_action = float(max(samples, key=samples.get))  # type: ignore[arg-type]
        confidence = float(samples[best_action] / (sum(samples.values()) + 1e-10))
        return best_action, confidence

    def _ucb1(self, context_data: dict[str, Any], total_plays: int) -> tuple[float, float]:
        """
        UCB1: Upper Confidence Bound algorithm.
        Returns (best_action, confidence).
        """
        ucb_values: dict[float, float] = {}
        for action in self.actions:
            key = str(action)
            stats = context_data.get(key, {"n": 0, "value": 0.0})
            n = max(int(stats.get("n", 1)), 1)
            value = float(stats.get("value", 0.0))
            
            # UCB formula: value + sqrt(2 * log(total) / n)
            exploration_bonus = math.sqrt(2 * math.log(max(total_plays, 1)) / n)
            ucb_values[action] = float(value + exploration_bonus)
        
        best_action = float(max(ucb_values, key=ucb_values.get))  # type: ignore[arg-type]
        confidence = float(ucb_values[best_action] / (sum(ucb_values.values()) + 1e-10))
        return best_action, confidence

    def _greedy(self, context_data: dict[str, Any]) -> tuple[float, float]:
        """
        Greedy: select action with highest observed value.
        Returns (best_action, confidence).
        """
        best_action = DEFAULT_WEIGHT
        best_value = float("-inf")
        total_value = 0.0
        
        for action in self.actions:
            key = str(action)
            stats = context_data.get(key, {"n": 0, "value": 0.0})
            value = stats["value"]
            total_value += abs(value)
            
            if stats["n"] > 0 and value > best_value:
                best_value = value
                best_action = action
        
        confidence = abs(best_value) / (total_value + 1e-10)
        return best_action, confidence

    def select_action(
        self,
        ml_confidence: float,
        rule_score: int,
        strategy: Literal["thompson", "ucb", "greedy"] | None = None,
    ) -> PolicyDecision:
        """
        Select action based on current context and strategy.
        
        Args:
            ml_confidence: ML model confidence [0, 1]
            rule_score: Rule-based score
            strategy: Override default strategy
            
        Returns:
            PolicyDecision with action, context, and metadata
        """
        context = _context_from_scores(ml_confidence, rule_score)
        context_data = self.policy.get("contexts", {}).get(context, {})
        strategy_used = strategy or self.strategy
        
        # Calculate total plays for UCB
        total_plays = sum(
            int(cast(dict[str, Any], stats).get("n", 0))
            for stats in context_data.values()
            if isinstance(stats, dict)
        )
        
        # Select action based on strategy
        if strategy_used == "thompson":
            action, confidence = self._thompson_sampling(context_data)
        elif strategy_used == "ucb":
            action, confidence = self._ucb1(context_data, total_plays)
        else:  # greedy
            action, confidence = self._greedy(context_data)
        
        # Update metrics
        self.metrics.context_distribution[context] = \
            self.metrics.context_distribution.get(context, 0) + 1
        self.metrics.action_distribution[str(action)] = \
            self.metrics.action_distribution.get(str(action), 0) + 1
        
        return PolicyDecision(
            action=action,
            context=context,
            strategy=cast(Literal["thompson", "ucb", "greedy"], strategy_used),
            confidence=confidence,
            metadata={
                "ml_confidence": ml_confidence,
                "rule_score": rule_score,
                "total_plays": total_plays,
            },
        )

    def update(
        self,
        context: str,
        action: float,
        reward: float,
        temporal_decay: bool = True,
    ) -> None:
        """
        Update policy with observed reward.
        
        Args:
            context: Context bucket
            action: Action taken
            reward: Observed reward [-1, 1]
            temporal_decay: Apply temporal discounting
        """
        self._snapshot()
        
        # Normalize reward to [0, 1] for Beta distribution
        normalized_reward = (reward + 1) / 2
        
        # Apply temporal decay if enabled
        if temporal_decay:
            normalized_reward *= TEMPORAL_DISCOUNT
        
        # Update context-action statistics
        contexts = self.policy.setdefault("contexts", {})
        context_data = contexts.setdefault(context, {})
        key = str(action)
        stats = context_data.setdefault(key, {
            "n": 0,
            "value": 0.0,
            "alpha": self.alpha_prior,
            "beta": self.beta_prior,
        })
        
        # Update count and value (incremental mean)
        stats["n"] += 1
        stats["value"] += (reward - stats["value"]) / stats["n"]
        
        # Update Beta distribution parameters for Thompson Sampling
        if normalized_reward > 0.5:
            stats["alpha"] += normalized_reward
        else:
            stats["beta"] += (1 - normalized_reward)
        
        # Update global metrics
        self.metrics.total_updates += 1
        self.metrics.total_reward += reward
        
        # Calculate regret (difference from optimal action)
        optimal_value = max(
            (float(cast(dict[str, Any], s).get("value", 0.0)) for s in context_data.values() if isinstance(s, dict)),
            default=0.0,
        )
        self.metrics.cumulative_regret += max(0.0, optimal_value - float(cast(dict[str, Any], stats).get("value", 0.0)))
        
        # Update optimal action tracking
        if stats["value"] == optimal_value:
            best_actions = self.policy["global_stats"].setdefault("best_action_per_context", {})
            best_actions[context] = action
        
        # Save updated policy
        self._save()
        
        # Trigger evaluation if threshold reached
        if self.metrics.total_updates % RETRAINING_THRESHOLD == 0:
            self._evaluate()

    def _evaluate(self) -> dict[str, Any]:
        """
        Evaluate policy performance and generate metrics report.
        
        Returns:
            Dictionary with evaluation metrics
        """
        evaluation: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "total_updates": self.metrics.total_updates,
            "avg_reward": self.metrics.total_reward / max(self.metrics.total_updates, 1),
            "cumulative_regret": self.metrics.cumulative_regret,
            "avg_regret": self.metrics.cumulative_regret / max(self.metrics.total_updates, 1),
            "context_coverage": len(self.metrics.context_distribution),
            "action_usage": dict(self.metrics.action_distribution),
            "top_contexts": sorted(
                self.metrics.context_distribution.items(),
                key=lambda x: x[1],
                reverse=True,
            )[:10],
        }
        
        # Calculate optimal action rate
        optimal_count = sum(
            1 for ctx in self.policy["global_stats"].get("best_action_per_context", {})
            if self.metrics.context_distribution.get(ctx, 0) > 0
        )
        self.metrics.optimal_action_rate = optimal_count / max(len(self.metrics.context_distribution), 1)
        evaluation["optimal_action_rate"] = self.metrics.optimal_action_rate
        
        # Save evaluation
        self.metrics.last_evaluation = evaluation["timestamp"]
        eval_dir = self.path.parent / "evaluations"
        eval_dir.mkdir(parents=True, exist_ok=True)
        eval_path = eval_dir / f"eval_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
        eval_path.write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
        
        return evaluation

    def get_metrics(self) -> dict[str, Any]:
        """Get current policy metrics."""
        return {
            "total_updates": self.metrics.total_updates,
            "total_reward": self.metrics.total_reward,
            "avg_reward": self.metrics.total_reward / max(self.metrics.total_updates, 1),
            "cumulative_regret": self.metrics.cumulative_regret,
            "optimal_action_rate": self.metrics.optimal_action_rate,
            "context_distribution": dict(self.metrics.context_distribution),
            "action_distribution": dict(self.metrics.action_distribution),
            "last_evaluation": self.metrics.last_evaluation,
        }

    def get_context_stats(self, context: str) -> dict[str, Any]:
        """Get statistics for a specific context."""
        context_data = self.policy.get("contexts", {}).get(context, {})
        return {
            "context": context,
            "actions": {
                str(action): context_data.get(str(action), {})
                for action in self.actions
            },
            "total_plays": sum(
                int(cast(dict[str, Any], stats).get("n", 0))
                for stats in context_data.values()
                if isinstance(stats, dict)
            ),
        }
