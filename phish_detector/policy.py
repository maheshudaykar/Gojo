from __future__ import annotations

import json
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

DEFAULT_ACTIONS = [0.8, 0.6, 0.4]
DEFAULT_EPSILON = 0.1
DEFAULT_WEIGHT = 0.6
MAX_SNAPSHOTS = 5


@dataclass(frozen=True)
class PolicyDecision:
    action: float
    context: str
    epsilon: float
    propensity: float
    source: str


def _context_from_scores(ml_confidence: float, rule_score: int) -> str:
    if ml_confidence < 0.6:
        ml_bucket = "ml_low"
    elif ml_confidence < 0.8:
        ml_bucket = "ml_mid"
    else:
        ml_bucket = "ml_high"

    if rule_score <= 10:
        rule_bucket = "rule_none"
    elif rule_score <= 40:
        rule_bucket = "rule_mild"
    else:
        rule_bucket = "rule_severe"

    return f"{ml_bucket}|{rule_bucket}"


def context_from_scores(ml_confidence: float, rule_score: int) -> str:
    return _context_from_scores(ml_confidence, rule_score)


class BanditPolicy:
    def __init__(self, path: str, epsilon: float = DEFAULT_EPSILON) -> None:
        self.path = Path(path)
        self.epsilon = epsilon
        self.actions = DEFAULT_ACTIONS
        self.policy = self._load()

    def _load(self) -> dict[str, Any]:
        if self.path.exists():
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return cast(dict[str, Any], data)
        return {
            "version": "policy_v1",
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "epsilon": self.epsilon,
            "actions": self.actions,
            "contexts": {},
        }

    def _snapshot(self) -> None:
        history_dir = self.path.parent / "policy_history"
        history_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        snapshot_path = history_dir / f"policy_{timestamp}.json"
        snapshot_path.write_text(json.dumps(self.policy, indent=2), encoding="utf-8")
        snapshots = sorted(history_dir.glob("policy_*.json"))
        if len(snapshots) > MAX_SNAPSHOTS:
            for old in snapshots[:-MAX_SNAPSHOTS]:
                old.unlink(missing_ok=True)

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.policy, indent=2), encoding="utf-8")

    def select_action(self, ml_confidence: float, rule_score: int, **kwargs: Any) -> PolicyDecision:
        context_override = kwargs.get("context_override")
        context = context_override or _context_from_scores(ml_confidence, rule_score)
        context_data = self.policy.get("contexts", {}).get(context)
        if not context_data:
            return PolicyDecision(
                action=DEFAULT_WEIGHT,
                context=context,
                epsilon=self.epsilon,
                propensity=1.0,
                source="default",
            )

        best_action = DEFAULT_WEIGHT
        best_value = float("-inf")
        for action in self.actions:
            stats = context_data.get(str(action), {"n": 0, "value": 0.0})
            if stats["value"] > best_value:
                best_value = stats["value"]
                best_action = action

        explore = random.random() < self.epsilon
        if explore:
            action = random.choice(self.actions)
            source = "explore"
        else:
            action = best_action
            source = "exploit"

        uniform = self.epsilon / max(len(self.actions), 1)
        if action == best_action:
            propensity = (1.0 - self.epsilon) + uniform
        else:
            propensity = uniform

        return PolicyDecision(
            action=action,
            context=context,
            epsilon=self.epsilon,
            propensity=propensity,
            source=source,
        )

    def update(self, context: str, action: float, reward: float) -> None:
        self._snapshot()
        contexts = self.policy.setdefault("contexts", {})
        context_data = contexts.setdefault(context, {})
        key = str(action)
        stats = context_data.setdefault(key, {"n": 0, "value": 0.0})
        stats["n"] += 1
        stats["value"] += (reward - stats["value"]) / stats["n"]
        self._save()
