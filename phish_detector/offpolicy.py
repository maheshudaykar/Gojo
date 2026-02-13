from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from phish_detector.feedback import FeedbackEntry, load_entries
from phish_detector.policy import DEFAULT_WEIGHT
from phish_detector.reward import cost_sensitive_reward


@dataclass(frozen=True)
class OffPolicyResult:
    ips: float
    snips: float
    dr: float
    count: int
    skipped: int
    guardrail_violations: int


def _coerce_entries(path: Path) -> list[FeedbackEntry]:
    entries: list[FeedbackEntry] = []
    for entry in load_entries(str(path)):
        if entry.status == "resolved" and entry.true_label:
            entries.append(entry)
    return entries


def _action_match(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(a - b) <= tol


def _reward(entry: FeedbackEntry, fn_cost: float, fp_cost: float) -> float:
    return cost_sensitive_reward(
        entry.predicted_label,
        entry.true_label or "",
        entry.confidence,
        fn_cost,
        fp_cost,
    )


def evaluate_offpolicy_entries(
    entries: list[FeedbackEntry],
    policy: Any,
    fn_cost: float,
    fp_cost: float,
    min_policy_confidence: float,
    max_weight_shift: float,
) -> OffPolicyResult:
    if not entries:
        return OffPolicyResult(ips=0.0, snips=0.0, dr=0.0, count=0, skipped=0, guardrail_violations=0)

    rewards: list[float] = []
    ips_weights: list[float] = []
    dr_estimates: list[float] = []
    skipped = 0
    guardrail_violations = 0

    # Reward model: average reward per (context, action)
    reward_table: dict[str, dict[float, list[float]]] = {}
    for entry in entries:
        reward_table.setdefault(entry.context, {}).setdefault(entry.action, []).append(
            _reward(entry, fn_cost, fp_cost)
        )

    reward_mean: dict[str, dict[float, float]] = {}
    for context, actions in reward_table.items():
        reward_mean[context] = {action: sum(values) / len(values) for action, values in actions.items()}

    for entry in entries:
        if entry.propensity is None or entry.propensity <= 0:
            skipped += 1
            continue

        decision = policy.select_action(
            entry.confidence,
            entry.rule_score,
            context_override=entry.context,
        )
        target_action = float(decision.action)
        target_propensity = float(getattr(decision, "propensity", 1.0))

        min_weight = max(0.0, DEFAULT_WEIGHT - max_weight_shift)
        max_weight = min(1.0, DEFAULT_WEIGHT + max_weight_shift)
        if entry.confidence < min_policy_confidence and target_action != DEFAULT_WEIGHT:
            guardrail_violations += 1
        if target_action < min_weight or target_action > max_weight:
            guardrail_violations += 1

        reward = _reward(entry, fn_cost, fp_cost)
        rewards.append(reward)

        if _action_match(target_action, entry.action):
            ips_weights.append(target_propensity / entry.propensity)
        else:
            ips_weights.append(0.0)

        # Doubly robust estimate
        context_rewards = reward_mean.get(entry.context, {})
        q_hat_target = context_rewards.get(target_action, 0.0)
        q_hat_logged = context_rewards.get(entry.action, 0.0)
        correction = (
            (target_propensity / entry.propensity) * (reward - q_hat_logged)
            if _action_match(target_action, entry.action)
            else 0.0
        )
        dr_estimates.append(q_hat_target + correction)

    if not rewards:
        return OffPolicyResult(
            ips=0.0,
            snips=0.0,
            dr=0.0,
            count=0,
            skipped=skipped,
            guardrail_violations=guardrail_violations,
        )

    ips = sum(w * r for w, r in zip(ips_weights, rewards)) / max(len(rewards), 1)
    snips_denom = sum(ips_weights) or 1.0
    snips = sum(w * r for w, r in zip(ips_weights, rewards)) / snips_denom
    dr = sum(dr_estimates) / max(len(dr_estimates), 1)

    return OffPolicyResult(
        ips=ips,
        snips=snips,
        dr=dr,
        count=len(rewards),
        skipped=skipped,
        guardrail_violations=guardrail_violations,
    )


def evaluate_offpolicy(
    feedback_path: str,
    policy: Any,
    fn_cost: float,
    fp_cost: float,
    min_policy_confidence: float,
    max_weight_shift: float,
) -> OffPolicyResult:
    entries = _coerce_entries(Path(feedback_path))
    return evaluate_offpolicy_entries(
        entries,
        policy,
        fn_cost,
        fp_cost,
        min_policy_confidence,
        max_weight_shift,
    )
