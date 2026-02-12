from __future__ import annotations


def cost_sensitive_reward(
    predicted_label: str,
    true_label: str,
    confidence: float,
    fn_cost: float,
    fp_cost: float,
) -> float:
    confidence = max(0.0, min(1.0, confidence))
    if predicted_label == true_label:
        return confidence

    if true_label == "phish" and predicted_label == "legit":
        penalty = fn_cost
    else:
        penalty = fp_cost

    return max(-1.0, -confidence * penalty)
