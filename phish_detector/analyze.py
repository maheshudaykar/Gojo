from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Protocol

from phish_detector.brand_risk import compute_brand_typo_risk
from phish_detector.enrichment import get_domain_enrichment
from phish_detector.feedback import create_entry, record_pending, resolve_feedback
from phish_detector.parsing import parse_url
from phish_detector.policy import BanditPolicy, DEFAULT_WEIGHT
from phish_detector.report import build_report
from phish_detector.rules import RuleHit, evaluate_rules
from phish_detector.scoring import binary_label_for_score, combine_scores, compute_score
from phish_detector.policy import context_from_scores
from phish_detector.typosquat import detect_typosquatting


# Protocol for policy compatibility (supports both v1 and v2)
class PolicyProtocol(Protocol):
    """Protocol for policy objects to support both v1 and v2."""
    
    def select_action(self, ml_confidence: float, rule_score: int, **kwargs: Any) -> Any:
        """Select action based on context."""
        ...
    
    def update(self, context: str, action: float, reward: float, **kwargs: Any) -> None:
        """Update policy with reward."""
        ...


@dataclass(frozen=True)
class AnalysisConfig:
    ml_mode: str
    lexical_model: str
    char_model: str
    policy_path: str
    feedback_store: str
    shadow_learn: bool = False
    label: str | None = None
    resolve_feedback: str | None = None
    enable_feedback: bool = True


def load_ml_context(config: AnalysisConfig) -> dict[str, Any] | None:
    if config.ml_mode == "none":
        return None
    try:
        if config.ml_mode == "lexical":
            from phish_detector.ml_lexical import load_lexical_model

            return {"mode": "lexical", "model": load_lexical_model(config.lexical_model), "available": True}
        if config.ml_mode == "char":
            from phish_detector.ml_char_ngram import load_char_model

            return {"mode": "char", "model": load_char_model(config.char_model), "available": True}
        from phish_detector.ml_ensemble import load_models

        lexical_model, char_model = load_models(config.lexical_model, config.char_model)
        return {
            "mode": "ensemble",
            "lexical_model": lexical_model,
            "char_model": char_model,
            "available": True,
        }
    except (ImportError, FileNotFoundError) as exc:
        logging.warning("ML unavailable: %s", exc)
        return {"mode": config.ml_mode, "available": False, "error": str(exc)}


def analyze_url(
    url: str,
    config: AnalysisConfig,
    ml_context: dict[str, Any] | None = None,
    policy: Any = None,  # Accept any policy object
) -> tuple[dict[str, Any], dict[str, Any]]:
    parsed = parse_url(url)
    features, hits = evaluate_rules(parsed)
    enrichment = get_domain_enrichment(parsed.host, parsed.original)
    typo_match = detect_typosquatting(parsed)
    brand_risk = compute_brand_typo_risk(parsed, features, typo_match, enrichment)
    if typo_match:
        if brand_risk.score >= 75 and brand_risk.corroborating:
            hits.append(
                RuleHit(
                    "brand_typo_contextual_high",
                    18,
                    f"Brand typo risk {brand_risk.score:.1f} corroborated by {', '.join(brand_risk.corroborating)}",
                )
            )
        elif brand_risk.score >= 45:
            hits.append(
                RuleHit(
                    "brand_typo_contextual",
                    10,
                    f"Brand typo risk {brand_risk.score:.1f}",
                )
            )

    rule_score = compute_score(hits)

    ml_info: dict[str, Any] = {"mode": config.ml_mode, "available": False}
    ml_score: int | None = None
    ml_confidence = 0.0
    ml_pred = 0

    if ml_context is None and config.ml_mode != "none":
        ml_context = load_ml_context(config)

    if ml_context and ml_context.get("available"):
        if ml_context["mode"] == "lexical":
            from phish_detector.ml_lexical import predict_lexical_proba

            proba = predict_lexical_proba(ml_context["model"], parsed.original)
            ml_info.update({"mode": "lexical", "lexical": proba, "available": True})
        elif ml_context["mode"] == "char":
            from phish_detector.ml_char_ngram import predict_char_proba

            proba = predict_char_proba(ml_context["model"], parsed.original)
            ml_info.update({"mode": "char", "char": proba, "available": True})
        else:
            from phish_detector.ml_ensemble import predict_ensemble_proba

            ensemble = predict_ensemble_proba(
                parsed.original,
                ml_context["lexical_model"],
                ml_context["char_model"],
            )
            proba = ensemble.ensemble
            ml_info.update(
                {
                    "mode": "ensemble",
                    "lexical": ensemble.lexical,
                    "char": ensemble.char,
                    "ensemble": ensemble.ensemble,
                    "available": True,
                }
            )
        ml_pred = 1 if proba >= 0.5 else 0
        ml_confidence = proba if ml_pred == 1 else 1.0 - proba
        ml_score = int(round(proba * 100))
        ml_info["confidence"] = ml_confidence
        ml_info["score"] = ml_score
    elif ml_context and not ml_context.get("available", True):
        ml_info.update({"available": False, "error": ml_context.get("error")})

    policy_info: dict[str, Any] | None = None
    weight = DEFAULT_WEIGHT
    if ml_score is not None:
        if policy is None:
            policy = BanditPolicy(config.policy_path)
        context_parts = [
            "typo1" if typo_match else "typo0",
            "intent1" if brand_risk.components["I"] >= 0.5 else "intent0",
        ]
        age_bucket = "age_unknown"
        if enrichment.age_days is not None:
            if enrichment.age_days < 30:
                age_bucket = "age_new"
            elif enrichment.age_days < 365:
                age_bucket = "age_mid"
            else:
                age_bucket = "age_old"
        rep_bucket = "rep_low" if enrichment.reputation_trust < 0.35 else "rep_high" if enrichment.reputation_trust > 0.7 else "rep_mid"
        context_parts.extend([age_bucket, rep_bucket])
        context_override = f"{context_from_scores(ml_confidence, rule_score.score)}|" + "|".join(context_parts)
        decision = policy.select_action(
            ml_confidence,
            rule_score.score,
            context_override=context_override,
        )  # type: ignore[union-attr]
        weight = DEFAULT_WEIGHT if config.shadow_learn else decision.action
        policy_info = {
            "weight": weight,
            "context": decision.context,
            "shadow": config.shadow_learn,
        }
        # Add v2-specific metadata if available
        if hasattr(decision, "strategy"):
            policy_info["strategy"] = decision.strategy
        if hasattr(decision, "confidence"):
            policy_info["confidence"] = decision.confidence
        if hasattr(decision, "epsilon"):
            policy_info["epsilon"] = decision.epsilon

    if ml_score is None:
        final_score = rule_score
    else:
        final_score = combine_scores(rule_score.score, ml_score, weight, hits)

    predicted_label = binary_label_for_score(final_score.score)
    feedback_info: dict[str, Any] | None = None

    if config.enable_feedback:
        if config.resolve_feedback and config.label:
            resolved = resolve_feedback(config.resolve_feedback, config.label, config.feedback_store)
            if resolved is None:
                feedback_info = {"status": "missing", "id": config.resolve_feedback}
            elif ml_score is not None and resolved.confidence >= 0.55:
                reward = resolved.confidence if resolved.predicted_label == config.label else -resolved.confidence
                (policy or BanditPolicy(config.policy_path)).update(resolved.context, resolved.action, reward)
                feedback_info = {"status": "resolved", "id": resolved.id, "updated": True}
            else:
                feedback_info = {"status": "resolved", "id": resolved.id, "updated": False}
        else:
            entry = create_entry(
                url=parsed.original,
                predicted_label=predicted_label,
                confidence=ml_confidence,
                context=policy_info["context"] if policy_info else "rule_only",
                action=weight,
                rule_score=rule_score.score,
                ml_score=float(ml_score or 0.0),
                final_score=final_score.score,
            )
            if config.label and ml_score is not None:
                entry.status = "resolved"
                entry.true_label = config.label
                if ml_confidence >= 0.55:
                    reward = ml_confidence if predicted_label == config.label else -ml_confidence
                    (policy or BanditPolicy(config.policy_path)).update(entry.context, entry.action, reward)
                    feedback_info = {"status": "resolved", "id": entry.id, "updated": True}
                else:
                    feedback_info = {"status": "resolved", "id": entry.id, "updated": False}
                record_pending(entry, config.feedback_store)
            elif config.label:
                entry.status = "resolved"
                entry.true_label = config.label
                record_pending(entry, config.feedback_store)
                feedback_info = {"status": "resolved", "id": entry.id, "updated": False}
            else:
                record_pending(entry, config.feedback_store)
                feedback_info = {"status": "pending", "id": entry.id}

    report: dict[str, Any] = build_report(
        parsed,
        features,
        final_score,
        ml_info,
        policy_info,
        feedback_info,
        context_info={
            "brand_typo_risk": brand_risk.score,
            "brand_typo_components": brand_risk.components,
            "brand_typo_corroborating": brand_risk.corroborating,
            "domain_enrichment": {
                "registrable_domain": enrichment.registrable_domain,
                "age_days": enrichment.age_days,
                "age_trust": enrichment.age_trust,
                "asn": enrichment.asn,
                "asn_org": enrichment.asn_org,
                "reputation_trust": enrichment.reputation_trust,
                "reputation_reasons": enrichment.reputation_reasons,
                "volatility_score": enrichment.volatility_score,
                "ip_addresses": enrichment.ip_addresses,
            },
        },
    )
    extra: dict[str, Any] = {
        "rule_score": rule_score.score,
        "ml_score": ml_score,
        "ml_confidence": ml_confidence,
        "policy_weight": weight if policy_info else None,
        "brand_typo_risk": brand_risk.score,
        "brand_typo_components": brand_risk.components,
        "domain_enrichment": {
            "registrable_domain": enrichment.registrable_domain,
            "age_days": enrichment.age_days,
            "age_trust": enrichment.age_trust,
            "reputation_trust": enrichment.reputation_trust,
            "reputation_reasons": enrichment.reputation_reasons,
            "volatility_score": enrichment.volatility_score,
            "ip_addresses": enrichment.ip_addresses,
        },
        "signals": [
            {"name": hit.name, "details": hit.details, "weight": hit.weight} for hit in hits
        ],
    }
    return report, extra
