from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Any, Protocol

from phish_detector.brand_risk import BrandRiskConfig, compute_brand_typo_risk
from phish_detector.enrichment import default_domain_enrichment, get_domain_enrichment
from phish_detector.feedback import create_entry, record_pending, resolve_feedback
from phish_detector.parsing import parse_url
from phish_detector.policy import BanditPolicy, DEFAULT_WEIGHT
from phish_detector.reward import cost_sensitive_reward
from phish_detector.report import build_report
from phish_detector.rules import RuleHit, evaluate_rules
from phish_detector.scoring import ScoreResult, binary_label_for_score, combine_scores, compute_score, label_for_score
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
    enable_enrichment: bool = True
    enable_brand_risk: bool = True
    enable_context_enrichment: bool = True
    score_mode: str = "fusion"
    static_weight: float = DEFAULT_WEIGHT
    enable_policy: bool = True
    brand_risk: BrandRiskConfig = field(default_factory=BrandRiskConfig)
    enable_cost_sensitive: bool = True
    fn_cost: float = 3.0
    fp_cost: float = 1.0
    enable_guardrails: bool = True
    min_policy_confidence: float = 0.55
    max_weight_shift: float = 0.25
    enable_abstain: bool = True
    abstain_min_confidence: float = 0.6
    abstain_min_score: int = 60


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
    if config.enable_enrichment:
        enrichment = get_domain_enrichment(
            parsed.host,
            parsed.original,
            enable_age=config.brand_risk.enable_age,
            enable_reputation=config.brand_risk.enable_reputation,
            enable_volatility=config.brand_risk.enable_volatility,
        )
    else:
        enrichment = default_domain_enrichment(parsed.host)
    typo_match = detect_typosquatting(parsed)
    brand_risk = compute_brand_typo_risk(
        parsed,
        features,
        typo_match,
        enrichment,
        config.brand_risk,
    )
    if config.enable_brand_risk and typo_match:
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
    weight = config.static_weight
    if ml_score is not None and config.enable_policy:
        if policy is None:
            policy = BanditPolicy(config.policy_path)
        context_override = None
        if config.enable_context_enrichment:
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
        guardrails: list[str] = []
        weight_raw = weight
        if config.enable_guardrails:
            if ml_confidence < config.min_policy_confidence:
                weight = DEFAULT_WEIGHT
                guardrails.append("min_confidence")
            else:
                min_weight = max(0.0, DEFAULT_WEIGHT - config.max_weight_shift)
                max_weight = min(1.0, DEFAULT_WEIGHT + config.max_weight_shift)
                if weight < min_weight or weight > max_weight:
                    weight = min(max(weight, min_weight), max_weight)
                    guardrails.append("max_shift")

        policy_info = {
            "weight": weight,
            "weight_raw": weight_raw,
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
        if hasattr(decision, "propensity"):
            policy_info["propensity"] = decision.propensity
        if hasattr(decision, "source"):
            policy_info["source"] = decision.source
        if guardrails:
            policy_info["guardrails"] = guardrails

    if config.score_mode == "rules_only" or ml_score is None:
        final_score = rule_score
    elif config.score_mode == "ml_only":
        final_score = ScoreResult(score=ml_score, label=label_for_score(ml_score), hits=hits)
    else:
        final_score = combine_scores(rule_score.score, ml_score, weight, hits)

    predicted_label = binary_label_for_score(final_score.score)
    review_flag = False
    if config.enable_abstain and ml_score is not None:
        if ml_confidence < config.abstain_min_confidence and final_score.score >= config.abstain_min_score:
            review_flag = True
    feedback_info: dict[str, Any] | None = None

    if config.enable_feedback:
        if config.resolve_feedback and config.label:
            resolved = resolve_feedback(config.resolve_feedback, config.label, config.feedback_store)
            if resolved is None:
                feedback_info = {"status": "missing", "id": config.resolve_feedback}
            elif ml_score is not None and resolved.confidence >= 0.55:
                if config.enable_cost_sensitive:
                    reward = cost_sensitive_reward(
                        resolved.predicted_label,
                        config.label,
                        resolved.confidence,
                        config.fn_cost,
                        config.fp_cost,
                    )
                else:
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
                propensity=policy_info.get("propensity") if policy_info else None,
                policy_strategy=policy_info.get("strategy") if policy_info else None,
            )
            if config.label and ml_score is not None:
                entry.status = "resolved"
                entry.true_label = config.label
                if ml_confidence >= 0.55:
                    if config.enable_cost_sensitive:
                        reward = cost_sensitive_reward(
                            predicted_label,
                            config.label,
                            ml_confidence,
                            config.fn_cost,
                            config.fp_cost,
                        )
                    else:
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
    report["summary"]["decision"] = "review" if review_flag else "block" if predicted_label == "phish" else "allow"
    report["summary"]["review"] = review_flag
    extra: dict[str, Any] = {
        "rule_score": rule_score.score,
        "ml_score": ml_score,
        "ml_confidence": ml_confidence,
        "policy_weight": weight if policy_info else None,
        "policy_weight_raw": policy_info.get("weight_raw") if policy_info else None,
        "decision": report["summary"]["decision"],
        "review": review_flag,
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
