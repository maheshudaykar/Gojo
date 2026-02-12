# Threat Model

## Overview
This document enumerates attacker capabilities, operational assumptions, and failure modes for Gojo.

## Attacker Capabilities
- Register or compromise domains and subdomains.
- Craft lookalike URLs using typos, transpositions, and homoglyphs.
- Abuse URL shorteners and nested redirects.
- Inject URL-encoded obfuscation and deep subdomain structures.
- Host content on shared infrastructure and fast-flux IP ranges.

## Defender Assumptions
- Only URL strings are available at decision time.
- Enrichment lookups may fail or be unavailable.
- Feedback labels can be delayed or noisy.

## Data Poisoning and Feedback Abuse
- Attackers may submit poisoned URLs to skew feedback rewards.
- Feedback labels can be spoofed to induce model drift.
- Mitigations:
  - Off-policy evaluation with IPS/SNIPS/DR before deployment.
  - Guardrails on policy weight shifts and confidence thresholds.
  - Label-quality scoring and review queues.

## API Abuse Scenarios
- Enrichment endpoints used to enumerate or exhaust quotas.
- Safe Browsing API key leakage or rate-limit exhaustion.
- Mitigations:
  - Disable enrichment via env when required.
  - Cache lookups and enforce timeouts.

## Failure Mode Taxonomy
- False negatives for brand-typo phishing.
- False positives on benign lookalikes.
- Domain-family leakage across splits.
- Concept drift after attacker behavior changes.

## Red-Team Evaluation
- Synthetic perturbation suite (typo, transposition, homoglyph, URL encoding, deep subdomain).
- Attack success rate and score degradation metrics.
- Required pass criteria before rollout updates.
