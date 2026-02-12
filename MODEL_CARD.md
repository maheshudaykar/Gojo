# Model Card

## Model Details
- Task: phishing URL detection.
- Inputs: URL strings; optional enrichment signals.
- Outputs: risk score (0-100), decision (allow/block/review).

## Intended Use
- Pre-screening for phishing URLs.
- Triage into review queues for uncertain cases.

## Training Data
- Mixed phishing and benign URLs.
- Deduplicated and domain-family leakage controlled where possible.

## Evaluation
- Metrics: AUROC, AUPRC, F1, ECE, MCE, Brier, latency.
- Robustness: adversarial perturbation suite.
- Off-policy estimators: IPS, SNIPS, DR.

## Novelty Statement
- Contextual RL with enriched context and guardrails for weight shifts.
- Integrated brand-typo risk gating with enrichment features.
- Formal evaluation protocol with time and random splits plus ablations.

## Statistical Significance
- Bootstrap confidence intervals and permutation tests for key metrics.
- Compare RL-v2 vs static fusion on held-out splits.

## Limitations and External Validity
- Only URL-level signals; content-based cues are not used.
- Enrichment may fail in offline or blocked environments.
- Dataset biases may under-represent certain TLDs or brands.

## Ethical Considerations
- False positives can cause user friction and business impact.
- Review queue is used for low-confidence high-impact cases.
- Continuous monitoring for harm and bias.
