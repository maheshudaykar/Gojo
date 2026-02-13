# Limitations & Research Directions

## Research Limitations

This section documents limitations in the Gojo benchmark and evaluation methodology that affect the generalizability and interpretation of results.

### 1. Dataset Limitations

#### 1.1 Dataset Size and Composition
- **Train set**: 1,000 URLs (small by modern ML standards)
- **Test set**: 100-200 URLs (prone to high variance in metrics)
- **OOD set**: 500 URLs from Phishing.Database (limited temporal coverage)

**Impact**: Confidence intervals wide (CI95 often 0.05-0.10 width); small dataset fluctuations significantly affect results.

**Recommendation**: Validation on 10k+ URLs recommended before production deployment.

#### 1.2 Brand Bias
- Training data concentrated on top brands (Amazon, Google, PayPal, Microsoft)
- SMBs and international businesses under-represented
- Typosquatting for long-tail brands poorly evaluated

**Impact**: Model may not generalize to your specific brand portfolio.

**Recommendation**: Create brand-specific validation set from your user base.

#### 1.3 Temporal Bias
- Benchmark uses static 50-year window (no concept drift)
- Real-world phishing patterns evolve monthly/quarterly
- Training data does not reflect current attack trends

**Impact**: Evaluation performance may be optimistic compared to steady-state production.

**Recommendation**: Monthly retraining recommended; quarterly drift detection validation.

#### 1.4 Benign URL Distribution
- Benign URLs from Tranco (top web domains)
- Rare: URLs from emerging services, international sites, specialized SaaS
- False positive rate may be underestimated for atypical legitimate sites

**Impact**: Production FPR may exceed benchmark estimates.

**Recommendation**: Include brand-specific benign URLs in validation set.

---

### 2. Methodological Limitations

#### 2.1 Metric Choice
- **Metrics reported**: AUROC, AUPRC, F1, ECE, Brier score
- **Metrics not reported**: Per-brand recall, FPR by URL pattern, latency SLOs
- **Threshold selection**: F1 metric assumes equal cost of FP/FN; your deployment may differ

**Impact**: Optimal threshold for your use case may differ from benchmark results.

**Recommendation**: Define your own cost model (FP cost vs FN cost) and optimize threshold accordingly.

#### 2.2 Confidence Interval Limitations
- **Bootstrap**: 500 iterations (standard); wider CIs acceptable for early research
- **Permutation test**: 500 iterations; p-value precision ±0.002
- **Assumes**: i.i.d. test data (may be violated if URLs from same phishing campaign)

**Impact**: Reported CIs may be slightly conservative/optimistic.

**Recommendation**: Re-bootstrap on your own dataset for production confidence intervals.

#### 2.3 Time Split Bias
- **Time split**: Uses temporal order of training data
- **Assumes**: Older data ≈ similar distribution as newer data (may not hold for concept drift)
- **Random split**: Overly optimistic (train/test from same distribution)

**Impact**: Time split may still overestimate real-world performance if attacker strategies shift rapidly.

**Recommendation**: Validate on held-out future data (production deployment) to measure true drift.

---

### 3. Model Limitations

#### 3.1 Lexical Model Constraints
- **Max URL length**: 2048 characters (Unicode-aware tokenization)
- **Max features**: 10,000 TF-IDF features (truncated for memory)
- **No semantic understanding**: URL parsed as string, not semantic components

**Impact**: Very long URLs or novel URL structures may be misclassified.

**Recommendation**: Add URL length/structure heuristics as preprocessing step.

#### 3.2 Character N-Gram Model Constraints
- **Max n-gram size**: 5 (capturing ~5-character patterns)
- **Vocabulary**: 10,000 most frequent n-grams
- **No positional bias**: Treats initial/final characters same as middle

**Impact**: Long-range dependencies in URL not captured; global patterns missed.

**Recommendation**: Consider RNN or Transformer models for deeper semantic understanding.

#### 3.3 Ensemble Voting Constraints
- **5 base learners**: rules-only, lexical, character, fusion (static), fusion (RL)
- **Simple average**: Equal weight to all learners (no learned weights)
- **No adaptive thresholding**: Fixed decision boundary across all test samples

**Impact**: Learners with poor performance on specific URL patterns not deprioritized.

**Recommendation**: Stacking or learned ensemble weights could improve robustness.

---

### 4. Enrichment Limitations

#### 4.1 DNS Enrichment Constraints
- **Query timeout**: 5 seconds (may be too strict for slow nameservers)
- **Cache TTL**: Not implemented (repeated lookups re-query DNS)
- **No recursive resolution**: May miss records behind CNAME chains

**Impact**: Enrichment failures on 5-10% of valid domains; features sparse for newly registered domains.

**Recommendation**: Implement DNS caching + recursive resolution fallback.

#### 4.2 Reputation Feeds
- **Domain age source**: WHOIS (private registrar data incomplete)
- **ASN lookups**: MaxMind GeoIP database (may be outdated)
- **No real-time threat feeds**: Relies on historical data only

**Impact**: Reputations stale; cannot detect newly weaponized infrastructure.

**Recommendation**: Subscribe to real-time threat intel feeds (e.g., Abuseipdb, URLhaus).

---

### 5. Evaluation Scenario Limitations

#### 5.1 Adversarial Perturbation Limits
- **Perturbation budget**: 1-2 character edits (Damerau-Levenshtein distance)
- **Attack types**: Typo, transposition, homoglyph, URL encoding, subdomain abuse
- **No biological constraints**: Does not consider human-recognizable typos vs. random mutations

**Impact**: May miss sophisticated homoglyph attacks or brand-confusion tactics.

**Recommendation**: Add human-in-the-loop adversarial evaluation (red team assessment).

#### 5.2 Sparse Feedback Scenarios
- **Sparsity rates**: 100%, 20%, 5% (simulating partial feedback labeling)
- **Poison scenarios**: 0%, 20% poisoning rate (binary poison)
- **No realistic feedback distribution**: Assumes uniform sampling (may be biased to popular brands)

**Impact**: Off-policy evaluation may not reflect real-world feedback patterns.

**Recommendation**: Capture real feedback distribution from pilot production deployment.

---

### 6. Comparison & Baseline Limitations

#### 6.1 Baseline Selection
- **Baselines**: 8 learners (rules, lexical, char, 3 fusion variants, 2 RL)
- **No external comparisons**: Does not compare to commercial services (Symantec, Fortinet)
- **No recent ML baselines**: Does not compare to transformer models, graph-neural networks

**Impact**: Cannot contextualize performance relative to industry standards.

**Recommendation**: Benchmark against open-source baselines (PhishTank's own ML model, URLhaus ML).

#### 6.2 Ablation Study Scope
- **Features ablated**: Typo, intent, age, reputation, volatility, context
- **Model architectures not ablated**: Fixed to LR + SVM (no neural networks tested)
- **No interaction analysis**: Assumes features independent (may miss important feature pairs)

**Impact**: Ablation results may not transfer to more complex architectures.

**Recommendation**: Extend ablations to deep learning architectures.

---

### 7. Reproducibility Limitations

#### 7.1 Non-Determinism Sources
- **Randomness seeds**: 5 different seeds used (deterministic per seed, but results vary)
- **Library versions**: sklearn, numpy versions affect numerical precision
- **DNS timing**: Enrichment queries non-deterministic if network latency varies

**Impact**: Results not exactly reproducible across different runs/machines.

**Mitigation**: Manifest captures code commit + data hashes + seed for reproducibility.

#### 7.2 Dependency Constraints
- **Python version**: 3.14+ (future compatibility uncertain)
- **sklearn**: Version 1.3+ required (old versions have different metrics)
- **DNS library**: dnspython 2.4+ (optional; enrichment disabled if unavailable)

**Impact**: Code may not run on older/newer Python versions.

**Recommendation**: Package as Docker container with pinned dependencies.

---

### 8. Generalization Limitations

#### 8.1 Out-of-Distribution Scenarios NOT Covered
- **Mobile URLs**: Encoded/shortened URLs not in Tranco top-1M
- **Internationalized URLs**: Non-Latin characters (Cyrillic, Arabic, CJK)
- **Enterprise URLs**: Internal corporate URLs, VPN-gated URLs
- **URL shorteners**: Tinyurl, bit.ly (resolved URLs unknown)

**Impact**: Model performance on these scenarios unknown.

**Recommendation**: Create OOD subsets for each scenario type.

#### 8.2 Temporal Coverage
- **Historical phishing**: 2020-2024 (data may not reflect current campaigns)
- **Future phishing**: 2025+ (unknown attack trends)
- **Seasonal patterns**: Not analyzed (phishing varies by quarter)

**Impact**: Model may be outdated relative to current threat landscape.

**Recommendation**: Monitor production metrics; trigger retraining if drift detected.

---

### 9. Operational Limitations

#### 9.1 Latency Constraints
- **Target**: p50 < 10ms, p95 < 20ms per URL
- **Measured**: p50 ≈ 6-7ms, p95 ≈ 10-15ms (without enrichment)
- **With enrichment**: p95 > 100ms (DNS lookups add latency)

**Impact**: Enrichment features unavailable in low-latency environments.

**Recommendation**: Pre-cache enrichment; batch requests; use async DNS queries.

#### 9.2 Deployment Complexity
- **Service dependencies**: DNS resolver, enrichment API, feedback store
- **Trust boundary crossing**: URLs parsed, scores logged, decisions stored
- **Model update frequency**: Quarterly retraining vs. real-time online learning

**Impact**: Operational overhead; single point of failure risks.

**Recommendation**: Implement circuit breakers; graceful degradation for service failures.

---

### 10. Future Work & Open Questions

1. **Can we achieve <5% false positive rate without analyst review?**
   - Current work suggests no; inherent URL-only limitation
   - Potential solution: Combine with page content analysis

2. **How does performance degrade under adversarial re-training (attacker knows model)?**
   - Current evaluation assumes attacker doesn't observe model
   - Potential solution: Adversarial robustness certification

3. **Can we detect concept drift automatically and trigger retraining?**
   - Current work uses manual quarterly schedule
   - Potential solution: ADWIN, DDM-based drift detection

4. **How does model transfer to different brand portfolios?**
   - Current evaluation on general web domains
   - Potential solution: Few-shot learning on brand-specific feedback

5. **Can we explain individual predictions to analysts?**
   - Current work provides aggregate feature importance
   - Potential solution: LIME, SHAP for per-URL explanations

---

**Document Version**: 1.0  
**Last Updated**: 2026-02-13  
**Reviewer**: Security Research Team

---

*Submit issues/feedback to: [PROJECT_MAINTAINERS]*
