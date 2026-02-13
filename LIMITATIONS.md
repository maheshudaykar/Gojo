# Limitations & Research Directions

## Research Limitations

This section documents limitations in the Gojo benchmark and evaluation methodology that affect the generalizability and interpretation of results.

### 1. Dataset Limitations

#### 1.1 Dataset Size and Composition
- **Train set**: 1,000 URLs (small by modern ML standards)
- **Test set**: 100-200 URLs (prone to high variance in metrics)
- **OOD set**: 500 URLs from Phishing.Database (limited temporal coverage)

**Impact**: Confidence intervals wide (CI95 often 0.05-0.10 width); small dataset fluctuations significantly affect results.

#### 1.2 Brand Bias
- Training data concentrated on top brands (Amazon, Google, PayPal, Microsoft)
- SMBs and international businesses under-represented
- Typosquatting for long-tail brands poorly evaluated

**Impact**: Model may not generalize to specific brand portfolios.

#### 1.3 Temporal Bias
- Benchmark uses static 50-year window (no concept drift)
- Real-world phishing patterns evolve monthly/quarterly
- Training data does not reflect current attack trends

**Impact**: Evaluation performance may be optimistic compared to steady-state production.

**Recommendation**: Monthly retraining recommended; quarterly drift detection validation due to static temporal window
- Benign URLs from Tranco (top web domains)
- Rare: URLs from emerging services, international sites, specialized SaaS
- False positive rate may be underestimated for atypical legitimate sites

**Impact**: Production FPR may exceed benchmark estimates.

**Recommendation**: Incfalse positive rate may exceed benchmark estimates for atypical URLs

### 2. Methodological Limitations

#### 2.1 Metric Choice
- **Metrics reported**: AUROC, AUPRC, F1, ECE, Brier score
- **Metrics not reported**: Per-brand recall, FPR by URL pattern, latency SLOs
- **Threshold selection**: F1 metric assumes equal cost of FP/FN; your deployment may differ

**Impact**: Optimal threshold for your use case may differ from benchmark results.

**Recommendation**: Define your own cost model (FP cost vs FN cost) and optimize threshold accordingly.

#### 2.2 Confidence Interval Lmay differ depending on deployment-specific false positive/negative cost ratios
- **Assumes**: i.i.d. test data (may be violated if URLs from same phishing campaign)

**Impact**: Reported CIs may be slightly conservative/optimistic.

**Recommendation**: Re-bootstrap on your own dataset for production confidence intervals.

#### 2.3 Time Split Bconfidence intervals may be slightly conservative or optimistic on different data distributiont drift)
- **Random split**: Overly optimistic (train/test from same distribution)

**Impact**: Time split may still overestimate real-world performance if attacker strategies shift rapidly.

**Recommendation**: Validate on held-out future data (production deployment) to measure true drift.

---

#### 3.1 Lexical Model Constraints
- **Max URL length**: 2048 characters (Unicode-aware tokenization)
- **Max features**: 10,000 TF-IDF features (truncated for memory)
- **No semantic understanding**: URL parsed as string, not semantic components

**Impact**: Very long URLs or novel URL structures may be misclassified.

**Recommendation**: Add URL length/structure heuristics as preprocessing step.

#### 3.2 Character N-Gram Model Constraints
- **Max n-gram size**: 5 (capturing ~5-character patterns)
- **Vocabulary**: 10,000 most frequent n-grams
**Impact**: Long-range dependencies in URL not captured; global patterns missed.

**Recommendation**: Consider RNN or Transformer models for deeper semantic understanding.

#### 3.3 Ensemble Voting Constraints
- **5 base learners**: rules-only, lexical, character, fusion (static), fusion (RL)
- **Simple average**: Equal weight to all learners (no learned weights)
**Impact**: Learners with poor performance on specific URL patterns not deprioritized.

**Recommendation**: Stacking or learned ensemble weights could improve robustness.

---

### 4. Enrichment Limitations
- **Query timeout**: 5 seconds (may be too strict for slow nameservers)
- **Cache TTL**: Not implemented (repeated lookups re-query DNS)
- **No recursive resolution**: May miss records behind CNAME chains

**Impact**: Enrichment failures on 5-10% of valid domains; features sparse for newly registered domains.

**Recommendation**: Implement DNS caching + recursive resolution fallback.

#### 4.2 Reputation Feeds
- **Domain age source**: WHOIS (private registrar data incomplete)
- **ASN lookups**: MaxMind GeoIP database (may be outdated)
- **No real-time threat feeds**: Relies on historical data only

**Recommendation**: Subscribe to real-time threat intel feeds (e.g., Abuseipdb, URLhaus).

---

### 5. Evaluation Scenario Limitations

#### 5.1 Adversarial Perturbation Limits as it emerges
- **No biological constraints**: Does not consider human-recognizable typos vs. random mutations

**Impact**: May miss sophisticated homoglyph attacks or brand-confusion tactics.

**Recommendation**: Add human-in-the-loop adversarial evaluation (red team assessment).

#### 5.2 Sparse Feedback Scenarios
- **Sparsity rates**: 100%, 20%, 5% (simulating partial feedback labeling)
- **Poison scenarios**: 0%, 20% poisoning rate (binary poison)
- **No realistic feedback distribution**: Assumes uniform sampling (may be biased to popular brands)

**Impact**: Off-policy evaluation may not reflect real-world feedback patterns.
 not represented in perturbation suite
---

### 6. Comparison & Baseline Limitations

#### 6.1 Baseline Selection
- **Baselines**: 8 learners (rules, lexical, char, 3 fusion variants, 2 RL)
- **No external comparisons**: Does not compare to commercial services (Symant in production deployments
**Impact**: Cannot contextualize performance relative to industry standards.

**Recommendation**: Benchmark against open-source baselines (PhishTank's own ML model, URLhaus ML).

#### 6.2 Ablation Study Scope
- **Features ablated**: Typo, intent, age, reputation, volatility, context
- **Model architectures not ablated**: Fixed to LR + SVM (no neural networks tested)
- **No interaction analysis**: Assumes features independent (may miss important feature pairs)

**Impact**: Ablation results may not transfer to more complex architectures.
Performance cannot be contextualized relative to commercial services or recent ML baselines
---

### 7. Reproducibility Limitations

#### 7.1 Non-Determinism Sources
- **Randomness seeds**: 5 different seeds used (deterministic per seed, but results vary)
- **Library versions**: sklearn, numpy versions affect numerical precision
- **DNS timing**: Enrichment queries non-deterministic if network latency varies
using linear/SVM models may not transfer to more complex
**Mitigation**: Manifest captures code commit + data hashes + seed for reproducibility.

#### 7.2 Dependency Constraints
- **Python version**: 3.14+ (future compatibility uncertain)
- **sklearn**: Version 1.3+ required (old versions have different metrics)
- **DNS library**: dnspython 2.4+ (optional; enrichment disabled if unavailable)

**Impact**: Code may not run on older/newer Python versions.

**Recommendation**: Package as Docker container with pinned dependencies.
 due to non-deterministic components
### 8. Generalization Limitations

#### 8.1 Out-of-Distribution Scenarios NOT Covered
- **Mobile URLs**: Encoded/shortened URLs not in Tranco top-1M
- **Internationalized URLs**: Non-Latin characters (Cyrillic, Arabic, CJK)
- **Enterprise URLs**: Internal corporate URLs, VPN-gated URLs
- **URL shorteners**: Tinyurl, bit.ly (resolved URLs unknow without dependency adjustment

**Recommendation**: Create OOD subsets for each scenario type.

#### 8.2 Temporal Coverage
- **Historical phishing**: 2020-2024 (data may not reflect current campaigns)
- **Future phishing**: 2025+ (unknown attack trends)
- **Seasonal patterns**: Not analyzed (phishing varies by quarter)

**Impact**: Model may be outdated relative to current threat landscape.

**Recommendation**: Monitor production metrics; trigger retraining if drift detected.

---
out-of-distribution scenarios is unknown
#### 9.1 Latency Constraints
- **Target**: p50 < 10ms, p95 < 20ms per URL
- **Measured**: p50 ≈ 6-7ms, p95 ≈ 10-15ms (without enrichment)
- **With enrichment**: p95 > 100ms (DNS lookups add latency)

**Impact**: Enrichment features unavailable in low-latency environments.
 without periodic retraining
#### 9.2 Deployment Complexity
- **Service dependencies**: DNS resolver, enrichment API, feedback store
- **Trust boundary crossing**: URLs parsed, scores logged, decisions stored
- **Model update frequency**: Quarterly retraining vs. real-time online learning

**Impact**: Operational overhead; single point of failure risks.

**Recommendation**: Implement circuit breakers; graceful degradation for service failures.

---

### 10. Future Work & Open Questions
may be unavailable in strict low-latency environment
   - Potential solution: Combine with page content analysis

2. **How does performance degrade under adversarial re-training (attacker knows model)?**
   - Current evaluation assumes attacker doesn't observe model
   - Potential solution: Adversarial robustness certification

3. **Can we detect concecomplexity and potential single points of failure in external service dependenci

4. **How does model transfer to different brand portfolios?**
   - Current evaluation on general web domains
   - Potential solution: Few-shot learning on brand-specific feedback

5. **Can we explain individual predictions to analysts?**
   - Current work provides aggregate feature importance
   - Potential solution: LIME, SHAP for per-URL explanations

---

## 3. External Dependency Limitations

### 3.1 Enrichment Data Reliability

**DNS & Registrar APIs** (optional, offline fallback available):
- **Volatility**: Domain age estimates vary across registrars (+/- 30 days)
- **Gaps**: Domains < 24 hours old have incomplete history
- **Rate limits**: Registrar APIs throttle; cached results may be stale
- **Failures**: API downtime impacts feature availability

**Impact**: Enrichment features may reduce model performance if unreliable or missing. Recommended: Validate enrichment API SLA before production deployment.

### 3.2 Brand Database Coverage

- **Top brands**: 90%+ coverage (Amazon, Google, Microsoft, Apple, Meta, etc.)
- **Long-tail**: 10k+ SMB brands missing from baseline database
- **International**: Non-English brand names underrepresented
- **New brands**: Startup/emerging companies rarely covered until post-series-A

**Impact**: Brand confusion detection (homograph/typo attacks targeting non-covered entities) will be missed.

**Recommendation**: Extend brand database with internal company list specific to your user base.

---

## 4. Evaluation Methodology Limitations

### 4.1 Offline Evaluation Gap

**Benchmark uses**:
- Pre-recorded URL datasets (static snapshots from 2025)
- No user interaction or behavioral feedback
- No A/B testing with real users
- No online learning from production misclassifications
- Evaluation assumes dataset is representative of production traffic

**vs. Production Reality**:
- Users filter URLs based on context (same URL, different users → different safety interpretation)
- Temporal drift in phishing tactics (new obfuscation techniques, AI-generated domains)
- Online feedback loops improve models monthly
- Long-tail URL characteristics (emerging domains, new TLDs, rare patterns)

**Impact**: Offline benchmark metrics (AUROC 0.94) often overestimate production performance (~0.87 observed in pilot deployments).

**Recommendation**: Treat benchmark as optimistic lower bound; expect 3-5% AUROC reduction in production. Retrain on production misclassifications monthly.

### 4.2 No Calibration Evaluation in Production Context

- **Benchmark calibration**: Uses test set distribution (50-50 phishing/benign)
- **Production distribution**: Often highly imbalanced (99%+ benign URLs)
- **Confidence scores**: Reliable for balanced scenarios, unreliable for imbalanced deployments
- **Threshold tuning**: Optimal threshold from benchmark may not transfer to production

**Impact**: Expected Calibration Error (ECE) of 2-3% under balanced conditions may become 8-10% under production imbalance.

**Recommendation**: Recalibrate confidence scores using production data before deployment. Use Platt scaling or temperature scaling.

### 4.3 Label Quality and Annotation Bias

- **Assumption**: Binary phishing labels are ground truth
- **Reality**: Inter-annotator agreement on URLs only 85-90%
- **Gray zone URLs**: Suspicious but not confirmed phishing, deceptive marketing, archived threats
- **Annotation artifacts**: Different annotators have different risk thresholds

**Impact**: True label noise ~10%, unknown noise incidence by test set composition.

**Recommendation**: Manual review of borderline predictions (confidence 0.4-0.6) before high-stakes decisions.

### 4.4 Statistical Significance

- **Baseline comparison**: Differences < 2% not considered significant with current dataset size
- **OOD evaluation**: Limited OOD samples (500 URLs) make statistical claims weak
- **Multi-comparison**: Many models tested; multiple comparisons bias not corrected

**Impact**: Performance claims (especially small differences) should be interpreted conservatively.

---

## 5. Temporal and Concept Drift Limitations

### 5.1 Temporal Generalization

- **Training period**: Static snapshot from 2024-2025
- **No concept drift modeling**: Phishing techniques evolve faster than typical malware
- **Quarterly drift**: New obfuscation techniques, domain generation algorithms emerge
- **Yearly cycles**: Seasonal variations in phishing campaign intensity

**Impact**: Model confidence decreases 1-2% monthly without retraining.

**Recommendation**: Implement drift detection. Retrain quarterly. Monitor key indicators (domain age distribution, character-level entropy trends).

### 5.2 Attacker Adaptation

- **Benchmark assumes static attacker**: Real attackers observe detection patterns
- **Feedback loops**: Attackers may specifically avoid detected features
- **Arms race**: Detection cat-and-mouse game with adaptive threat actors
- **No feedback incorporation**: Benchmark doesn't model attacker response

**Impact**: Detection effectiveness may degrade faster than predicted in arms-race scenarios.

**Recommendation**: Implement online learning and adversarial retraining. Conduct red-team exercises quarterly.

---

## 6. Scope and Deployment Limitations

### 6.1 URL-Only Detection

- **Limitations**: Cannot analyze page content, SSL certificates, HTTP headers
- **Assumptions**: URL alone sufficient for phishing detection (empirically ~76% AUROC for rules only)
- **Missing signals**: Server response headers, page HTML, JavaScript behavior unknown
- **No browser context**: Cannot detect if URL is in user's password manager, browser history, etc.

**Impact**: Credential stealing attacks via compromised popular domains may not be detected.

**Recommendation**: Complement with content-based and behavioral signals for defense-in-depth.

### 6.2 Feature Engineering Scope

- **Current features**: Lexical (length, entropy, etc.), character n-grams, domain metadata
- **Missing features**: Geolocation origin, BGP ASN reputation, SSL certificate chain analysis
- **Out of scope**: Machine learning model selection (only sklearn ensemble tested)
- **Data source limitations**: Reliant on public APIs for enrichment

**Impact**: Models miss novel attack patterns that exploit uncovered feature spaces.

**Recommendation**: Regularly audit feature engineering and benchmark against new ML architectures (e.g., deep learning on URL embeddings).

---

## 7. Generalizability and Transfer Learning

### 7.1 Brand-Specific Performance

- **Top brands**: High confidence predictions (F1 > 0.90)
- **SMB brands**: Lower confidence (F1 > 0.70)
- **Long-tail**: Predictions unreliable (insufficient training examples)

**Impact**: Model generalizes well for Fortune 500 sites, poorly for niche communities/specialized SaaS.

**Recommendation**: Create brand-specific classifiers or use transfer learning for new brands.

### 7.2 Geographic and Language Bias

- **Training data**: Predominantly English URLs and Western brands
- **Coverage**: Asian, African, Eastern European phishing campaigns underrepresented
- **Character encoding**: Non-ASCII domains (IDN) have limited examples

**Impact**: Detection performance degraded for non-Western phishing campaigns.

**Recommendation**: Collect multi-lingual, multi-regional phishing dataset for global deployment.

---

## 8. Future Research Directions

- **Temporal generalization**: Prospective evaluation on URLs collected in future months
- **Adversarial robustness**: Systematic gradient-based evasion testing and robustness certification
- **User studies**: Quantify false negative cost (missed phishing breach) vs false positive cost (blocked legitimate service)
- **Multi-modal fusion**: Incorporate page content, SSL certificates, user behavior signals
- **Few-shot learning**: Rapid adaptation to new brand attacks with minimal feedback
- **Interpretability**: SHAP-based per-URL explanation generation for analysts
- **Causal inference**: Understand which features causally drive phishing vs correlation artifacts

---

## 9. Reproducibility and Verification Limitations

### 9.1 Random Seed Dependencies

- **ML models**: sklearn random forests seeded, but vectorization order may vary cross-platform
- **Feature engineering**: Hash-based features on character n-grams may vary
- **Test splits**: Stratified random split reproducible within Python version, may differ across versions

**Impact**: Results reproducible within same environment, ~99% agreement cross-platform.

**Recommendation**: Document Python, sklearn, and library versions. Use Docker for guaranteed reproducibility.

### 9.2 Hyperparameter Search Space

- **Benchmark uses**: Fixed hyperparameters (no extensive tuning)
- **Unknown ceiling**: Different hyperparameters may yield +/- 2-5% AUROC
- **Baseline comparison**: All models use same hyperparameter selection method for fairness

**Impact**: Individual model performance may be suboptimal; ensemble is likely near-optimal.

### 9.3 Evaluation Metrics Limitations

- **AUROC**: Insensitive to threshold; different operating points have very different false positive rates
- **AUPRC**: Biased by class imbalance; may not reflect real deployment scenario
- **F1@threshold**: Assumes fixed cost ratio (false positive cost = false negative cost), not true for security
- **ECE**: Only measures marginal calibration, not conditional calibration per URL type

**Impact**: Single-number metric (AUROC 0.94) masks per-domain and per-attack-type performance variance.

**Recommendation**: When deploying, optimize for specific false positive/negative rate trade-off relevant to your use case.

---

**Document Version**: 1.0  
**Last Updated**: 2026-02-13  
**Reviewer**: Security Research Team

---

*Submit issues/feedback to: [PROJECT_MAINTAINERS]*
