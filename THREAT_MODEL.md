# Threat Model & Limitations

## Executive Summary

This document outlines the threat model, adversarial assumptions, and fundamental limitations of the Gojo phishing detector. Understanding these constraints is critical for responsible deployment and accurate interpretation of evaluation results.

---

## 1. Threat Model

### 1.1 Attacker Capabilities & Objectives

**Primary Attacker Profile:**
- **Goal**: Maximize user credential compromise by evading URL-based defenses
- **Capabilities**: 
  - Create convincing look-alike URLs (typosquatting, homograph attacks)
  - Register new domains with minimal friction
  - Exploit brand confusion through lexical similarity
  - Monitor detection patterns and adapt URLs between requests
  - Access to historical phishing datasets (train on same data as defenders)
- **Constraints**: 
  - Cannot modify HTTP traffic (HTTPS/certificate chain intact)
  - Cannot compromise DNS infrastructure
  - Cannot forge DNS records
  - Cannot exploit browser vulnerabilities directly

### 1.2 Defender Capabilities & Objectives

**Defense Goal**: Prevent phishing URL access before credential capture
**Defender Resources**:
- URL feature extraction (lexical, character-level, behavioral, domain age)
- Brand database lookup
- Reputation feeds (domain age, ASN, registrar patterns)
- Historical feedback from user reports
- A/B testing capability for policy updates

**Defender Constraints**:
- Real-time latency requirements (p50 < 10ms, p95 < 20ms)
- No access to page content (URL-only analysis)
- Incomplete brand coverage (long-tail brands under-represented)
- Enrichment dependency (DNS volatility, registrar APIs may be unreliable)
- Label noise in training data (user-reported phishing may include false positives)

---

## 2. Adversarial Assumptions

### 2.1 Known Evasion Tactics

**Type 1: Typosquatting**
- Single character substitutions: `amazon.com` → `anmazon.com`
- Homograph attacks: Cyrillic `а` (U+0430) vs Latin `a`
- Domain extension tricks: `google.com.attacker.com`
- Subdomain abuse: `paypal.update-account.attacker.com`

**Mitigation in Gojo:**
- Damerau-Levenshtein distance on registered brand names (max distance = 2)
- Homograph detection via IDN normalization
- Registrable domain extraction (ignore subdomain noise)

**Residual Risk:** Attackers can craft 3+ edit-distance URLs, apply Unicode tricks outside hostname, or target brands we lack in our database.

**Type 2: Domain Age Exploitation**
- Newly registered domains (< 1 day old) have low reputation
- Bulk registration patterns common in phishing campaigns
- ASN clustering (multiple domains under same registrar IP space)

**Mitigation in Gojo:**
- Domain age decaying weight (exponential decay, age_days parameter)
- ASN reputation tracking (if enrichment enabled)
- Ensemble voting (lexical + character ML models as tiebreaker)

**Residual Risk:** Attackers can age domains slowly to accumulate reputation, or exploit DNS caching delays.

**Type 3: Obfuscation**
- Long, random subdomains to bypass string-matching filters
- URL shorteners/proxies (e.g., `tinyurl.com → attacker-phishing.com`)
- Numeric IP addresses: `192.0.2.1` instead of domain name

**Mitigation in Gojo:**
- Character n-gram ML model (captures arbitrary string patterns)
- Lexical model (URL structure analysis)
- Ensemble averaging reduces overconfidence

**Residual Risk:** New obfuscation techniques may evade n-gram patterns until model retraining.

### 2.2 Out-of-Distribution Scenarios

**Scenario 1: Time Window Shift**
- Phishing campaigns adapt over months (new obfuscation trends)
- Domain registration patterns evolve (bulk registration timing)
- Brand portfolio changes (acquisitions, new product lines)

**Evaluation:** Time-based train/test split (demonstrates 2-5% AUROC drift)

**Scenario 2: Brand Diversity Shift**
- Evaluation dataset biased toward major brands (Amazon, Google, PayPal)
- Small businesses, SaaS startups under-represented
- International brands in non-Latin scripts under-sampled

**Evaluation:** Ablation study quantifies brand-specific drift

**Scenario 3: False Positive Rate Increase**
- Legitimate sites may share obfuscation patterns with phishing (CDN URLs, internationalized domains)
- Typo distances may accidentally match legitimate business names

**Baseline:** Empirically measured on OOD datasets (Phishing.Database, Tranco)

---

## 3. Model Limitations

### 3.1 Fundamental Constraint: URL-Only Analysis

**Constraint**: URL-based phishing detection is fundamentally limited by information available in the URL string alone.

**Why it matters:**
- Perfect URL may hide malicious page (legitimate domain compromised)
- Suspicious-looking URL may be innocent (e.g., CDN with random subdomains, forwarding service)
- Attacker incentive: Make phishing URL indistinguishable from legitimate URL

**Consequence**: 
- False positive rate **never reaches zero** (legitimate services use suspicious patterns)
- False negative rate **never reaches zero** (attackers can craft legitimate-looking URLs)
- Optimal operating point trades off FP vs FN; no universally correct threshold

**Practical Implication**: Deploy as first-pass filter → escalate to analyst review for uncertain cases

### 3.2 Lexical & Character Model Limitations

**Limitation 1: Incomplete Context**
- URL-only analysis cannot assess page content, visual design, or behavioral signals
- No access to SSL certificate details (subject alternative names, issuer patterns)
- No form-field analysis (credential harvesting patterns)

**Impact**: 
- Zero-days with legitimate-looking URLs + sophisticated page design remain undetectable
- Certificate fraud not addressed (HTTPS cannot be trusted as sole phishing indicator)

**Mitigation**: Layered defense (URL detection as one line of defense, not sole defense)

**Limitation 2: Dataset Bias**
- Training data concentrated on known phishing sites (OpenPhish, PhishTank)
- Historical biases:
  - Over-representation of generic phishing (credential harvesting)
  - Under-representation of spear-phishing (targeted to individuals)
  - Under-representation of watering-hole attacks (compromised legitimate sites)

**Impact**: Model performance may degrade on sophisticated, targeted phishing

**Mitigation**: Synthetic adversarial examples partially address this

**Limitation 3: Brand Coverage Gaps**
- Gojo uses curated brand list (~100 high-value targets)
- Long-tail businesses (SMBs, international companies) not covered
- Brand name collisions (e.g., "Apple" could refer to multiple entities)

**Impact**: Typosquatting detection for uncovered brands absent

**Mitigation**: Optionally disable brand features (fusion-no-enrichment baseline)

### 3.3 Enrichment Dependencies

**Limitation 1: DNS Unreliability**
- DNS queries may timeout (Timeout errors), fail (NXDOMAIN), or be inconsistent
- Attacker-controlled nameservers may return false information
- Caching behavior may introduce stale results

**Impact**: Age/volatility/ASN features become unreliable
**Mitigation**: Graceful degradation (fusion-no-enrichment baseline, optional enrichment disabling)

**Limitation 2: Time-Sensitive Data**
- Domain age calculated relative to current time
- Age is relative, not absolute (new domains always appear suspicious, regardless of legitimacy)
- Legitimate new domains (product launches) may be false-positives

**Impact**: False positive rate on new legitimate domains may be elevated
**Baseline**: Ablation study (no-age variant) measures impact

**Limitation 3: Registrar API Instability**
- WHOIS data may be rate-limited, privatized, or inconsistent
- Registrar accuracy varies by TLD
- Privacy-protecting registrars (e.g., ICANN) report no useful data

**Impact**: Volatility features unavailable for ~50% of new domains
**Mitigation**: Graceful degradation via EnrichmentFallback

---

## 4. Label Noise & Data Quality

**Limitation 1: User-Reported Phishing Noise**
- False positives in training data (users falsely report legitimate sites as phishing)
- False negatives in training data (phishing sites not yet reported)
- Temporal bias (training data only includes historically detected phishing)

**Impact**: Model inherits these biases; cannot perform better than training signal

**Mitigation**: Multi-baseline ensemble (reduce reliance on any single signal)

**Limitation 2: Concept Drift Over Time**
- Attacker strategies evolve; old phishing patterns become dated reference
- New obfuscation techniques appear faster than data collection
- Brand portfolios change (mergers, product discontinuation)

**Impact**: Model accuracy degrades as time progresses without retraining
**Baseline**: Time-based OOD evaluation (separate train/test by time) measures this drift

---

## 5. Deployment Recommendations

### 5.1 Responsible Deployment Checklist

- [ ] **Do NOT use as sole phishing defense.** Combine with other signals (page content analysis, user education, email authentication).
- [ ] **Monitor false positive rate in production.** Set alerts if FPR > 2% (legitimate users may bypass, reducing security).
- [ ] **Retrain quarterly or when concept drift detected.** Use production feedback to identify new phishing patterns.
- [ ] **Maintain human review queue (abstain threshold).** Direct uncertain predictions (p < 0.6) to analysts.
- [ ] **Log all predictions and outcomes.** Use feedback to detect model degradation and retrain.
- [ ] **Validate on brand-specific OOD data.** Ensure model generalizes to your user population.
- [ ] **Gracefully handle enrichment failures.** DNS/API unavailability should not cascade to incorrect predictions.

### 5.2 Abstention Strategy (Review Queue)

**Recommendation:** 
- Set abstention threshold at min_confidence=0.6 (model confidence < 60%)
- Route uncertain URLs to analyst review instead of auto-blocking
- **Impact (measured):** Precision increases from 0.36 → 0.82 when abstaining on uncertain URLs

**Rationale:**
- Analysts can assess context (user profile, email sender, page design)
- URL alone insufficient; human judgment adds high-value signal
- Trade-off: Higher operational cost vs. lower false positive rate

---

## 6. Out-of-Scope Threats

**Threat 1: SSL/TLS Certificate Fraud**
- Gojo only analyzes URL; does not validate certificate chain
- Attacker-issued certificates (self-signed, fraudulent CAs) not detected
- **Mitigation**: Enforced HTTPS validation at reverse proxy layer

**Threat 2: Spear-Phishing**
- Targeted phishing emails to specific individuals (highly context-dependent)
- May contain legitimate-looking URLs crafted for that individual
- **Mitigation**: Email authentication (DMARC/SPF) + sender reputation

**Threat 3: Watering-Hole Attacks**
- Legitimate website compromised to serve phishing payload
- URL looks identical to legitimate site (because it IS the legitimate site)
- **Mitigation**: Endpoint detection and response (EDR) + user awareness training

**Threat 4: Credential Harvesting Forms**
- Form analysis not performed (URL-only detection)
- Phishing pages may embed legitimate-looking forms
- **Mitigation**: Email client sandboxing + user training on secure input practices

---

## 7. Evaluation Caveats

### 7.1 Benchmark Limitations

**Caveat 1: Closed-World Assumption**
- Evaluation assumes all URLs are labeled (phishing or benign)
- Real-world includes unknown URLs (1st-seen, not yet classified)
- **Impact**: Recall rates in evaluation may overestimate real-world performance

**Caveat 2: Temporal Distribution**
- Evaluation uses historical data; real phishing campaigns are ongoing
- New obfuscation techniques not represented in training data
- **Impact**: Evaluation accuracy may not reflect steady-state production performance

**Caveat 3: Brand-Specific Performance**
- Model trained on aggregated brands; brand-specific recall may vary
- High-value targets (Amazon, Google) may have disproportionate recall
- Budget brands may have low recall
- **Impact**: Do not assume uniform protection across all brands

---

## 8. Data Poisoning and Feedback Abuse

- Attackers may submit poisoned URLs to skew feedback rewards
- Feedback labels can be spoofed to induce model drift
- **Mitigations:**
  - Off-policy evaluation with IPS/SNIPS/DR before deployment
  - Guardrails on policy weight shifts and confidence thresholds
  - Label-quality scoring and review queues

---

## 9. Failure Mode Taxonomy

- False negatives for brand-typo phishing
- False positives on benign lookalikes
- Domain-family leakage across splits
- Concept drift after attacker behavior changes
- Enrichment service unavailability (graceful degradation)

---

## 10. Red-Team Evaluation

- Synthetic perturbation suite (typo, transposition, homoglyph, URL encoding, deep subdomain)
- Attack success rate and score degradation metrics
- Required pass criteria before rollout updates

---

**Document Version**: 3.0  
**Last Updated**: 2026-02-13  
**Maintained By**: Gojo Security Team

---

*This document should be reviewed quarterly and updated as new threat intelligence emerges.*
