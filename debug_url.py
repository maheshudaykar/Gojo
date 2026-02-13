#!/usr/bin/env python
"""Debug script to diagnose false negatives."""

from phish_detector.parsing import parse_url
from phish_detector.rules import evaluate_rules
from phish_detector.analyze import analyze_url, AnalysisConfig
from phish_detector.brand_risk import BrandRiskConfig

# Test URL
test_url = "http://www.roblox.com.ml/users/3531589541/profile/"

print(f"Debugging URL: {test_url}\n")
print("=" * 80)

# Step 1: URL Parsing
print("\n[1] URL PARSING")
print("-" * 80)
try:
    parsed = parse_url(test_url)
    print(f"OK - Parsed successfully")
    print(f"   Original: {parsed.original}")
    print(f"   Normalized: {parsed.normalized}")
    print(f"   Scheme: {parsed.scheme}")
    print(f"   Host: {parsed.host}")
    print(f"   Path: {parsed.path}")
except Exception as e:
    print(f"ERROR - Parsing error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Step 2: Feature Extraction & Rule Evaluation
print("\n[2] FEATURE EXTRACTION & RULE EVALUATION")
print("-" * 80)
try:
    features, rule_hits = evaluate_rules(parsed)
    print(f"OK - Extracted {len(features)} features and evaluated rules")
    print(f"\nKey features:")
    for k in ['url_length', 'host_length', 'tld', 'is_suspicious_tld', 'host_entropy', 
              'num_subdomains', 'has_homoglyph', 'has_at_symbol', 'has_ip']:
        val = features.get(k, 'N/A')
        if isinstance(val, float):
            print(f"   - {k}: {val:.2f}")
        else:
            print(f"   - {k}: {val}")
    
    print(f"\n   Rules triggered: {len(rule_hits)}")
    if rule_hits:
        total_rule_score = 0
        for hit in rule_hits:
            print(f"     * {hit.name:20} | Weight: {hit.weight:3} | {hit.details}")
            total_rule_score += hit.weight
        print(f"\n   Total rule score: {total_rule_score}")
    else:
        print("   WARNING - NO RULES TRIGGERED!")
        
except Exception as e:
    print(f"ERROR - Rule evaluation error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Step 3: Full Pipeline Analysis
print("\n[3] FULL DETECTION PIPELINE")
print("-" * 80)
try:
    config = AnalysisConfig(
        ml_mode="ensemble",
        lexical_model="models/lexical_model.joblib",
        char_model="models/char_model.joblib",
        policy_path="models/policy.json",
        feedback_store="models/feedback.json",
        shadow_learn=False,
        enable_enrichment=False,  # Disable enrichment for clearer debugging
        brand_risk=BrandRiskConfig()
    )
    
    result, metadata = analyze_url(test_url, config)
    
    print(f"OK - Full analysis complete:\n")
    print(f"Result keys: {list(result.keys())}\n")
    print(f"Scoring breakdown:")
    print(f"   - Rule Score: {result.get('rule_score', 'N/A')}")
    print(f"   - ML Score: {result.get('ml_score', 'N/A')}")
    print(f"   - Final Score: {result.get('score', 'N/A')}")
    print(f"   - Confidence: {result.get('confidence', 'N/A')}")
    print(f"   - Summary: {result.get('summary', {})}")
    
    verdict = result.get('summary', {}).get('label', 'UNKNOWN')
    print(f"\nVERDICT: {verdict}")
    
    if verdict.upper() != "PHISHING" and verdict != "red":
        print(f"\nWARNING - False negative detected!")
        print(f"   Expected: RED/PHISHING")
        print(f"   Actual: {verdict}")
        
    signals = result.get('signals', [])
    print(f"\nDetection signals: {len(signals)}")
    for sig in signals:
        print(f"   * {sig.get('name')}: weight={sig.get('weight')}")
    
except Exception as e:
    print(f"ERROR - Analysis error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("Debugging complete")
