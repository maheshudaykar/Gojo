import sys
import json
from pathlib import Path
from phish_detector.analyze import analyze_url
from phish_detector.benchmark import build_config

def main() -> None:
    print("===========================================")
    print("      GOJO PRACTICAL DEMONSTRATION")
    print("===========================================\n")
    
    config, policy_obj = build_config("rl-v2", enable_enrichment=False)
    
    # We will pass ml_context_dict as unavailable to rely strictly on the rule engine 
    # to demonstrate that Gojo safely falls back and scores without crashing.
    ml_context = {"mode": "ensemble", "available": False}
    
    test_urls = [
        "https://www.google.com/search?q=cybersecurity",
        "http://secure-login-update-paypal-auth.com/verify-account",
        "https://login.microsoftonline.com/",
        "http://bit.ly/3xY2Z1q" # shortened URL, potentially risky
    ]
    
    for url in test_urls:
        print(f"[*] Analyzing URL: {url}")
        report, extra = analyze_url(url, config, ml_context=ml_context, policy=policy_obj)
        
        score = report['summary']['score']
        verdict = report['summary']['label']
        rule_score = extra.get('rule_score', 0)
        
        print(f"    -> Final Score: {score:.2f}/100")
        print(f"    -> Verdict:     {verdict.upper()}")
        print(f"    -> Rule Score:  {rule_score:.2f}")
        for rule in extra.get('signals', []):
            print(f"       - {rule['name']} (+{rule['weight']})")
        print("-" * 45)

if __name__ == '__main__':
    main()
