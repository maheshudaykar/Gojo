"""
Quick test to verify advanced detection integration.
Run: python test_integration.py
"""
from typing import Any
from phish_detector.analyze import AnalysisConfig, analyze_url

def test_basic_integration():
    """Test that advanced features can be enabled without errors."""
    print("Testing Advanced Detection Integration...")
    print("=" * 60)
    
    # Test URL (known phishing from dataset)
    test_url = "http://www.roblox.com.ml/users/3531589541/profile/"
    
    # Configuration with advanced features
    config = AnalysisConfig(
        ml_mode="ensemble",
        lexical_model="models/lexical_model.joblib",
        char_model="models/char_model.joblib",
        policy_path="models/policy.json",
        feedback_store="models/feedback.json",
        enable_content_analysis=False,  # Disable for quick test (no network)
        enable_advanced_detection=True,  # Enable drift/TLD learning
        models_dir="models",
    )
    
    print(f"\n🔍 Analyzing: {test_url}")
    print(f"   Content Analysis: {'Enabled' if config.enable_content_analysis else 'Disabled (no network fetch)'}")
    print(f"   Advanced Detection: {'Enabled' if config.enable_advanced_detection else 'Disabled'}")
    
    try:
        report, extra = analyze_url(test_url, config)
        
        print(f"\n📊 Results:")
        print(f"   Score: {report['summary']['score']}")
        print(f"   Verdict: {report['summary']['label'].upper()}")
        print(f"   Decision: {report['summary']['decision'].upper()}")
        
        if extra.get('content_analysis'):
            ca = extra['content_analysis']
            print(f"\n🔍 Content Analysis:")
            print(f"   Risk Score: {ca['content_risk_score']}")
            print(f"   Signals: {', '.join(ca['signals']) if ca['signals'] else 'None'}")
            if ca.get('score_boost'):
                print(f"   Score Boost: +{ca['score_boost']}")
        
        if extra.get('advanced_enhancements'):
            adv = extra['advanced_enhancements']
            print(f"\n⚡ Advanced Enhancements:")
            
            if adv.get('content'):
                print(f"   Content Boost: +{adv['content'].get('score_boost', 0)}")
            
            if adv.get('dynamic_tld'):
                dtld = adv['dynamic_tld']
                print(f"   Dynamic TLD: {dtld.get('tld')} (Suspicious: {dtld.get('is_learned_suspicious')})")
                print(f"   TLD Boost: +{dtld.get('score_boost', 0)}")
            
            if adv.get('attack_pattern'):
                ap = adv['attack_pattern']
                print(f"   Attack Pattern: {ap.get('technique')}")
        
        print(f"\n✅ Integration test PASSED")
        print(f"   Final Score: {extra['rule_score']} (rules) + {extra['ml_score']} (ML) = {report['summary']['score']}")
        
        # Check if score is above threshold
        if report['summary']['score'] >= 60:
            print(f"   ⚠️  Correctly detected as PHISHING")
            return True
        else:
            print(f"   ⚠️  WARNING: Score below threshold (expected >= 60)")
            return False
            
    except Exception as e:
        print(f"\n❌ Integration test FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_config_flags():
    """Verify config flags are properly defined."""
    print("\n\n🔧 Testing Configuration Flags...")
    print("=" * 60)
    
    config = AnalysisConfig(
        ml_mode="ensemble",
        lexical_model="models/lexical_model.joblib",
        char_model="models/char_model.joblib",
        policy_path="models/policy.json",
        feedback_store="models/feedback.json",
        enable_content_analysis=True,
        enable_advanced_detection=True,
        models_dir="models",
    )
    
    checks: list[tuple[str, Any, Any]] = [
        ("enable_content_analysis", config.enable_content_analysis, True),
        ("enable_advanced_detection", config.enable_advanced_detection, True),
        ("models_dir", config.models_dir, "models"),
    ]
    
    all_passed = True
    for name, actual, expected in checks:
        status = "✅" if actual == expected else "❌"
        print(f"{status} {name}: {actual} (expected: {expected})")
        if actual != expected:
            all_passed = False
    
    return all_passed


if __name__ == "__main__":
    print("\n🚀 Advanced Detection Integration Test Suite")
    print("=" * 60)
    
    # Test 1: Config flags
    test1 = test_config_flags()
    
    # Test 2: Basic integration
    test2 = test_basic_integration()
    
    # Summary
    print("\n\n" + "=" * 60)
    print("📋 Test Summary:")
    print("=" * 60)
    print(f"   Config Flags: {'✅ PASS' if test1 else '❌ FAIL'}")
    print(f"   Integration: {'✅ PASS' if test2 else '❌ FAIL'}")
    
    if test1 and test2:
        print("\n🎉 All tests PASSED! Advanced detection is ready.")
    else:
        print("\n⚠️  Some tests failed. Review output above.")
