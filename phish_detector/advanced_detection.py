"""
Integration layer for advanced phishing detection features.

Combines:
- Content-based HTML analysis
- Drift detection and adaptive learning
- Dynamic TLD learning
- Attack pattern tracking
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Any
from phish_detector.content_analysis import ContentAnalysis, fetch_and_analyze
from phish_detector.drift_detection import (
    DriftDetector,
    AttackPatternTracker,
    AdaptiveConfig
)
from phish_detector.dynamic_tld_learning import DynamicTLDLearner


class AdvancedDetector:
    """
    Enhanced detector with content analysis and adaptive learning.

    Usage:
        detector = AdvancedDetector("models")
        result = detector.analyze_url_enhanced(
            url="http://suspicious-site.ml",
            perform_content_analysis=True,
            features={'tld': 'ml', 'num_subdomains': 2}
        )
    """

    def __init__(self, model_dir: Path | str):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.config = AdaptiveConfig()
        self.drift_detector = DriftDetector(self.config)
        self.attack_tracker = AttackPatternTracker(
            self.model_dir / "attack_patterns.json",
            self.config
        )
        self.tld_learner = DynamicTLDLearner(
            self.model_dir / "dynamic_tlds.json"
        )

        # Load baseline from first N URLs
        self._baseline_initialized = False
        self._baseline_count = 0
        self._baseline_target = 500

    def analyze_url_enhanced(
        self,
        url: str,
        features: Dict[str, Any],
        ml_score: float,
        rule_score: int,
        perform_content_analysis: bool = False,
        is_phishing: Optional[bool] = None,
        brand_target: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Enhanced URL analysis with all advanced features.

        Args:
            url: URL to analyze
            features: Extracted features from URL parsing
            ml_score: ML model score (0-100)
            rule_score: Rule-based score (0-100)
            perform_content_analysis: Whether to fetch and analyze HTML
            is_phishing: Ground truth label (if known, for learning)
            brand_target: Brand being impersonated (if detected)

        Returns:
            Enhanced analysis results
        """
        result: dict[str, Any] = {
            'url': url,
            'base_score': max(ml_score, rule_score),
            'ml_score': ml_score,
            'rule_score': rule_score,
            'enhancements': {}
        }

        # 1. Content-based analysis (if enabled)
        content_analysis: Optional[ContentAnalysis] = None
        if perform_content_analysis:
            content_analysis = fetch_and_analyze(url)
            if content_analysis:
                result['enhancements']['content'] = {
                    'has_credential_form': content_analysis.has_credential_form,
                    'suspicious_js_count': content_analysis.suspicious_js_count,
                    'external_form_action': content_analysis.external_form_action,
                    'brand_visual_match': content_analysis.brand_visual_match,
                    'content_risk_score': content_analysis.risk_score,
                    'signals': content_analysis.signals
                }

                # Boost score if content analysis detects high risk
                if content_analysis.risk_score >= 60:
                    result['base_score'] = min(100, result['base_score'] + 20)
                    result['enhancements']['content']['score_boost'] = 20

        # 2. Drift detection
        if self._baseline_initialized:
            drift_signals = self.drift_detector.detect_drift(features)
            if drift_signals:
                result['enhancements']['drift'] = [
                    {
                        'metric': s.metric,
                        'magnitude': s.drift_magnitude,
                        'baseline': s.baseline_value,
                        'current': s.current_value
                    }
                    for s in drift_signals
                ]
        else:
            # Build baseline
            self.drift_detector.update_baseline(features)
            self._baseline_count += 1
            if self._baseline_count >= self._baseline_target:
                self._baseline_initialized = True

        # 3. Dynamic TLD learning
        tld = features.get('tld', '')
        if tld and is_phishing is not None:
            # Region detection (simplified - could use GeoIP)
            region = self._detect_region(tld)
            self.tld_learner.update(tld, is_phishing, region)

        if tld:
            is_tld_suspicious = self.tld_learner.is_suspicious(tld)
            result['enhancements']['dynamic_tld'] = {
                'tld': tld,
                'is_learned_suspicious': is_tld_suspicious
            }

            # Boost score if dynamic learning flags TLD
            if is_tld_suspicious and not features.get('is_suspicious_tld', False):
                result['base_score'] = min(100, result['base_score'] + 15)
                result['enhancements']['dynamic_tld']['score_boost'] = 15

        # 4. Attack pattern tracking
        if is_phishing or (result['base_score'] >= 60):
            technique = self._identify_technique(features)
            pattern_id = self.attack_tracker.record_attack(
                url=url,
                features=features,
                tld=tld,
                brand=brand_target,
                technique=technique
            )
            result['enhancements']['attack_pattern'] = {
                'pattern_id': pattern_id,
                'technique': technique
            }

        # 5. Final verdict with enhancements
        final_score = result['base_score']
        result['final_score'] = final_score
        result['verdict'] = 'red' if final_score > 60 else ('yellow' if final_score > 25 else 'green')

        return result

    def _detect_region(self, tld: str) -> Optional[str]:
        """
        Detect geographic region from TLD (simplified).

        In production, use MaxMind GeoIP or similar.
        """
        regional_tlds = {
            'africa': {'za', 'ng', 'ke', 'gh', 'tz', 'ug', 'ml', 'ga', 'cf'},
            'asia': {'cn', 'jp', 'in', 'kr', 'sg', 'hk', 'my', 'th', 'vn', 'id'},
            'europe': {'de', 'uk', 'fr', 'it', 'es', 'nl', 'pl', 'ru', 'ch'},
            'latin_america': {'br', 'mx', 'ar', 'cl', 'co', 'pe'},
            'middle_east': {'sa', 'ae', 'il', 'tr', 'eg'},
        }

        for region, tlds in regional_tlds.items():
            if tld in tlds:
                return region

        return None

    def _identify_technique(self, features: Dict[str, Any]) -> Optional[str]:
        """Identify phishing technique from features."""
        if features.get('has_homoglyph'):
            return 'homoglyph_attack'
        if features.get('num_subdomains', 0) >= 3:
            return 'subdomain_abuse'
        if features.get('is_suspicious_tld'):
            return 'suspicious_tld_abuse'
        if features.get('has_encoded_chars'):
            return 'url_obfuscation'
        if features.get('has_ip'):
            return 'ip_based_hosting'
        return 'unknown'

    def get_adaptive_insights(self) -> Dict[str, Any]:
        """
        Generate insights for model adaptation and retraining.

        Returns:
            Dictionary with drift signals, emerging patterns, and recommendations
        """
        emerging_attacks = self.attack_tracker.get_emerging_patterns(
            min_frequency=5,
            days=7
        )

        tld_report = self.tld_learner.generate_report()

        recommendations: list[dict[str, Any]] = []

        # Recommend retraining if significant drift
        if len(self.drift_detector.drift_signals) >= self.config.retraining_trigger_threshold:
            recommendations.append({
                'action': 'retrain_model',
                'reason': f'{len(self.drift_detector.drift_signals)} drift signals detected',
                'priority': 'high'
            })

        # Recommend updating TLD list
        if len(tld_report['emerging_threats']) > 0:
            recommendations.append({
                'action': 'update_suspicious_tlds',
                'reason': f"{len(tld_report['emerging_threats'])} emerging threat TLDs",
                'priority': 'medium',
                'new_tlds': [t['tld'] for t in tld_report['emerging_threats']]
            })

        return {
            'drift_signals': len(self.drift_detector.drift_signals),
            'emerging_attack_patterns': [
                {
                    'pattern_id': p.pattern_id,
                    'frequency': p.frequency,
                    'technique': p.technique,
                    'brand_target': p.brand_target,
                    'first_seen': p.first_seen
                }
                for p in emerging_attacks
            ],
            'tld_insights': tld_report,
            'recommendations': recommendations
        }
