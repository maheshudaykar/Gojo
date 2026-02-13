"""
Drift detection and adaptive learning for evolving phishing attacks.

This module monitors distribution shifts and adapts the detection policy
to handle new attack patterns not seen during training.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from datetime import datetime, timezone
from typing import Dict, List, Optional
import json
from pathlib import Path


@dataclass
class DriftSignal:
    """Detected distribution drift signal."""
    metric: str
    baseline_value: float
    current_value: float
    drift_magnitude: float
    timestamp: str
    triggered: bool


@dataclass
class AdaptiveConfig:
    """Configuration for adaptive learning."""
    drift_detection_window: int = 1000  # URLs to analyze per window
    psi_threshold: float = 0.2  # Population Stability Index threshold
    min_confidence_for_update: float = 0.6
    retraining_trigger_threshold: int = 5  # Drift signals before retraining
    feature_drift_tracking: bool = True
    attack_pattern_memory: int = 10000  # Remember last N attack patterns


@dataclass
class AttackPattern:
    """Represents a detected attack pattern."""
    pattern_id: str
    first_seen: str
    last_seen: str
    frequency: int
    indicators: Dict[str, Any]  # Fixed: changed 'any' to 'Any'
    tld: Optional[str] = None
    brand_target: Optional[str] = None
    technique: Optional[str] = None  # e.g., "typosquat", "homoglyph", "subdomain_abuse"


class DriftDetector:
    """Monitors feature distributions and detects concept drift."""
    
    def __init__(self, config: AdaptiveConfig):
        self.config = config
        self.baseline_distributions: Dict[str, Dict[str, float]] = {}
        self.current_window: List[Dict[str, Any]] = []
        self.drift_signals: List[DriftSignal] = []
        
    def update_baseline(self, features: Dict[str, float]) -> None:
        """Update baseline feature distributions."""
        for feature, value in features.items():
            # Skip non-numeric features (TLD strings, URLs, etc.)
            if not isinstance(value, (int, float, bool)):
                continue
            
            # Convert bool to int for numeric calculations
            if isinstance(value, bool):
                value = int(value)
            
            if feature not in self.baseline_distributions:
                self.baseline_distributions[feature] = {
                    'sum': 0.0,
                    'sum_squared': 0.0,
                    'count': 0,
                    'mean': 0.0,
                    'variance': 0.0
                }
            
            stats = self.baseline_distributions[feature]
            stats['count'] += 1
            stats['sum'] += value
            stats['sum_squared'] += value ** 2
            stats['mean'] = stats['sum'] / stats['count']
            
            if stats['count'] > 1:
                stats['variance'] = (
                    stats['sum_squared'] / stats['count'] - stats['mean'] ** 2
                )
    
    def detect_drift(self, features: Dict[str, float]) -> List[DriftSignal]:
        """
        Detect drift using Population Stability Index (PSI).
        
        PSI = Σ (actual% - expected%) × ln(actual% / expected%)
        PSI < 0.1: No significant change
        0.1 ≤ PSI < 0.2: Moderate change
        PSI ≥ 0.2: Significant drift detected
        """
        self.current_window.append(features)
        
        if len(self.current_window) < self.config.drift_detection_window:
            return []
        
        signals = []
        
        for feature, baseline in self.baseline_distributions.items():
            if baseline['count'] < 10:  # Need minimum baseline
                continue
            
            # Calculate current window stats (filter numeric values)
            current_values = [
                (int(w[feature]) if isinstance(w.get(feature), bool) else w.get(feature, 0.0))
                for w in self.current_window
                if isinstance(w.get(feature), (int, float, bool))
            ]
            
            if not current_values:
                continue
                
            current_mean = sum(current_values) / len(current_values)
            
            # Calculate PSI (simplified version)
            if baseline['mean'] > 0:
                psi = abs((current_mean - baseline['mean']) / baseline['mean'])
            else:
                psi = abs(current_mean)
            
            if psi >= self.config.psi_threshold:
                signal = DriftSignal(
                    metric=feature,
                    baseline_value=baseline['mean'],
                    current_value=current_mean,
                    drift_magnitude=psi,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    triggered=True
                )
                signals.append(signal)
                self.drift_signals.append(signal)
        
        # Reset window
        self.current_window = []
        
        return signals


class AttackPatternTracker:
    """Tracks and learns from emerging attack patterns."""
    
    def __init__(self, storage_path: Path, config: AdaptiveConfig):
        self.storage_path = storage_path
        self.config = config
        self.patterns: Dict[str, AttackPattern] = {}
        self.load_patterns()
    
    def load_patterns(self) -> None:
        """Load attack patterns from disk."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                    for pattern_data in data.get('patterns', []):
                        pattern = AttackPattern(**pattern_data)
                        self.patterns[pattern.pattern_id] = pattern
            except (json.JSONDecodeError, IOError):
                pass
    
    def save_patterns(self) -> None:
        """Persist attack patterns to disk."""
        data = {
            'patterns': [
                {
                    'pattern_id': p.pattern_id,
                    'first_seen': p.first_seen,
                    'last_seen': p.last_seen,
                    'frequency': p.frequency,
                    'indicators': p.indicators,
                    'tld': p.tld,
                    'brand_target': p.brand_target,
                    'technique': p.technique
                }
                for p in self.patterns.values()
            ],
            'last_updated': datetime.now(timezone.utc).isoformat()
        }
        
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.storage_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def record_attack(
        self,
        url: str,
        features: Dict[str, Any],
        tld: Optional[str] = None,
        brand: Optional[str] = None,
        technique: Optional[str] = None
    ) -> str:
        """
        Record a detected phishing attack pattern.
        
        Returns:
            Pattern ID
        """
        # Generate pattern signature
        pattern_id = self._generate_pattern_id(features, tld, brand, technique)
        
        now = datetime.now(timezone.utc).isoformat()
        
        if pattern_id in self.patterns:
            # Update existing pattern
            pattern = self.patterns[pattern_id]
            pattern.last_seen = now
            pattern.frequency += 1
        else:
            # Create new pattern
            pattern = AttackPattern(
                pattern_id=pattern_id,
                first_seen=now,
                last_seen=now,
                frequency=1,
                indicators=features,
                tld=tld,
                brand_target=brand,
                technique=technique
            )
            self.patterns[pattern_id] = pattern
        
        # Prune old patterns (keep most recent)
        if len(self.patterns) > self.config.attack_pattern_memory:
            sorted_patterns = sorted(
                self.patterns.items(),
                key=lambda x: x[1].last_seen,
                reverse=True
            )
            self.patterns = dict(sorted_patterns[:self.config.attack_pattern_memory])
        
        self.save_patterns()
        return pattern_id
    
    def _generate_pattern_id(
        self,
        features: Dict[str, Any],
        tld: Optional[str],
        brand: Optional[str],
        technique: Optional[str]
    ) -> str:
        """Generate unique pattern identifier."""
        components = []
        
        if technique:
            components.append(technique)
        if tld:
            components.append(f"tld_{tld}")
        if brand:
            components.append(f"brand_{brand}")
        
        # Add feature fingerprint
        feature_sig = "_".join([
            f"{k}:{int(v) if isinstance(v, (int, float)) else v}"
            for k, v in sorted(features.items())
            if k in ['is_suspicious_tld', 'has_homoglyph', 'num_subdomains']
        ])
        components.append(feature_sig)
        
        return "_".join(components)[:128]
    
    def get_emerging_patterns(self, min_frequency: int = 5, days: int = 7) -> List[AttackPattern]:
        """Get recently emerging attack patterns."""
        from datetime import timedelta
        
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        
        emerging = [
            pattern for pattern in self.patterns.values()
            if pattern.frequency >= min_frequency
            and datetime.fromisoformat(pattern.first_seen) >= cutoff
        ]
        
        return sorted(emerging, key=lambda x: x.frequency, reverse=True)
