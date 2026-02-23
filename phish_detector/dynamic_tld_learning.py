"""
Dynamic TLD and regional pattern learning.

Automatically learns suspicious TLDs and regional phishing patterns
from production traffic instead of relying on static lists.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Set
import json


@dataclass
class TLDStats:
    """Statistics for a specific TLD."""
    tld: str
    total_seen: int
    phishing_count: int
    legitimate_count: int
    phishing_rate: float
    first_seen: str
    last_seen: str
    is_emerging: bool
    is_regional: bool
    region: str | None


class DynamicTLDLearner:
    """
    Learns suspicious TLDs dynamically from production feedback.

    Instead of static lists, this tracks TLD patterns in real-time
    and flags new suspicious TLDs as they emerge.
    """

    def __init__(self, storage_path: Path):
        self.storage_path = storage_path
        self.tld_stats: Dict[str, TLDStats] = {}
        self.suspicious_threshold = 0.7  # 70% phishing rate
        self.emerging_window_days = 30
        self.min_observations = 10  # Minimum URLs needed for confidence
        self.load_stats()

    def load_stats(self) -> None:
        """Load TLD statistics from disk."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                    for tld, stats_dict in data.get('tld_stats', {}).items():
                        self.tld_stats[tld] = TLDStats(**stats_dict)
            except (json.JSONDecodeError, IOError):
                pass

    def save_stats(self) -> None:
        """Persist TLD statistics to disk."""
        data: dict[str, Any] = {
            'tld_stats': {
                tld: {
                    'tld': stats.tld,
                    'total_seen': stats.total_seen,
                    'phishing_count': stats.phishing_count,
                    'legitimate_count': stats.legitimate_count,
                    'phishing_rate': stats.phishing_rate,
                    'first_seen': stats.first_seen,
                    'last_seen': stats.last_seen,
                    'is_emerging': stats.is_emerging,
                    'is_regional': stats.is_regional,
                    'region': stats.region
                }
                for tld, stats in self.tld_stats.items()
            },
            'last_updated': datetime.now(timezone.utc).isoformat()
        }

        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.storage_path, 'w') as f:
            json.dump(data, f, indent=2)

    def update(self, tld: str, is_phishing: bool, region: str | None = None) -> None:
        """
        Update TLD statistics with new observation.

        Args:
            tld: Top-level domain (e.g., 'ml', 'com', 'xyz')
            is_phishing: Whether this URL was phishing
            region: Geographic region if known (e.g., 'africa', 'asia', 'europe')
        """
        now = datetime.now(timezone.utc).isoformat()

        if tld not in self.tld_stats:
            self.tld_stats[tld] = TLDStats(
                tld=tld,
                total_seen=0,
                phishing_count=0,
                legitimate_count=0,
                phishing_rate=0.0,
                first_seen=now,
                last_seen=now,
                is_emerging=True,
                is_regional=region is not None,
                region=region
            )

        stats = self.tld_stats[tld]
        stats.total_seen += 1
        stats.last_seen = now

        if is_phishing:
            stats.phishing_count += 1
        else:
            stats.legitimate_count += 1

        stats.phishing_rate = stats.phishing_count / stats.total_seen

        # Mark as emerging if first seen within window
        first_seen_dt = datetime.fromisoformat(stats.first_seen)
        cutoff = datetime.now(timezone.utc) - timedelta(days=self.emerging_window_days)
        stats.is_emerging = first_seen_dt >= cutoff

        if region and not stats.region:
            stats.region = region

        self.save_stats()

    def is_suspicious(self, tld: str) -> bool:
        """
        Check if TLD should be flagged as suspicious.

        Returns True if:
        - Phishing rate >= threshold AND
        - Sufficient observations for statistical confidence
        """
        if tld not in self.tld_stats:
            return False

        stats = self.tld_stats[tld]

        if stats.total_seen < self.min_observations:
            return False

        return stats.phishing_rate >= self.suspicious_threshold

    def get_suspicious_tlds(
        self,
        include_emerging: bool = True,
        min_phishing_rate: float = 0.6
    ) -> Set[str]:
        """
        Get set of suspicious TLDs based on learned patterns.

        Args:
            include_emerging: Include newly emerging TLDs
            min_phishing_rate: Minimum phishing rate to consider suspicious

        Returns:
            Set of suspicious TLD strings
        """
        suspicious: set[str] = set()

        for tld, stats in self.tld_stats.items():
            if stats.total_seen < self.min_observations:
                continue

            if stats.phishing_rate >= min_phishing_rate:
                suspicious.add(tld)

            # Include emerging TLDs with lower threshold
            if include_emerging and stats.is_emerging:
                if stats.phishing_rate >= 0.5 and stats.total_seen >= 5:
                    suspicious.add(tld)

        return suspicious

    def get_emerging_threats(self) -> List[TLDStats]:
        """Get TLDs that are emerging threats."""
        emerging = [
            stats for stats in self.tld_stats.values()
            if stats.is_emerging
            and stats.phishing_rate >= 0.5
            and stats.total_seen >= 5
        ]

        return sorted(emerging, key=lambda x: x.phishing_rate, reverse=True)

    def get_regional_patterns(self) -> Dict[str, List[TLDStats]]:
        """Group suspicious TLDs by region."""
        regional: Dict[str, List[TLDStats]] = {}

        for stats in self.tld_stats.values():
            if stats.is_regional and stats.region and stats.phishing_rate >= 0.5:
                if stats.region not in regional:
                    regional[stats.region] = []
                regional[stats.region].append(stats)

        return regional

    def merge_with_static_list(self, static_tlds: Set[str]) -> Set[str]:
        """
        Combine dynamically learned TLDs with static baseline.

        Args:
            static_tlds: Baseline suspicious TLDs from config

        Returns:
            Union of static and learned TLDs
        """
        learned = self.get_suspicious_tlds()
        return static_tlds | learned

    def generate_report(self) -> Dict[str, Any]:
        """Generate human-readable report of TLD patterns."""
        return {
            'total_tlds_tracked': len(self.tld_stats),
            'suspicious_tlds': list(self.get_suspicious_tlds()),
            'emerging_threats': [
                {
                    'tld': s.tld,
                    'phishing_rate': f"{s.phishing_rate:.2%}",
                    'observations': s.total_seen,
                    'first_seen': s.first_seen
                }
                for s in self.get_emerging_threats()
            ],
            'regional_patterns': {
                region: [s.tld for s in tlds]
                for region, tlds in self.get_regional_patterns().items()
            }
        }
