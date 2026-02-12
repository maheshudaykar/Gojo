# Dataset Card

## Summary
- Labeled URLs for phishing vs benign detection.
- Required columns: url, label (optional time).

## Collection and Labeling
- Sources: open datasets and curated feeds.
- Labels normalized to {phish, legit}.

## Quality Controls
- Deduplication by normalized URL.
- Domain-family leakage control via registrable domain splits.
- Label-quality scoring and review queue for disagreements.

## Known Biases
- Over-representation of popular brands.
- Class imbalance may vary over time.
- Shorteners and deep subdomains are under-sampled in some sources.

## Recommended Splits
- Time-based split when timestamps exist.
- Family split to avoid leakage across train/test.

## Maintenance
- Refresh frequency: quarterly or when drift detected.
- Archive older snapshots for reproducibility.
