# Git LFS Migration Guide

## Overview

This guide explains how to migrate large dataset files to Git LFS (Large File Storage) when repository size becomes an issue.

## Current Status

Repository size: 11.5 MB (including OOD datasets)

Large files currently:
- data/ood/phishingdb_ood.csv: 450 KB
- data/ood/ood_combined.csv: 20 KB

Recommendation: Not needed yet. Activate LFS when:
- Repository size exceeds 100 MB
- Individual files exceed 50 MB
- Scaling to 1M+ URLs

## Setup Instructions

### 1. Install Git LFS

Windows: choco install git-lfs
macOS: brew install git-lfs
Linux: sudo apt-get install git-lfs

### 2. Initialize

```bash
git lfs install
```

### 3. Configure File Types

Update .gitattributes with:
```
data/ood/*.csv filter=lfs diff=lfs merge=lfs -text
models/*.pkl filter=lfs diff=lfs merge=lfs -text
```

### 4. Migrate Files

```bash
git lfs migrate import --include="data/ood/*.csv" --everything
git push origin --force-with-lease
```

## For Manuscript Submission

1. Keep code in GitHub
2. Upload data to Zenodo (permanent DOI)
3. Link in README with DOI
4. Provide download and validation scripts

## Reference

GitHub LFS Storage Limits:
- Free: 1 GB total
- Pro: 50 GB total

Scaling estimates:
- 100k URLs: 5 MB (no LFS)
- 1M URLs: 50 MB (LFS recommended)
- 10M URLs: 500 MB (LFS required)

---

Version: 1.0
Last Updated: 2026-02-13
