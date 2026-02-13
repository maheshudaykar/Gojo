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

## Incremental Migration Strategy

If repository size grows, follow this phased approach:

### Phase 1: Monitor (< 50 MB)
- Continue using standard Git
- Track file sizes in CI/CD
- No action required

### Phase 2: Prepare (50-100 MB)
- Set up Git LFS infrastructure
- Migrate datasets to `.gitattributes`
- Test clone speed on slow connections
- Document LFS workflow for team

### Phase 3: Deploy (100-500 MB)
- Migrate all datasets to LFS
- Update CI/CD to handle LFS tracking
- Push to remote with `--force-with-lease`
- Announce migration to collaborators

### Phase 4: Scale (> 500 MB)
- Consider separate data repository with LFS
- Use symlinks or submodules for code repo
- Host datasets on institutional storage (Zenodo, OSF)
- Automate download/validation on clone

## Publication Workflow

**For peer review:**
1. Publish code to GitHub (no LFS needed if < 100 MB)
2. Upload datasets to Zenodo with Creative Commons license
3. Add Zenodo DOI and download script to README
4. Provide `setup.py` dependency for reviewers

**For reproducibility archive:**
1. Generate experiment manifest (timestamps, seeds, hashes)
2. Archive all models and results to OSF
3. Link OSF DOI in paper appendix
4. Include `scripts/verify_publication_infrastructure.py` output in supplementary

## Troubleshooting

**Problem**: Git clone hangs or fails
- Solution: Check LFS credential configuration: `git config --list | grep lfs`
- Verify pointer files exist: `file data/ood/*.csv` should show `ASCII text`

**Problem**: LFS storage quota exceeded
- Solution: Delete old experiment runs from LFS history: `git lfs prune`
- Consider archiving to Zenodo instead of GitHub for long-term storage

**Problem**: Collaborators see pointer files instead of actual data
- Solution: Ensure they have Git LFS installed: `git lfs install`
- Pull with: `git lfs fetch --all && git pull`

---

Version: 1.0
Last Updated: 2026-02-13
