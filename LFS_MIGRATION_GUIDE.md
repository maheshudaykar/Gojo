# Git LFS Migration Guide

## Overview

This guide explains how to migrate large dataset files to Git LFS (Large File Storage) when repository size becomes an issue.

## Current Status

**Repository size**: ~11.5 MB (including OOD datasets)

**Large files**:
- `data/ood/phishingdb_ood.csv`: ~450 KB
- `data/ood/ood_combined.csv`: ~20 KB
- `data/ood/phishingdb_ood_sources.json`, `ood_sources.json`: ~2 KB

**Recommendation**: Not needed yet. Activate LFS when:
- Repository size exceeds 100 MB
- Individual dataset files exceed 50 MB
- Scaling to 1M+ URL benchmarks

---

## Setup Instructions

### 1. Install Git LFS

**Windows (via Chocolatey):**
```powershell
choco install git-lfs
```

**Windows (via Scoop):**
```powershell
scoop install git-lfs
```

**macOS (via Homebrew):**
```bash
brew install git-lfs
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get install git-lfs
```

### 2. Initialize LFS

```bash
cd c:\Mahi\Project\Gojo
git lfs install
```

This updates your `.git/config` to reference the LFS interface.

### 3. Configure Tracked File Types

Update `.gitattributes`:

```
# Large CSV datasets
data/ood/*.csv filter=lfs diff=lfs merge=lfs -text

# Future: Large model files
models/*.pkl filter=lfs diff=lfs merge=lfs -text
models/*.joblib filter=lfs diff=lfs merge=lfs -text

# Future: Large benchmark results
results/*.json filter=lfs diff=lfs merge=lfs -text
```

### 4. Migrate Existing Files (if needed)

```bash
# Migrate CSV files
git lfs migrate import --include="data/ood/*.csv" --everything

# Migrate specific file
git lfs migrate import --include="data/ood/phishingdb_ood.csv"

# Push migrated changes
git push origin --force-with-lease
```

**⚠️ Warning**: `--force-with-lease` rewrites history. Coordinate with team before pushing.

### 5. Verify LFS Setup

```bash
# Check LFS objects
git lfs ls-files

# Expected output:
# 9b1798ba80249bba39f4b3706850a68aab46cead78af642b921bc5f1c848cb0 - data/ood/phishingdb_ood.csv
```

---

## Updating .gitignore (Optional)

To prevent accidental commits of large files:

```
# Large datasets (if not using LFS)
*.csv

# Model checkpoints
models/*.pkl
models/*.joblib

# Benchmark results
results/*.json
```

---

## Usage After LFS Migration

After LFS is enabled, workflows remain identical:

```powershell
# Clone (LFS files auto-downloaded)
git clone https://github.com/your-org/gojo.git

# Push LFS files (automatic)
git add data/ood/phishingdb_ood.csv
git commit -m "Add large OOD dataset"
git push

# Pull (automatic LFS download)
git pull
```

---

## LFS Storage Quota

**GitHub Free Plan**: 1 GB total LFS storage (across all repos in org)
**GitHub Pro Plan**: 50 GB total LFS storage

**Cost estimation for scaling**:
- 100k URLs benchmark: ~5 MB (no LFS needed)
- 1M URLs benchmark: ~50 MB (LFS recommended)
- 10M URLs benchmark: ~500 MB (LFS required)

---

## Troubleshooting

### Issue: "This repository is over its data quota"

**Solution**: Delete old benchmark runs to free space:

```bash
git filter-branch --tree-filter 'rm -f results/old_run_*.json' -- --all
git push origin --force-with-lease
```

### Issue: "LFS: Authentication required"

**Solution**: Configure credentials:

```bash
git lfs install
git config --global credential.helper wincred  # Windows
git config --global credential.helper osxkeychain  # macOS
git config --global credential.helper cache  # Linux
```

### Issue: "Git LFS is not installed"

**Solution**: Re-run installation commands above, then:

```bash
git lfs install --force
```

---

## Alternative: Release Artifacts

Instead of LFS, consider GitHub Releases:

```bash
# Create release archive
tar -czf gojo-v1.0-ood-datasets.tar.gz data/ood/

# Upload via GitHub CLI
gh release create v1.0 gojo-v1.0-ood-datasets.tar.gz \
  --title "Gojo v1.0 with OOD Datasets" \
  --notes "Pre-built OOD datasets for benchmarking"

# Users download from Release page
```

**Pros**:
- No storage quota limits
- Versioned datasets
- Clear separation of code/data

**Cons**:
- Manual download step required
- Not integrated into `git clone`

---

## Recommendations for Manuscript Submission

When submitting paper with code:

1. **Keep code in GitHub** (public repo)
2. **Upload data to Zenodo** (permanent DOI)
3. **Link data in README**:

```markdown
## Data

OOD evaluation datasets available at:
- [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.xxxxx.svg)](https://doi.org/10.5281/zenodo.xxxxx)

Download: `wget https://zenodo.org/record/xxxxx/files/gojo-ood-datasets.tar.gz`
```

4. **Reproducibility**: Provide simple script to download and validate:

```bash
#!/bin/bash
# scripts/download_ood_datasets.sh

URL="https://zenodo.org/record/xxxxx/files/gojo-ood-datasets.tar.gz"
SHA256="9b1798ba80249bba39f4b3706850a68aab46cead78af642b921bc5f1c848cb0"

wget $URL -O ood-datasets.tar.gz
echo "$SHA256  ood-datasets.tar.gz" | sha256sum -c - || exit 1
tar -xzf ood-datasets.tar.gz
rm ood-datasets.tar.gz
```

---

**Document Version**: 1.0  
**Last Updated**: 2026-02-13
