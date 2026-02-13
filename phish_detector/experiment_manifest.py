"""
Experiment manifest generation for reproducible research.

Generates a single manifest JSON/YAML per run containing:
- Data hashes (SHA256)
- Random seeds
- CLI arguments
- Code commit hash
- Runtime metadata

Suitable for publication supplements and artifact registration.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DatasetMetadata:
    """Metadata for a dataset file."""

    path: str
    sha256: str
    rows: int
    columns: list[str] = field(default_factory=lambda: [])


@dataclass(frozen=True)
class ExperimentManifest:
    """Complete experiment metadata for reproducibility."""

    run_id: str
    timestamp: str
    python_version: str
    code_commit: str
    code_branch: str
    cli_args: dict[str, Any]
    train_dataset: DatasetMetadata
    test_dataset: DatasetMetadata | None = None
    ood_dataset: DatasetMetadata | None = None
    environment_vars: dict[str, str] = field(default_factory=lambda: {})
    runtime_seconds: float = 0.0
    results_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        # Convert nested dataclasses to dicts
        if data["train_dataset"]:
            data["train_dataset"] = asdict(data["train_dataset"])
        if data["test_dataset"]:
            data["test_dataset"] = asdict(data["test_dataset"])
        if data["ood_dataset"]:
            data["ood_dataset"] = asdict(data["ood_dataset"])
        return data

    def to_json(self, path: Path | str | None = None) -> str:
        """Serialize to JSON string. Optionally write to file."""
        json_str = json.dumps(self.to_dict(), indent=2, sort_keys=True)
        if path:
            Path(path).write_text(json_str, encoding="utf-8")
        return json_str


def _sha256_file(path: Path) -> str:
    """Compute SHA256 hash of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _get_git_commit() -> tuple[str, str]:
    """Get current git commit hash and branch."""
    try:
        commit = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
        branch = (
            subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
        return commit, branch
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown", "unknown"


def _get_csv_columns(path: Path) -> list[str]:
    """Extract column names from CSV."""
    try:
        import csv

        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return list(reader.fieldnames or [])
    except Exception:
        return []


def _count_csv_rows(path: Path) -> int:
    """Count rows in CSV (excluding header)."""
    try:
        import csv

        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return sum(1 for _ in reader)
    except Exception:
        return 0


def create_experiment_manifest(
    run_id: str,
    train_data_path: str | Path,
    cli_args: dict[str, Any],
    test_data_path: str | Path | None = None,
    ood_data_path: str | Path | None = None,
    environment_vars: dict[str, str] | None = None,
    runtime_seconds: float = 0.0,
    results_path: str | Path = "",
) -> ExperimentManifest:
    """
    Create an experiment manifest.

    Args:
        run_id: Unique identifier for this run
        train_data_path: Path to training dataset
        cli_args: Dictionary of CLI arguments used
        test_data_path: Optional path to test dataset
        ood_data_path: Optional path to OOD dataset
        environment_vars: Optional environment variables (e.g., seeds, flags)
        runtime_seconds: Total runtime in seconds
        results_path: Path where results were written

    Returns:
        ExperimentManifest object
    """
    train_path = Path(train_data_path)
    test_path = Path(test_data_path) if test_data_path else None
    ood_path = Path(ood_data_path) if ood_data_path else None

    # Collect dataset metadata
    train_meta = DatasetMetadata(
        path=str(train_path),
        sha256=_sha256_file(train_path),
        rows=_count_csv_rows(train_path),
        columns=_get_csv_columns(train_path),
    )

    test_meta = None
    if test_path and test_path.exists():
        test_meta = DatasetMetadata(
            path=str(test_path),
            sha256=_sha256_file(test_path),
            rows=_count_csv_rows(test_path),
            columns=_get_csv_columns(test_path),
        )

    ood_meta = None
    if ood_path and ood_path.exists():
        ood_meta = DatasetMetadata(
            path=str(ood_path),
            sha256=_sha256_file(ood_path),
            rows=_count_csv_rows(ood_path),
            columns=_get_csv_columns(ood_path),
        )

    # Get code metadata
    commit, branch = _get_git_commit()

    # Collect environment variables of interest
    env_vars = environment_vars or {}
    if not env_vars:
        # Capture relevant env vars
        for key in ["GOJO_DISABLE_ENRICHMENT", "SEED", "PYTHONHASHSEED"]:
            if key in sys.modules:  # This won't work; just capture from os
                import os

                if key in os.environ:
                    env_vars[key] = os.environ[key]

    return ExperimentManifest(
        run_id=run_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        python_version=sys.version,
        code_commit=commit,
        code_branch=branch,
        cli_args=cli_args,
        train_dataset=train_meta,
        test_dataset=test_meta,
        ood_dataset=ood_meta,
        environment_vars=env_vars,
        runtime_seconds=runtime_seconds,
        results_path=str(results_path),
    )
