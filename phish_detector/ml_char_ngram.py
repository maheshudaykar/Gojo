from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence, cast

from phish_detector.parsing import parse_url

try:
    from joblib import dump, load  # type: ignore[import-not-found]
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import Pipeline
    from sklearn.svm import LinearSVC
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "scikit-learn and joblib are required for ML features. "
        "Install with 'pip install scikit-learn joblib'."
    ) from exc


@dataclass(frozen=True)
class MLMetadata:
    model_version: str
    created_at: str
    kind: str


def _normalize_label(value: str) -> int:
    label = value.strip().lower()
    if label in {"1", "phish", "phishing", "malicious", "bad"}:
        return 1
    if label in {"0", "legit", "benign", "good", "safe"}:
        return 0
    raise ValueError(f"Unsupported label: {value}")


def _read_dataset(path: Path, url_col: str, label_col: str) -> tuple[list[str], list[int]]:
    urls: list[str] = []
    labels: list[int] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if url_col not in row or label_col not in row:
                raise KeyError("Dataset missing required columns.")
            url = row[url_col].strip()
            label = _normalize_label(row[label_col])
            if not url:
                continue
            urls.append(url)
            labels.append(label)
    return urls, labels


def train_char_model(
    dataset_path: str,
    url_col: str = "url",
    label_col: str = "label",
    output_path: str = "models/char_model.joblib",
    metadata_path: str = "models/char_metadata.json",
) -> None:
    dataset = Path(dataset_path)
    urls, labels = _read_dataset(dataset, url_col, label_col)

    try:
        calibrated = CalibratedClassifierCV(estimator=LinearSVC(), cv=3)
    except TypeError:
        calibrated = CalibratedClassifierCV(base_estimator=LinearSVC(), cv=3)

    pipeline: Any = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char",
                    ngram_range=(3, 5),
                    min_df=2,
                ),
            ),
            ("clf", calibrated),
        ]
    )
    pipeline.fit(urls, labels)

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    dump(pipeline, output_file)

    metadata = MLMetadata(
        model_version="char_v1",
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        kind="char_ngram",
    )
    Path(metadata_path).write_text(json.dumps(asdict(metadata), indent=2), encoding="utf-8")


def load_char_model(path: str) -> Any:
    return load(path)


def predict_char_proba(model: Any, url: str) -> float:
    parsed = parse_url(url)
    cleaned = parsed.normalized
    proba = cast(Sequence[float], model.predict_proba([cleaned])[0])
    classes = list(cast(Sequence[int], model.classes_))
    if 1 in classes:
        idx = classes.index(1)
        return float(proba[idx])
    return float(proba[-1])
