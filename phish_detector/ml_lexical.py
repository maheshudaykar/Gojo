from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence, cast

from phish_detector.features import extract_features, get_feature_schema, load_suspicious_tlds, vectorize_features
from phish_detector.parsing import parse_url

try:
    from joblib import dump, load  # type: ignore[import-not-found]
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "scikit-learn and joblib are required for ML features. "
        "Install with 'pip install scikit-learn joblib'."
    ) from exc


@dataclass(frozen=True)
class MLMetadata:
    model_version: str
    feature_schema: list[str]
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


def train_lexical_model(
    dataset_path: str,
    url_col: str = "url",
    label_col: str = "label",
    output_path: str = "models/lexical_model.joblib",
    metadata_path: str = "models/lexical_metadata.json",
) -> None:
    dataset = Path(dataset_path)
    urls, labels = _read_dataset(dataset, url_col, label_col)

    suspicious_tlds = load_suspicious_tlds()
    feature_rows: list[list[float]] = []
    for url in urls:
        parsed = parse_url(url)
        features = extract_features(parsed, suspicious_tlds)
        feature_rows.append(vectorize_features(features))

    base_model = LogisticRegression(max_iter=1000, class_weight="balanced")
    try:
        calibrated = CalibratedClassifierCV(estimator=base_model, cv=3)
    except TypeError:
        calibrated = CalibratedClassifierCV(base_estimator=base_model, cv=3)

    pipeline: Any = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", calibrated),
        ]
    )
    pipeline.fit(feature_rows, labels)

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    dump(pipeline, output_file)

    metadata = MLMetadata(
        model_version="lexical_v1",
        feature_schema=get_feature_schema(),
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        kind="lexical",
    )
    Path(metadata_path).write_text(json.dumps(asdict(metadata), indent=2), encoding="utf-8")


def load_lexical_model(path: str) -> Any:
    return load(path)


def predict_lexical_proba(model: Any, url: str) -> float:
    parsed = parse_url(url)
    features = extract_features(parsed, load_suspicious_tlds())
    vector = vectorize_features(features)
    proba = cast(Sequence[float], model.predict_proba([vector])[0])
    classes = list(cast(Sequence[int], model.classes_))
    if 1 in classes:
        idx = classes.index(1)
        return float(proba[idx])
    return float(proba[-1])
