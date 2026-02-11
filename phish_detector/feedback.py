from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast


@dataclass
class FeedbackEntry:
    id: str
    url: str
    predicted_label: str
    confidence: float
    context: str
    action: float
    rule_score: int
    ml_score: float
    final_score: int
    status: str
    created_at: str
    resolved_at: str | None = None
    true_label: str | None = None


def _load_entries(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        return []
    data_list = cast(list[Any], data)
    filtered = [cast(dict[str, Any], item) for item in data_list if isinstance(item, dict)]
    return filtered


def _save_entries(path: Path, entries: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries, indent=2), encoding="utf-8")


def _coerce_entry(entry: dict[str, Any]) -> FeedbackEntry:
    return FeedbackEntry(
        id=str(entry.get("id", "")),
        url=str(entry.get("url", "")),
        predicted_label=str(entry.get("predicted_label", "")),
        confidence=float(entry.get("confidence", 0.0)),
        context=str(entry.get("context", "")),
        action=float(entry.get("action", 0.0)),
        rule_score=int(entry.get("rule_score", 0)),
        ml_score=float(entry.get("ml_score", 0.0)),
        final_score=int(entry.get("final_score", 0)),
        status=str(entry.get("status", "pending")),
        created_at=str(entry.get("created_at", "")),
        resolved_at=entry.get("resolved_at"),
        true_label=entry.get("true_label"),
    )


def record_pending(entry: FeedbackEntry, path: str) -> None:
    storage = Path(path)
    entries = _load_entries(storage)
    entries.append(asdict(entry))
    _save_entries(storage, entries)


def resolve_feedback(feedback_id: str, true_label: str, path: str) -> FeedbackEntry | None:
    storage = Path(path)
    entries = _load_entries(storage)
    updated: FeedbackEntry | None = None

    for entry in entries:
        if entry.get("id") == feedback_id:
            entry["status"] = "resolved"
            entry["true_label"] = true_label
            entry["resolved_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            updated = _coerce_entry(entry)
            break

    if updated:
        _save_entries(storage, entries)
    return updated


def create_entry(
    url: str,
    predicted_label: str,
    confidence: float,
    context: str,
    action: float,
    rule_score: int,
    ml_score: float,
    final_score: int,
) -> FeedbackEntry:
    return FeedbackEntry(
        id=str(uuid.uuid4()),
        url=url,
        predicted_label=predicted_label,
        confidence=confidence,
        context=context,
        action=action,
        rule_score=rule_score,
        ml_score=ml_score,
        final_score=final_score,
        status="pending",
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
