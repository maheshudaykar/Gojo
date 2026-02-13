"""Gojo phishing URL detector package."""

from __future__ import annotations

import importlib
from typing import Any

_MODULES = {
    "parsing",
    "features",
    "rules",
    "scoring",
    "report",
    "typosquat",
    "ml_lexical",
    "ml_char_ngram",
    "ml_ensemble",
    "policy",
    "policy_v2",
    "feedback",
    "analyze",
    "brand_risk",
    "enrichment",
    "intent",
}

__all__ = sorted(_MODULES)


def __getattr__(name: str) -> Any:
    if name in _MODULES:
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(_MODULES))
