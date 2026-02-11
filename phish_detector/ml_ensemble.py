from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from phish_detector.ml_char_ngram import load_char_model, predict_char_proba
from phish_detector.ml_lexical import load_lexical_model, predict_lexical_proba


@dataclass(frozen=True)
class EnsembleResult:
    lexical: float
    char: float
    ensemble: float


def load_models(lexical_path: str, char_path: str) -> tuple[Any, Any]:
    return load_lexical_model(lexical_path), load_char_model(char_path)


def predict_ensemble_proba(
    url: str,
    lexical_model: Any,
    char_model: Any,
    weight_char: float = 0.6,
) -> EnsembleResult:
    char_score = predict_char_proba(char_model, url)
    lexical_score = predict_lexical_proba(lexical_model, url)
    weight_char = min(max(weight_char, 0.0), 1.0)
    ensemble = (weight_char * char_score) + ((1.0 - weight_char) * lexical_score)
    return EnsembleResult(lexical=lexical_score, char=char_score, ensemble=ensemble)
