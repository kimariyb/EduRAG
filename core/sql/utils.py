from __future__ import annotations
import json
import math
import unicodedata
import warnings
from collections.abc import Sequence
from typing import Any
from base.logger import logger


log = logger.bind(module=__name__)


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(text)).lower().strip()
    log.debug("Normalized text: original_length={}, normalized_length={}", len(str(text)), len(normalized))
    return normalized


def tokenize(text: str) -> list[str]:
    normalized = normalize_text(text)
    if not normalized:
        log.warning("Tokenization skipped because text is empty")
        return []

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="pkg_resources is deprecated as an API.*",
            category=UserWarning,
        )
        import jieba

    jieba.setLogLevel(60)
    tokens = [token.strip() for token in jieba.lcut(normalized) if token.strip()]
    log.debug("Tokenized text: tokens={}", len(tokens))
    return tokens


def softmax(scores: Sequence[float], *, temperature: float = 0.35) -> list[float]:
    if temperature <= 0:
        raise ValueError("temperature must be greater than 0")
    if not scores:
        log.warning("Softmax skipped because score list is empty")
        return []

    scaled_scores = [float(score) / temperature for score in scores]
    max_score = max(scaled_scores)
    exp_scores = [math.exp(score - max_score) for score in scaled_scores]
    total = sum(exp_scores)
    if total == 0:
        probabilities = [1 / len(exp_scores)] * len(exp_scores)
    else:
        probabilities = [score / total for score in exp_scores]

    log.debug("Softmax normalized scores: count={}, temperature={}", len(probabilities), temperature)
    return probabilities


def config_value(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def encode(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return json.dumps(value, ensure_ascii=False)


def decode(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, str):
        return value
    return str(value)


def validate_identifier(identifier: str) -> str:
    if not identifier.replace("_", "").isalnum():
        raise ValueError(f"invalid SQL identifier: {identifier}")
    return identifier


preprocess_text = tokenize
