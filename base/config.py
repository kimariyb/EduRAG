from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"
REQUIRED_SECTIONS = ("mysql", "redis", "log")


class ConfigError(RuntimeError):
    """Raised when the project configuration cannot be loaded."""


@dataclass(frozen=True)
class MySQLConfig:
    host: str = "localhost"
    port: int = 3306
    username: str = "root"
    password: str = ""
    database: str = "edurag"
    charset: str = "utf8mb4"


@dataclass(frozen=True)
class RedisConfig:
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: str | None = None
    decode_responses: bool = True


@dataclass(frozen=True)
class LogConfig:
    level: str = "INFO"
    file: str = "logs/app.log"
    rotation: str = "10 MB"
    retention: str = "14 days"
    compression: str = "zip"
    enqueue: bool = True


@dataclass(frozen=True)
class MilvusConfig:
    host: str = "localhost"
    port: int = 19530
    database: str = "default"
    collection: str = "edurag_knowledge"


@dataclass(frozen=True)
class LLMConfig:
    provider: str = "ollama"
    model: str = "qwen3.5:4b-mlx"
    api_key: str | None = "ollama"
    base_url: str = "http://localhost:11434/v1"
    temperature: float = 0.1
    max_tokens: int = 384
    reasoning_effort: str = "none"
    stream: bool = True


@dataclass(frozen=True)
class RAGConfig:
    parent_chunk_size: int = 1000
    child_chunk_size: int = 300
    chunk_overlap: float = 0.25
    retrieval_k: int = 10
    candidate_m: int = 3
    customer_service_phone: str = "400-000-0000"
    knowledge_base_path: str = "data/ai_data"
    query_base_model: str = "bert-base-chinese"
    query_model_path: str = "core/bert_query_classifier"
    query_training_data_path: str = "finetuning_data.jsonl"
    embedding_model_path: str = "BAAI/bge-m3"
    reranker_model_path: str = "./bge/bge-reranker-large"
    model_device: str = "cpu"
    segmenter_device: str = "cpu"


@dataclass(frozen=True)
class EvalConfig:
    quality_threshold: int = 4
    critique_max_retries: int = 3
    test_samples_path: str = "eval/data/test_samples.jsonl"
    critique_results_path: str = (
        "eval/data/test_samples_critiqued.jsonl"
    )
    filtered_samples_path: str = "eval/data/test_samples_filtered.jsonl"
    rag_predictions_path: str = "eval/data/rag_predictions.jsonl"
    rag_evaluation_path: str = "eval/data/rag_evaluation.jsonl"
    rag_summary_path: str = "eval/data/rag_evaluation_summary.json"
    ragas_max_workers: int = 1
    ragas_timeout: int = 180


@dataclass(frozen=True)
class AppConfig:
    mysql: MySQLConfig
    redis: RedisConfig
    log: LogConfig
    milvus: MilvusConfig
    llm: LLMConfig
    rag: RAGConfig
    eval: EvalConfig
    raw: dict[str, Any]

    def get(self, key: str, default: Any = None) -> Any:
        value: Any = self.raw
        for part in key.split("."):
            if not isinstance(value, dict) or part not in value:
                return default
            value = value[part]
        return value


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> AppConfig:
    config_path = Path(path).expanduser().resolve()
    if not config_path.exists():
        raise ConfigError(f"config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ConfigError("config file must contain a YAML mapping")

    missing_sections = [section for section in REQUIRED_SECTIONS if section not in data]
    if missing_sections:
        names = ", ".join(missing_sections)
        raise ConfigError(f"missing required section: {names}")

    merged = _apply_environment_overrides(data)
    _resolve_filesystem_paths(merged, config_path.parent)
    mysql = _mysql_config(_section(merged, "mysql"))
    redis = _redis_config(_section(merged, "redis"))
    log = _log_config(_section(merged, "log"))
    milvus = _milvus_config(_section(merged, "milvus"))
    llm = _llm_config(_section(merged, "llm"))
    rag = _rag_config(_section(merged, "rag"))
    eval_config = _eval_config(_section(merged, "eval"))
    _validate_llm_config(llm)
    _validate_rag_config(rag)
    _validate_eval_config(eval_config)

    normalized = dict(merged)
    normalized.update(
        {
            "mysql": asdict(mysql),
            "redis": asdict(redis),
            "log": asdict(log),
            "milvus": asdict(milvus),
            "llm": asdict(llm),
            "rag": asdict(rag),
            "eval": asdict(eval_config),
        }
    )
    return AppConfig(
        mysql=mysql,
        redis=redis,
        log=log,
        milvus=milvus,
        llm=llm,
        rag=rag,
        eval=eval_config,
        raw=normalized,
    )


def _section(data: dict[str, Any], name: str) -> dict[str, Any]:
    value = data.get(name, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConfigError(f"config section must be a YAML mapping: {name}")
    return value


def _apply_environment_overrides(data: dict[str, Any]) -> dict[str, Any]:
    result = {
        key: dict(value) if isinstance(value, dict) else value
        for key, value in data.items()
    }
    for section in (
        "mysql",
        "redis",
        "log",
        "milvus",
        "llm",
        "rag",
        "eval",
    ):
        values = result.setdefault(section, {})
        if not isinstance(values, dict):
            continue
        known_fields = _section_field_names(section)
        for field_name in known_fields:
            env_name = f"EDURAG_{section}_{field_name}".upper()
            if env_name in os.environ:
                values[field_name] = os.environ[env_name]

    return result


def _resolve_filesystem_paths(
    data: dict[str, Any],
    config_directory: Path,
) -> None:
    path_fields = {
        "log": {"file": "logs/app.log"},
        "rag": {
            "knowledge_base_path": "data/ai_data",
            "query_model_path": "core/bert_query_classifier",
            "query_training_data_path": "finetuning_data.jsonl",
        },
        "eval": {
            "test_samples_path": "eval/data/test_samples.jsonl",
            "critique_results_path": (
                "eval/data/test_samples_critiqued.jsonl"
            ),
            "filtered_samples_path": (
                "eval/data/test_samples_filtered.jsonl"
            ),
            "rag_predictions_path": "eval/data/rag_predictions.jsonl",
            "rag_evaluation_path": "eval/data/rag_evaluation.jsonl",
            "rag_summary_path": (
                "eval/data/rag_evaluation_summary.json"
            ),
        },
    }
    for section, fields in path_fields.items():
        values = data.setdefault(section, {})
        if not isinstance(values, dict):
            continue
        for field_name, default in fields.items():
            configured_path = Path(str(values.get(field_name) or default))
            configured_path = configured_path.expanduser()
            if not configured_path.is_absolute():
                configured_path = config_directory / configured_path
            values[field_name] = str(configured_path.resolve())


def _section_field_names(section: str) -> tuple[str, ...]:
    config_type = {
        "mysql": MySQLConfig,
        "redis": RedisConfig,
        "log": LogConfig,
        "milvus": MilvusConfig,
        "llm": LLMConfig,
        "rag": RAGConfig,
        "eval": EvalConfig,
    }[section]
    return tuple(config_type.__dataclass_fields__)


def _mysql_config(values: dict[str, Any]) -> MySQLConfig:
    return MySQLConfig(
        host=str(values.get("host") or "localhost"),
        port=_as_int(values.get("port"), 3306, "mysql.port"),
        username=str(values.get("username") or values.get("user") or "root"),
        password=str(values.get("password") or ""),
        database=str(values.get("database") or "edurag"),
        charset=str(values.get("charset") or "utf8mb4"),
    )


def _redis_config(values: dict[str, Any]) -> RedisConfig:
    password = values.get("password")
    return RedisConfig(
        host=str(values.get("host") or "localhost"),
        port=_as_int(values.get("port"), 6379, "redis.port"),
        db=_as_int(values.get("db"), 0, "redis.db"),
        password=str(password) if password not in (None, "") else None,
        decode_responses=_as_bool(
            values.get("decode_responses"),
            True,
            "redis.decode_responses",
        ),
    )


def _log_config(values: dict[str, Any]) -> LogConfig:
    return LogConfig(
        level=str(values.get("level") or "INFO").upper(),
        file=str(values.get("file") or "logs/app.log"),
        rotation=str(values.get("rotation") or "10 MB"),
        retention=str(values.get("retention") or "14 days"),
        compression=str(values.get("compression") or "zip"),
        enqueue=_as_bool(values.get("enqueue"), True, "log.enqueue"),
    )


def _milvus_config(values: dict[str, Any]) -> MilvusConfig:
    return MilvusConfig(
        host=str(values.get("host") or "localhost"),
        port=_as_int(values.get("port"), 19530, "milvus.port"),
        database=str(values.get("database") or "default"),
        collection=str(values.get("collection") or "edurag_knowledge"),
    )


def _llm_config(values: dict[str, Any]) -> LLMConfig:
    provider = str(values.get("provider") or "ollama").lower()
    api_key = values.get("api_key")
    if api_key not in (None, ""):
        normalized_api_key = str(api_key)
    elif provider == "ollama":
        normalized_api_key = "ollama"
    else:
        normalized_api_key = None

    return LLMConfig(
        provider=provider,
        model=str(values.get("model") or "qwen3.5:4b-mlx"),
        api_key=normalized_api_key,
        base_url=str(
            values.get("base_url") or "http://localhost:11434/v1"
        ),
        temperature=_as_float(
            values.get("temperature"),
            0.1,
            "llm.temperature",
        ),
        max_tokens=_as_int(
            values.get("max_tokens"),
            384,
            "llm.max_tokens",
        ),
        reasoning_effort=str(
            values.get("reasoning_effort") or "none"
        ).lower(),
        stream=_as_bool(values.get("stream"), True, "llm.stream"),
    )


def _rag_config(values: dict[str, Any]) -> RAGConfig:
    return RAGConfig(
        parent_chunk_size=_as_int(
            values.get("parent_chunk_size"),
            1000,
            "rag.parent_chunk_size",
        ),
        child_chunk_size=_as_int(
            values.get("child_chunk_size"),
            300,
            "rag.child_chunk_size",
        ),
        chunk_overlap=_as_float(
            values.get("chunk_overlap"),
            0.25,
            "rag.chunk_overlap",
        ),
        retrieval_k=_as_int(
            values.get("retrieval_k"),
            10,
            "rag.retrieval_k",
        ),
        candidate_m=_as_int(
            values.get("candidate_m"),
            3,
            "rag.candidate_m",
        ),
        customer_service_phone=str(
            values.get("customer_service_phone") or "400-000-0000"
        ),
        knowledge_base_path=str(
            values.get("knowledge_base_path") or "data/ai_data"
        ),
        query_base_model=str(
            values.get("query_base_model") or "bert-base-chinese"
        ),
        query_model_path=str(
            values.get("query_model_path") or "core/bert_query_classifier"
        ),
        query_training_data_path=str(
            values.get("query_training_data_path")
            or "finetuning_data.jsonl"
        ),
        embedding_model_path=str(
            values.get("embedding_model_path") or "BAAI/bge-m3"
        ),
        reranker_model_path=str(
            values.get("reranker_model_path")
            or "./bge/bge-reranker-large"
        ),
        model_device=str(
            values.get("model_device")
            or values.get("embedding_device")
            or "cpu"
        ),
        segmenter_device=str(values.get("segmenter_device") or "cpu"),
    )


def _eval_config(values: dict[str, Any]) -> EvalConfig:
    return EvalConfig(
        quality_threshold=_as_int(
            values.get("quality_threshold"),
            4,
            "eval.quality_threshold",
        ),
        critique_max_retries=_as_int(
            values.get("critique_max_retries"),
            3,
            "eval.critique_max_retries",
        ),
        test_samples_path=str(
            values.get("test_samples_path")
            or "eval/data/test_samples.jsonl"
        ),
        critique_results_path=str(
            values.get("critique_results_path")
            or "eval/data/test_samples_critiqued.jsonl"
        ),
        filtered_samples_path=str(
            values.get("filtered_samples_path")
            or "eval/data/test_samples_filtered.jsonl"
        ),
        rag_predictions_path=str(
            values.get("rag_predictions_path")
            or "eval/data/rag_predictions.jsonl"
        ),
        rag_evaluation_path=str(
            values.get("rag_evaluation_path")
            or "eval/data/rag_evaluation.jsonl"
        ),
        rag_summary_path=str(
            values.get("rag_summary_path")
            or "eval/data/rag_evaluation_summary.json"
        ),
        ragas_max_workers=_as_int(
            values.get("ragas_max_workers"),
            1,
            "eval.ragas_max_workers",
        ),
        ragas_timeout=_as_int(
            values.get("ragas_timeout"),
            180,
            "eval.ragas_timeout",
        ),
    )


def _validate_rag_config(config: RAGConfig) -> None:
    if config.parent_chunk_size <= 0:
        raise ConfigError("rag.parent_chunk_size must be greater than 0")
    if config.child_chunk_size <= 0:
        raise ConfigError("rag.child_chunk_size must be greater than 0")
    if config.chunk_overlap < 0:
        raise ConfigError("rag.chunk_overlap cannot be negative")
    if config.retrieval_k <= 0:
        raise ConfigError("rag.retrieval_k must be greater than 0")
    if config.candidate_m <= 0:
        raise ConfigError("rag.candidate_m must be greater than 0")
    if config.chunk_overlap >= 1:
        overlap = int(config.chunk_overlap)
        if overlap >= min(config.parent_chunk_size, config.child_chunk_size):
            raise ConfigError(
                "rag.chunk_overlap must be smaller than both chunk sizes"
            )


def _validate_llm_config(config: LLMConfig) -> None:
    if not 0 <= config.temperature <= 2:
        raise ConfigError("llm.temperature must be between 0 and 2")
    if config.max_tokens <= 0:
        raise ConfigError("llm.max_tokens must be greater than 0")
    allowed_reasoning_efforts = {"none", "low", "medium", "high"}
    if config.reasoning_effort not in allowed_reasoning_efforts:
        allowed_values = ", ".join(sorted(allowed_reasoning_efforts))
        raise ConfigError(
            "llm.reasoning_effort must be one of: "
            f"{allowed_values}"
        )


def _validate_eval_config(config: EvalConfig) -> None:
    if not 1 <= config.quality_threshold <= 5:
        raise ConfigError(
            "eval.quality_threshold must be between 1 and 5"
        )
    if config.critique_max_retries <= 0:
        raise ConfigError(
            "eval.critique_max_retries must be greater than 0"
        )
    if config.ragas_max_workers <= 0:
        raise ConfigError(
            "eval.ragas_max_workers must be greater than 0"
        )
    if config.ragas_timeout <= 0:
        raise ConfigError("eval.ragas_timeout must be greater than 0")


def _as_int(value: Any, default: int, name: str) -> int:
    if value in (None, ""):
        return default
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"config value must be an integer: {name}") from exc


def _as_float(value: Any, default: float, name: str) -> float:
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"config value must be a number: {name}") from exc


def _as_bool(value: Any, default: bool, name: str) -> bool:
    if value in (None, ""):
        return default
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ConfigError(f"config value must be a boolean: {name}")
