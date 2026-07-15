from pathlib import Path

import pytest

from base.config import ConfigError, load_config


def test_load_config_reads_default_yaml_sections():
    config = load_config()

    assert config.mysql.host == "localhost"
    assert config.mysql.port == 3306
    assert config.mysql.database == "edurag"
    assert config.redis.host == "localhost"
    assert config.redis.port == 6379
    assert config.log.level == "INFO"
    assert config.milvus.port == 19530
    assert config.milvus.collection == "edurag_knowledge"
    assert config.llm.provider == "ollama"
    assert config.llm.model == "qwen3.5:4b-mlx"
    assert config.llm.api_key == "ollama"
    assert config.llm.base_url == "http://localhost:11434/v1"
    assert config.llm.max_tokens == 384
    assert config.llm.reasoning_effort == "none"
    assert config.llm.stream is True
    assert config.rag.parent_chunk_size == 1200
    assert config.rag.retrieval_k == 5
    assert config.rag.knowledge_base_path == str(
        Path(__file__).resolve().parents[1] / "data/ai_data"
    )
    assert config.rag.embedding_model_path.endswith("/bge-m3")
    assert config.rag.reranker_model_path.endswith("/bge-reranker-v2-m3")
    assert config.rag.query_base_model.endswith("/bert-base-chinese")
    assert config.rag.query_training_data_path == str(
        Path(__file__).resolve().parents[1]
        / "core/rag/data/finetuning_data.jsonl"
    )
    assert config.rag.model_device == "mps"
    assert config.rag.segmenter_device == "cpu"
    assert config.eval.quality_threshold == 4
    assert config.eval.critique_max_retries == 3
    assert config.eval.test_samples_path == str(
        Path(__file__).resolve().parents[1]
        / "eval/data/test_samples.jsonl"
    )
    assert config.eval.critique_results_path == str(
        Path(__file__).resolve().parents[1]
        / "eval/data/test_samples_critiqued.jsonl"
    )
    assert config.eval.filtered_samples_path == str(
        Path(__file__).resolve().parents[1]
        / "eval/data/test_samples_filtered.jsonl"
    )
    assert config.eval.rag_predictions_path == str(
        Path(__file__).resolve().parents[1]
        / "eval/data/rag_predictions.jsonl"
    )
    assert config.eval.rag_evaluation_path == str(
        Path(__file__).resolve().parents[1]
        / "eval/data/rag_evaluation.jsonl"
    )
    assert config.eval.rag_summary_path == str(
        Path(__file__).resolve().parents[1]
        / "eval/data/rag_evaluation_summary.json"
    )
    assert config.eval.ragas_max_workers == 1
    assert config.eval.ragas_timeout == 180


def test_load_config_supports_nested_dot_lookup(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
mysql:
  host: db.local
  port: 3307
  username: edu
  password: secret
  database: edu_test
  charset: utf8mb4
redis:
  host: cache.local
  port: 6380
  db: 2
  password: redis-secret
  decode_responses: true
log:
  level: DEBUG
  file: logs/test.log
  rotation: 1 day
  retention: 7 days
  compression: zip
  enqueue: false
""".strip(),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.get("mysql.host") == "db.local"
    assert config.get("redis.db") == 2
    assert config.get("log.rotation") == "1 day"
    assert config.get("milvus.collection") == "edurag_knowledge"
    assert config.get("rag.child_chunk_size") == 300
    assert config.get("missing.key", "fallback") == "fallback"


def test_load_config_resolves_filesystem_paths_from_config_directory(
    tmp_path: Path,
):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
mysql: {}
redis: {}
log:
  file: logs/app.log
rag:
  knowledge_base_path: data/knowledge
  query_model_path: models/query_classifier
  query_training_data_path: data/finetuning.jsonl
eval:
  test_samples_path: data/test_samples.jsonl
  critique_results_path: data/test_samples_critiqued.jsonl
  filtered_samples_path: data/test_samples_filtered.jsonl
  rag_predictions_path: data/rag_predictions.jsonl
  rag_evaluation_path: data/rag_evaluation.jsonl
  rag_summary_path: data/rag_summary.json
""".strip(),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.log.file == str(tmp_path / "logs/app.log")
    assert config.rag.knowledge_base_path == str(tmp_path / "data/knowledge")
    assert config.rag.query_model_path == str(
        tmp_path / "models/query_classifier"
    )
    assert config.rag.query_training_data_path == str(
        tmp_path / "data/finetuning.jsonl"
    )
    assert config.eval.test_samples_path == str(
        tmp_path / "data/test_samples.jsonl"
    )
    assert config.eval.critique_results_path == str(
        tmp_path / "data/test_samples_critiqued.jsonl"
    )
    assert config.eval.filtered_samples_path == str(
        tmp_path / "data/test_samples_filtered.jsonl"
    )
    assert config.eval.rag_predictions_path == str(
        tmp_path / "data/rag_predictions.jsonl"
    )
    assert config.eval.rag_evaluation_path == str(
        tmp_path / "data/rag_evaluation.jsonl"
    )
    assert config.eval.rag_summary_path == str(
        tmp_path / "data/rag_summary.json"
    )


def test_load_config_rejects_missing_required_sections(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("mysql: {}\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="missing required section"):
        load_config(config_path)


def test_load_config_applies_typed_environment_overrides(monkeypatch):
    monkeypatch.setenv("EDURAG_MILVUS_PORT", "19531")
    monkeypatch.setenv("EDURAG_RAG_RETRIEVAL_K", "7")
    monkeypatch.setenv("EDURAG_RAG_MODEL_DEVICE", "cpu")
    monkeypatch.setenv("EDURAG_REDIS_DECODE_RESPONSES", "false")
    monkeypatch.setenv("EDURAG_LLM_API_KEY", "edurag-key")
    monkeypatch.setenv("EDURAG_LLM_MAX_TOKENS", "256")
    monkeypatch.setenv("EDURAG_LLM_REASONING_EFFORT", "low")
    monkeypatch.setenv("EDURAG_LLM_STREAM", "false")
    monkeypatch.setenv("EDURAG_EVAL_QUALITY_THRESHOLD", "5")
    monkeypatch.setenv("EDURAG_EVAL_CRITIQUE_MAX_RETRIES", "2")

    config = load_config()

    assert config.milvus.port == 19531
    assert config.rag.retrieval_k == 7
    assert config.rag.model_device == "cpu"
    assert config.redis.decode_responses is False
    assert config.llm.api_key == "edurag-key"
    assert config.llm.max_tokens == 256
    assert config.llm.reasoning_effort == "low"
    assert config.llm.stream is False
    assert config.eval.quality_threshold == 5
    assert config.eval.critique_max_retries == 2
    assert config.get("rag.retrieval_k") == 7
    assert config.get("eval.quality_threshold") == 5


def test_admin_token_is_read_only_from_environment(monkeypatch, tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
mysql: {}
redis: {}
log: {}
admin_token: yaml-secret
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("EDURAG_ADMIN_TOKEN", "admin-secret")

    config = load_config(config_path)

    assert config.admin_token == "admin-secret"


def test_admin_token_is_not_loaded_from_yaml(monkeypatch, tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
mysql: {}
redis: {}
log: {}
admin_token: yaml-secret
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.delenv("EDURAG_ADMIN_TOKEN", raising=False)

    config = load_config(config_path)

    assert config.admin_token is None


def test_admin_token_is_excluded_from_raw_config(monkeypatch, tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
mysql: {}
redis: {}
log: {}
admin_token: yaml-secret
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.delenv("EDURAG_ADMIN_TOKEN", raising=False)

    config = load_config(config_path)

    assert config.get("admin_token") is None


def test_load_config_rejects_invalid_rag_values(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
mysql: {}
redis: {}
log: {}
rag:
  parent_chunk_size: 0
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(
        ConfigError,
        match="rag.parent_chunk_size must be greater than 0",
    ):
        load_config(config_path)


def test_load_config_rejects_non_positive_llm_max_tokens(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
mysql: {}
redis: {}
log: {}
llm:
  max_tokens: 0
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(
        ConfigError,
        match="llm.max_tokens must be greater than 0",
    ):
        load_config(config_path)


def test_load_config_rejects_unknown_reasoning_effort(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
mysql: {}
redis: {}
log: {}
llm:
  reasoning_effort: extreme
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(
        ConfigError,
        match="llm.reasoning_effort must be one of",
    ):
        load_config(config_path)


def test_load_config_rejects_eval_threshold_outside_score_range(
    tmp_path: Path,
):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
mysql: {}
redis: {}
log: {}
eval:
  quality_threshold: 6
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(
        ConfigError,
        match="eval.quality_threshold must be between 1 and 5",
    ):
        load_config(config_path)


def test_load_config_rejects_non_positive_critique_retries(
    tmp_path: Path,
):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
mysql: {}
redis: {}
log: {}
eval:
  critique_max_retries: 0
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(
        ConfigError,
        match="eval.critique_max_retries must be greater than 0",
    ):
        load_config(config_path)


@pytest.mark.parametrize(
    ("field_name", "message"),
    [
        ("ragas_max_workers", "eval.ragas_max_workers"),
        ("ragas_timeout", "eval.ragas_timeout"),
    ],
)
def test_load_config_rejects_non_positive_ragas_runtime_values(
    tmp_path: Path,
    field_name: str,
    message: str,
):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
mysql: {{}}
redis: {{}}
log: {{}}
eval:
  {field_name}: 0
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(
        ConfigError,
        match=rf"{message} must be greater than 0",
    ):
        load_config(config_path)
