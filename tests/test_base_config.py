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
    assert config.llm.model == "qwen3.5:2b"
    assert config.llm.api_key == "ollama"
    assert config.llm.base_url == "http://localhost:11434/v1"
    assert config.rag.parent_chunk_size == 1000
    assert config.rag.retrieval_k == 10
    assert config.rag.embedding_model_path.endswith("/bge-m3")
    assert config.rag.reranker_model_path.endswith("/bge-reranker-v2-m3")
    assert config.rag.query_base_model.endswith("/bert-base-chinese")
    assert config.rag.query_training_data_path == "finetuning_data.jsonl"
    assert config.rag.model_device == "mps"
    assert config.rag.segmenter_device == "cpu"


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

    config = load_config()

    assert config.milvus.port == 19531
    assert config.rag.retrieval_k == 7
    assert config.rag.model_device == "cpu"
    assert config.redis.decode_responses is False
    assert config.llm.api_key == "edurag-key"
    assert config.get("rag.retrieval_k") == 7


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
