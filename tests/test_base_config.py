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
    assert config.get("missing.key", "fallback") == "fallback"


def test_load_config_rejects_missing_required_sections(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("mysql: {}\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="missing required section"):
        load_config(config_path)
