from pathlib import Path

from base.logger import logger
from main import initialize_app


def test_initialize_app_configures_shared_logger(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    log_path = tmp_path / "logs" / "app.log"
    config_path.write_text(
        f"""
mysql:
  host: localhost
  port: 3306
  username: root
  password: ""
  database: edurag
  charset: utf8mb4
redis:
  host: localhost
  port: 6379
  db: 0
  password:
  decode_responses: true
log:
  level: INFO
  file: "{log_path}"
  rotation: 10 MB
  retention: 7 days
  compression: zip
  enqueue: false
""".strip(),
        encoding="utf-8",
    )

    config = initialize_app(config_path)
    logger.info("shared logger configured from main")
    logger.complete()

    assert config.log.file == str(log_path)
    assert log_path.exists()
    assert "shared logger configured from main" in log_path.read_text(encoding="utf-8")
