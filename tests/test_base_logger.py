from pathlib import Path

from base.config import load_config
from base.logger import setup_logger


def test_setup_logger_writes_to_configured_file(tmp_path: Path):
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

    config = load_config(config_path)
    logger = setup_logger(config)

    logger.info("base logger smoke test")
    logger.complete()

    assert log_path.exists()
    assert "base logger smoke test" in log_path.read_text(encoding="utf-8")
