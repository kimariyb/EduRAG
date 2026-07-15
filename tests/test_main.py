import os
from pathlib import Path

from api import deps
from base.logger import logger
import main as main_module
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


def test_main_sets_mock_mode_before_initializing(monkeypatch):
    calls = []
    configured = object()
    monkeypatch.delenv("EDURAG_API_MOCK", raising=False)
    monkeypatch.setattr(main_module, "initialize_app", lambda _: configured)
    monkeypatch.setattr(
        main_module,
        "initialize_system",
        lambda: calls.append(os.environ.get("EDURAG_API_MOCK")),
    )
    monkeypatch.setattr(main_module, "run_server", lambda **_: None)
    monkeypatch.setattr(deps, "_config", None, raising=False)

    main_module.main(["--mock"])

    assert calls == ["true"]
    assert deps._config is configured


def test_run_server_sets_the_explicit_mock_gate(monkeypatch):
    observed = {}
    monkeypatch.delenv("EDURAG_API_MOCK", raising=False)
    monkeypatch.setattr(
        main_module.uvicorn,
        "run",
        lambda *args, **kwargs: observed.update(
            mock_enabled=os.environ.get("EDURAG_API_MOCK")
        ),
    )

    main_module.run_server(mock=True)

    assert observed["mock_enabled"] == "true"


def test_main_exports_selected_config_path_for_reload_worker(monkeypatch, tmp_path):
    config_path = tmp_path / "selected.yaml"
    configured = object()
    observed = {}
    monkeypatch.delenv("EDURAG_CONFIG_PATH", raising=False)
    monkeypatch.setattr(main_module, "initialize_app", lambda path: configured)
    monkeypatch.setattr(main_module, "initialize_system", lambda: None)
    monkeypatch.setattr(
        main_module,
        "run_server",
        lambda **kwargs: observed.update(kwargs),
    )

    main_module.main(["--config", str(config_path), "--reload"])

    assert os.environ["EDURAG_CONFIG_PATH"] == str(config_path.resolve())
    assert observed["reload"] is True
