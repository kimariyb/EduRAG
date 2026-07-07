from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"
REQUIRED_SECTIONS = ("mysql", "redis", "log")


class ConfigError(RuntimeError):
    """Raised when the project configuration cannot be loaded."""


@dataclass(frozen=True)
class AppConfig:
    mysql: SimpleNamespace
    redis: SimpleNamespace
    log: SimpleNamespace
    raw: dict[str, Any]

    def get(self, key: str, default: Any = None) -> Any:
        value: Any = self.raw
        for part in key.split("."):
            if not isinstance(value, dict) or part not in value:
                return default
            value = value[part]
        return value


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> AppConfig:
    config_path = Path(path)
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

    return AppConfig(
        mysql=_to_namespace(data["mysql"]),
        redis=_to_namespace(data["redis"]),
        log=_to_namespace(data["log"]),
        raw=data,
    )


def _to_namespace(value: Any) -> SimpleNamespace:
    if not isinstance(value, dict):
        raise ConfigError("config sections must be YAML mappings")

    return SimpleNamespace(
        **{
            key: _to_namespace(item) if isinstance(item, dict) else item
            for key, item in value.items()
        }
    )
