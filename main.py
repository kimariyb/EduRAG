from __future__ import annotations

from pathlib import Path

from base.config import AppConfig, load_config
from base.logger import logger, setup_logger


def initialize_app(config_path: str | Path | None = None) -> AppConfig:
    config = load_config(config_path) if config_path is not None else load_config()
    setup_logger(config)
    logger.info("EduRAG application initialized")
    return config


def main() -> None:
    initialize_app()


if __name__ == "__main__":
    main()
