from __future__ import annotations

from pathlib import Path
from typing import Any

from loguru import logger

from base.config import AppConfig


def setup_logger(config: AppConfig) -> Any:
    log_file = Path(config.log.file)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger.add(
        log_file,
        level=config.log.level,
        rotation=config.log.rotation,
        retention=config.log.retention,
        compression=config.log.compression,
        enqueue=config.log.enqueue,
        encoding="utf-8",
        backtrace=True,
        diagnose=False,
    )

    return logger

