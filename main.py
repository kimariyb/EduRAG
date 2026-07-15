from __future__ import annotations

import argparse
import os
from pathlib import Path

import uvicorn

from base.config import AppConfig, DEFAULT_CONFIG_PATH, load_config
from base.logger import logger, setup_logger


def initialize_app(config_path: str | Path | None = None) -> AppConfig:
    """加载配置并初始化日志。"""
    config = load_config(config_path) if config_path is not None else load_config()
    setup_logger(config)
    logger.info("EduRAG application initialized")
    return config


def initialize_system() -> None:
    """预热问答系统（EducationQASystem 或演示模式回退）。

    后端（MySQL/Redis/Milvus/LLM）未就绪时会自动回退到内存演示模式，
    见 api.deps 的懒加载与降级逻辑，这里仅触发其初始化以获得即时状态。
    """
    from api.deps import ensure_system, get_system_status

    ensure_system()
    state = get_system_status()
    mode = "演示模式(mock)" if state["mock"] else "完整后端"
    logger.info("问答系统已初始化: {}", mode)
    if state["error"]:
        logger.warning("后端初始化异常: {}", state["error"])


def run_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    reload: bool = False,
    mock: bool = False,
    config_path: str | Path | None = None,
) -> None:
    """启动 FastAPI 服务（前端由 api.app 静态挂载一并托管）。"""
    if mock:
        os.environ["EDURAG_API_MOCK"] = "true"
        logger.warning("已启用演示模式 (EDURAG_API_MOCK=true)")
    if config_path is not None:
        os.environ["EDURAG_CONFIG_PATH"] = str(
            Path(config_path).expanduser().resolve()
        )

    logger.info("启动服务 http://{}:{}", host, port)
    uvicorn.run(
        "api.app:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="EduRAG",
        description="EduRAG 教育问答系统：初始化并启动前端 + 后端服务",
    )
    parser.add_argument("-c", "--config", default=None, help="配置文件路径 (默认 config.yaml)")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址 (默认 0.0.0.0)")
    parser.add_argument("-p", "--port", type=int, default=8001, help="监听端口 (默认 8001)")
    parser.add_argument("--reload", action="store_true", help="开启热重载（开发用）")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="强制演示模式，无需 MySQL/Redis/Milvus/LLM 等后端",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.mock:
        os.environ["EDURAG_API_MOCK"] = "true"
    config_path = Path(args.config or DEFAULT_CONFIG_PATH).expanduser().resolve()
    os.environ["EDURAG_CONFIG_PATH"] = str(config_path)
    config = initialize_app(config_path)
    from api.deps import configure_application

    configure_application(config)
    initialize_system()
    run_server(
        host=args.host,
        port=args.port,
        reload=args.reload,
        config_path=config_path,
    )


if __name__ == "__main__":
    main()
