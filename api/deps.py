from __future__ import annotations

import hmac
import os
import threading
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from fastapi import Header, HTTPException, status

from base.config import AppConfig
from base.logger import logger


log = logger.bind(module=__name__)

@dataclass
class _QAResult:
    """Minimal stand-in for core.system.QAResponse used in mock mode."""

    session_id: str
    source: str
    answer: str
    history: list[dict[str, Any]] = field(default_factory=list)

# Lazy-initialized singleton. The in-memory implementation is only available
# when demo mode is explicitly enabled.
_lock = threading.Lock()
_system: Any | None = None
_init_error: str | None = None
_is_mock = False
_config: AppConfig | None = None


def configure_application(config: AppConfig) -> None:
    """Store the selected application configuration before initialization."""
    global _config
    _config = config


def _mock_enabled() -> bool:
    return os.environ.get("EDURAG_API_MOCK", "").lower() == "true"


def _chunk_text(text: str, size: int = 8) -> list[str]:
    return [text[i : i + size] for i in range(0, len(text), size)] or [text]


class MockEducationQASystem:
    """In-memory stand-in used when the real backends are unavailable."""

    def __init__(self) -> None:
        self._sessions: dict[str, list[dict[str, Any]]] = {}
        self._counter = 0

    @staticmethod
    def _validate_query(query: str) -> str:
        question = query.strip()
        if not question:
            raise ValueError("query must not be empty")
        return question

    def _append(self, session_id: str, question: str, answer: str) -> list[dict[str, Any]]:
        self._counter += 1
        turn = {
            "id": self._counter,
            "session_id": session_id,
            "question": question,
            "answer": answer,
            "created_at": "now()",
        }
        self._sessions.setdefault(session_id, []).append(turn)
        return list(self._sessions[session_id])

    def query(self, query: str, source_filter: str | None = None, session_id: str | None = None):
        query = self._validate_query(query)
        session_id = session_id or str(uuid4())
        answer = (
            f"[演示模式] 针对「{query}」的示例回答。"
            "真实环境会从 MySQL FAQ 或 RAG 知识库检索后由大模型生成答案。"
        )
        history = self._append(session_id, query, answer)
        return _QAResult(session_id, "mock", answer, history)

    def stream_query(self, query: str, source_filter: str | None = None, session_id: str | None = None):
        query = self._validate_query(query)
        session_id = session_id or str(uuid4())
        answer = (
            f"[演示模式] 针对「{query}」的示例回答。"
            "真实环境会从 MySQL FAQ 或 RAG 知识库检索后由大模型生成答案。"
        )

        def generator() -> Any:
            for piece in _chunk_text(answer):
                yield piece
            self._append(session_id, query, answer)

        return session_id, "mock", generator()

    def get_session_history(self, session_id: str) -> list[dict[str, Any]]:
        return list(self._sessions.get(session_id, []))

    def clear_session_history(self, session_id: str) -> bool:
        self._sessions.pop(session_id, None)
        return True


def _create_system() -> None:
    global _system, _init_error, _is_mock
    if _mock_enabled():
        logger.warning("EDURAG_API_MOCK enabled; using mock QA system")
        _system = MockEducationQASystem()
        _is_mock = True
        _init_error = None
        return
    try:
        from core.system import EducationQASystem

        if _config is None:
            raise RuntimeError("application configuration is not configured")
        _system = EducationQASystem.from_config(_config)
        _is_mock = False
        _init_error = None
        logger.info("EducationQASystem initialized for API server")
    except Exception as exc:  # noqa: BLE001
        _system = None
        _is_mock = False
        _init_error = type(exc).__name__
        log.exception("Education QA system initialization failed")


def ensure_system() -> None:
    global _system, _init_error
    if _system is not None or _init_error is not None:
        return
    with _lock:
        if _system is not None or _init_error is not None:
            return
        _create_system()


def get_system() -> Any | None:
    ensure_system()
    return _system


def _get_system_or_none() -> Any | None:
    ensure_system()
    return _system


def get_system_status() -> dict[str, Any]:
    ensure_system()
    return {
        "ready": _system is not None,
        "mock": _is_mock,
        "error": _init_error,
    }


def _admin_token() -> str | None:
    return os.environ.get("EDURAG_ADMIN_TOKEN") or (
        _config.admin_token if _config is not None else None
    )


def require_admin(authorization: str | None = Header(default=None)) -> None:
    """Require the configured Bearer token for FAQ mutations."""
    token = _admin_token()
    expected = f"Bearer {token}" if token else ""
    if not authorization or not token or not hmac.compare_digest(authorization, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Administrator authorization is required",
            headers={"WWW-Authenticate": "Bearer"},
        )


class InMemoryFAQStore:
    """内存版 FAQ 存储，仅在真实 MySQL 不可用时作为演示模式回退。

    数据保存在进程内存中，服务重启后会清空——仅用于无后端时的演示。
    """

    def __init__(self) -> None:
        self._items: dict[int, dict[str, Any]] = {}
        self._seq = 0

    def list_faqs(self, limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
        items = sorted(self._items.values(), key=lambda x: x["id"])
        return items[offset : offset + limit]

    def insert_faq(self, question: str, answer: str, subject: str | None = None) -> int:
        self._seq += 1
        self._items[self._seq] = {
            "id": self._seq,
            "question": question,
            "answer": answer,
            "subject": subject,
        }
        return self._seq

    def get_faq(self, faq_id: int) -> dict[str, Any] | None:
        return self._items.get(faq_id)

    def update_faq(
        self,
        faq_id: int,
        question: str | None = None,
        answer: str | None = None,
        subject: str | None = None,
    ) -> None:
        item = self._items.get(faq_id)
        if item is None:
            raise ValueError("FAQ 不存在")
        if question is not None:
            item["question"] = question
        if answer is not None:
            item["answer"] = answer
        if subject is not None:
            item["subject"] = subject

    def delete_faq(self, faq_id: int) -> None:
        self._items.pop(faq_id, None)


_faq_store = InMemoryFAQStore()


def get_faq_backend() -> tuple[Any, bool]:
    """返回 FAQ 存储后端：(backend, is_mock)。

    真实 MySQL 可用时返回 MySQL 客户端（is_mock=False）；
    否则返回进程内内存存储（is_mock=True），使演示模式下 FAQ 仍可用。
    """
    system = _get_system_or_none()
    client = getattr(system, "mysql_client", None) if system is not None else None
    if client is None and _is_mock:
        return _faq_store, True
    if client is None:
        raise HTTPException(status_code=503, detail="MySQL backend is unavailable")
    return client, False


def get_mysql_client():
    from fastapi import HTTPException

    system = _get_system_or_none()
    client = getattr(system, "mysql_client", None) if system is not None else None
    if client is None:
        raise HTTPException(status_code=503, detail="MySQL 未就绪，FAQ 管理功能不可用")
    return client


def invalidate_faq_cache() -> None:
    """Drop the cached FAQ questions so the retriever reloads on next query."""
    system = _get_system_or_none()
    sql_system = getattr(system, "sql_system", None) if system is not None else None
    redis_client = getattr(sql_system, "redis_client", None) if sql_system is not None else None
    if redis_client is not None:
        try:
            redis_client.clear_faq_cache()
        except Exception:  # noqa: BLE001
            logger.warning("Failed to clear FAQ Redis cache")
