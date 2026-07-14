from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from base.logger import logger
from core.rag.system import RAGSystem
from core.sql.db import MySQLClient
from core.sql.system import MySqlQASystem


log = logger.bind(module=__name__)


@dataclass(frozen=True)
class QAResponse:
    session_id: str
    source: str
    answer: str
    history: list[dict[str, Any]]


class EducationQASystem:
    """Coordinate SQL FAQ answers, RAG fallback, and persistent sessions."""

    def __init__(
        self,
        *,
        sql_system: MySqlQASystem | Any | None = None,
        rag_system: RAGSystem | Any | None = None,
        mysql_client: MySQLClient | Any | None = None,
    ) -> None:
        self.sql_system = sql_system or (
            MySqlQASystem(mysql_client=mysql_client)
            if mysql_client is not None
            else MySqlQASystem()
        )
        self.mysql_client = mysql_client or self.sql_system.mysql_client
        self.rag_system = rag_system or RAGSystem.from_config()
        self.init_conversation_table()

    def init_conversation_table(self) -> None:
        self.mysql_client.create_conversation_table()

    @staticmethod
    def _validate_query(query: str) -> str:
        query = query.strip()
        if not query:
            raise ValueError("query must not be empty")
        return query

    def _session_id(self, session_id: str | None) -> str:
        return session_id or str(uuid4())

    def _is_sql_hit(self, answer: str) -> bool:
        return answer != self.sql_system.fallback_answer

    def _fetch_recent_history(self, session_id: str) -> list[dict[str, Any]]:
        return self.mysql_client.fetch_recent_conversations(session_id, limit=5)

    def update_session_history(
        self, session_id: str, question: str, answer: str
    ) -> list[dict[str, Any]]:
        self.mysql_client.append_conversation_turn(session_id, question, answer)
        return self._fetch_recent_history(session_id)

    def get_session_history(self, session_id: str) -> list[dict[str, Any]]:
        return self._fetch_recent_history(session_id)

    def clear_session_history(self, session_id: str) -> bool:
        return self.mysql_client.clear_conversations(session_id)

    def query(
        self,
        query: str,
        source_filter: str | None = None,
        session_id: str | None = None,
    ) -> QAResponse:
        question, active_session = self._validate_query(query), self._session_id(session_id)
        sql_answer = self.sql_system.query(question)
        answer, source = (
            (sql_answer, "sql")
            if self._is_sql_hit(sql_answer)
            else (
                self.rag_system.generate_answer(question, source_filter=source_filter),
                "rag",
            )
        )
        return QAResponse(
            active_session,
            source,
            answer,
            self.update_session_history(active_session, question, answer),
        )

    def stream_query(
        self,
        query: str,
        source_filter: str | None = None,
        session_id: str | None = None,
    ) -> tuple[str, str, Iterator[str]]:
        question, active_session = self._validate_query(query), self._session_id(session_id)
        sql_answer = self.sql_system.query(question)
        if self._is_sql_hit(sql_answer):
            return active_session, "sql", self._persisted_stream(
                active_session, question, iter((sql_answer,))
            )
        generate_answer_stream = getattr(
            self.rag_system, "generate_answer_stream", None
        )
        chunks = (
            generate_answer_stream(question, source_filter=source_filter)
            if callable(generate_answer_stream)
            else iter(
                (
                    self.rag_system.generate_answer(
                        question, source_filter=source_filter
                    ),
                )
            )
        )
        return active_session, "rag", self._persisted_stream(
            active_session,
            question,
            chunks,
        )

    def _persisted_stream(
        self, session_id: str, question: str, chunks: Iterator[str]
    ) -> Iterator[str]:
        parts: list[str] = []
        for chunk in chunks:
            parts.append(chunk)
            yield chunk
        self.update_session_history(session_id, question, "".join(parts))
