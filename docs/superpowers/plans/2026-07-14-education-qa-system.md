# Education QA System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a session-aware coordinator that returns SQL FAQ answers first and uses RAG only for SQL misses.

**Architecture:** `MySQLClient` owns the persistent `conversations` table and exposes public persistence methods. `EducationQASystem` composes the existing SQL and RAG systems, persists completed turns, and offers synchronous and streaming paths without changing either backend's answer behavior.

**Tech Stack:** Python 3.10+, PyMySQL, pytest, existing SQL BM25 and RAG systems.

## Global Constraints

- Do not alter SQL answer matching, RAG retrieval, prompts, or LLM generation behavior.
- Do not pass history into RAG prompts or modify `main.py`.
- Use English logs/comments and parameterized values in every SQL statement.
- Tests must use fakes; no test may require MySQL, Redis, Milvus, Ollama, or model files.
- Fetch no more than five turns and return them in chronological order.

---

## File Structure

- Modify: `core/sql/db.py` — public `conversations` table persistence API.
- Modify: `core/system.py` — response model and SQL-first orchestration.
- Modify: `tests/test_sql_db.py` — database-level conversation query tests.
- Create: `tests/test_core_system.py` — coordinator behavior tests using fakes.

### Task 1: Add Conversation Persistence to `MySQLClient`

**Files:**
- Modify: `core/sql/db.py:19-222`
- Test: `tests/test_sql_db.py`

**Interfaces:**
- Produces `create_conversation_table(table_name: str = "conversations") -> None`.
- Produces `append_conversation_turn(session_id: str, question: str, answer: str, table_name: str = "conversations") -> int`.
- Produces `fetch_recent_conversations(session_id: str, limit: int = 5, table_name: str = "conversations") -> list[dict[str, Any]]`.
- Produces `clear_conversations(session_id: str, table_name: str = "conversations") -> bool`.

- [ ] **Step 1: Write the failing database tests**

Append this code to `tests/test_sql_db.py`:

```python
def test_mysql_client_conversation_operations():
    cursor = FakeCursor(lastrowid=8, rows=[
        {"id": 3, "session_id": "s1", "question": "old", "answer": "a"},
        {"id": 4, "session_id": "s1", "question": "new", "answer": "b"},
    ])
    connection = FakeConnection(cursor)
    client = MySQLClient(connection=connection, auto_import=False)

    client.create_conversation_table()
    assert "CREATE TABLE IF NOT EXISTS `conversations`" in cursor.executed[0][0]
    assert "INDEX idx_conversations_session_created" in cursor.executed[0][0]

    assert client.append_conversation_turn("s1", "question", "answer") == 8
    assert cursor.executed[1][1] == ("s1", "question", "answer")

    assert client.fetch_recent_conversations("s1") == [
        {"id": 3, "session_id": "s1", "question": "old", "answer": "a"},
        {"id": 4, "session_id": "s1", "question": "new", "answer": "b"},
    ]
    assert cursor.executed[2][1] == ("s1", 5)

    assert client.clear_conversations("s1") is True
    assert cursor.executed[3][1] == ("s1",)
    assert connection.commit_count == 3
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `conda run -n edurag pytest tests/test_sql_db.py -k conversation -v`

Expected: FAIL because the four public conversation methods do not exist.

- [ ] **Step 3: Write the minimal database implementation**

Add these methods before `MySQLClient.close` in `core/sql/db.py`:

```python
    def create_conversation_table(self, table_name: str = "conversations") -> None:
        table = validate_identifier(table_name)
        sql = f"""
        CREATE TABLE IF NOT EXISTS `{table}` (
            id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
            session_id VARCHAR(255) NOT NULL,
            question TEXT NOT NULL,
            answer LONGTEXT NOT NULL,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_conversations_session_created (session_id, created_at)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """
        self._execute(sql, commit=True)
        log.info("Ensured MySQL conversation table exists: table={}", table)

    def append_conversation_turn(self, session_id: str, question: str, answer: str, table_name: str = "conversations") -> int:
        table = validate_identifier(table_name)
        cursor = self._execute(
            f"INSERT INTO `{table}` (session_id, question, answer) VALUES (%s, %s, %s)",
            (session_id, question, answer),
            commit=True,
        )
        return int(cursor.lastrowid)

    def fetch_recent_conversations(self, session_id: str, limit: int = 5, table_name: str = "conversations") -> list[dict[str, Any]]:
        if limit <= 0:
            raise ValueError("limit must be greater than 0")
        table = validate_identifier(table_name)
        rows = [dict(row) for row in self._fetch_all(
            f"SELECT id, session_id, question, answer, created_at FROM `{table}` "
            "WHERE session_id = %s ORDER BY id DESC LIMIT %s",
            (session_id, limit),
        )]
        rows.reverse()
        return rows

    def clear_conversations(self, session_id: str, table_name: str = "conversations") -> bool:
        table = validate_identifier(table_name)
        self._execute(f"DELETE FROM `{table}` WHERE session_id = %s", (session_id,), commit=True)
        return True
```

- [ ] **Step 4: Run the database tests and verify they pass**

Run: `conda run -n edurag pytest tests/test_sql_db.py -v`

Expected: PASS.

- [ ] **Step 5: Commit the database layer**

```bash
git add core/sql/db.py tests/test_sql_db.py
git commit -m "feat: persist conversation turns in mysql"
```

### Task 2: Implement SQL-First Education QA Orchestration

**Files:**
- Modify: `core/system.py`
- Create: `tests/test_core_system.py`

**Interfaces:**
- Consumes Task 1 methods and existing `MySqlQASystem.query`, `RAGSystem.generate_answer`, and `RAGSystem.generate_answer_stream`.
- Produces immutable `QAResponse(session_id, source, answer, history)`.
- Produces `query(...) -> QAResponse`, `stream_query(...) -> tuple[str, str, Iterator[str]]`, `get_session_history`, and `clear_session_history`.

- [ ] **Step 1: Write the failing coordinator tests**

Create `tests/test_core_system.py` with these fakes and tests:

```python
import pytest

from core.system import EducationQASystem


class FakeDatabase:
    def __init__(self):
        self.turns = []
        self.initialized = False

    def create_conversation_table(self):
        self.initialized = True

    def append_conversation_turn(self, session_id, question, answer):
        self.turns.append({"session_id": session_id, "question": question, "answer": answer})
        return len(self.turns)

    def fetch_recent_conversations(self, session_id, limit=5):
        return [row for row in self.turns if row["session_id"] == session_id][-limit:]

    def clear_conversations(self, session_id):
        self.turns = [row for row in self.turns if row["session_id"] != session_id]
        return True


class FakeSQL:
    fallback_answer = "在 sql 中没有找到答案"

    def __init__(self, answer):
        self.answer = answer

    def query(self, query):
        return self.answer


class FakeRAG:
    def __init__(self):
        self.calls = []

    def generate_answer(self, query, source_filter=None):
        self.calls.append(("answer", query, source_filter))
        return "rag answer"

    def generate_answer_stream(self, query, source_filter=None):
        self.calls.append(("stream", query, source_filter))
        yield "rag "
        yield "answer"


def build(sql_answer="sql answer"):
    database = FakeDatabase()
    system = EducationQASystem(sql_system=FakeSQL(sql_answer), rag_system=FakeRAG(), mysql_client=database)
    return system, database


def test_sql_hit_has_priority_and_is_persisted():
    system, database = build()
    response = system.query("faq", session_id="s1")
    assert (response.session_id, response.source, response.answer) == ("s1", "sql", "sql answer")
    assert database.turns == [{"session_id": "s1", "question": "faq", "answer": "sql answer"}]


def test_sql_miss_uses_rag_and_forwards_filter():
    system, _ = build("在 sql 中没有找到答案")
    response = system.query("question", source_filter="python", session_id="s1")
    assert response.source == "rag"
    assert system.rag_system.calls == [("answer", "question", "python")]


def test_streaming_persists_only_after_full_consumption():
    system, database = build("在 sql 中没有找到答案")
    session_id, source, chunks = system.stream_query("question", session_id="s1")
    assert (session_id, source, database.turns) == ("s1", "rag", [])
    assert "".join(chunks) == "rag answer"
    assert database.turns[-1]["answer"] == "rag answer"


def test_history_is_limited_and_clearable_and_blank_query_is_rejected():
    system, _ = build()
    for index in range(6):
        system.query(f"q{index}", session_id="s1")
    assert [row["question"] for row in system.get_session_history("s1")] == ["q1", "q2", "q3", "q4", "q5"]
    assert system.clear_session_history("s1") is True
    assert system.get_session_history("s1") == []
    with pytest.raises(ValueError, match="query must not be empty"):
        system.query("  ")
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `conda run -n edurag pytest tests/test_core_system.py -v`

Expected: FAIL because `core/system.py` is only a stub.

- [ ] **Step 3: Replace the coordinator stub**

Replace `core/system.py` with:

```python
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

    def __init__(self, *, sql_system: MySqlQASystem | Any | None = None, rag_system: RAGSystem | Any | None = None, mysql_client: MySQLClient | Any | None = None) -> None:
        self.sql_system = sql_system or MySqlQASystem()
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

    def update_session_history(self, session_id: str, question: str, answer: str) -> list[dict[str, Any]]:
        self.mysql_client.append_conversation_turn(session_id, question, answer)
        return self._fetch_recent_history(session_id)

    def get_session_history(self, session_id: str) -> list[dict[str, Any]]:
        return self._fetch_recent_history(session_id)

    def clear_session_history(self, session_id: str) -> bool:
        return self.mysql_client.clear_conversations(session_id)

    def query(self, query: str, source_filter: str | None = None, session_id: str | None = None) -> QAResponse:
        question, active_session = self._validate_query(query), self._session_id(session_id)
        sql_answer = self.sql_system.query(question)
        answer, source = (sql_answer, "sql") if self._is_sql_hit(sql_answer) else (self.rag_system.generate_answer(question, source_filter=source_filter), "rag")
        return QAResponse(active_session, source, answer, self.update_session_history(active_session, question, answer))

    def stream_query(self, query: str, source_filter: str | None = None, session_id: str | None = None) -> tuple[str, str, Iterator[str]]:
        question, active_session = self._validate_query(query), self._session_id(session_id)
        sql_answer = self.sql_system.query(question)
        if self._is_sql_hit(sql_answer):
            return active_session, "sql", self._persisted_stream(active_session, question, iter((sql_answer,)))
        return active_session, "rag", self._persisted_stream(active_session, question, self.rag_system.generate_answer_stream(question, source_filter=source_filter))

    def _persisted_stream(self, session_id: str, question: str, chunks: Iterator[str]) -> Iterator[str]:
        parts: list[str] = []
        for chunk in chunks:
            parts.append(chunk)
            yield chunk
        self.update_session_history(session_id, question, "".join(parts))
```

- [ ] **Step 4: Run the coordinator tests and verify they pass**

Run: `conda run -n edurag pytest tests/test_core_system.py -v`

Expected: PASS.

- [ ] **Step 5: Run relevant regressions**

Run: `conda run -n edurag pytest tests/test_sql_db.py tests/test_sql_system.py tests/test_rag_core.py tests/test_core_system.py -v`

Expected: PASS. Do not run or edit `tests/test_main.py`; `main.py` has an unrelated active user change.

- [ ] **Step 6: Commit the coordinator**

```bash
git add core/system.py tests/test_core_system.py
git commit -m "feat: add education QA orchestration"
```

## Self-Review

- Spec coverage: Task 1 delivers public parameterized persistence. Task 2 covers SQL priority, RAG fallback, source-filter forwarding, generated session IDs, sync and streaming answers, complete-turn persistence, five-turn history, clearing, and blank query validation.
- Implementation completeness: every task includes concrete test code, implementation code, commands, and expected outcomes.
- Type consistency: Task 1 exports the exact methods consumed by Task 2; Task 2 uses the existing SQL and RAG signatures.
