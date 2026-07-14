# Education QA System Design

**Date:** 2026-07-14

## Goal

Implement `core/system.py` as the application-facing coordinator for the
existing SQL and RAG question-answering systems. It must preserve the current
answering behavior: prefer an exact SQL knowledge-base answer and otherwise
fall back to RAG. It must also provide stable conversation sessions backed by
MySQL, including support for streaming RAG answers.

## Scope

### In scope

- Add public conversation-history operations to `MySQLClient`.
- Implement `EducationQASystem` in `core/system.py`.
- Route every query to SQL QA first, then RAG only when SQL has no answer.
- Expose synchronous and streaming query APIs.
- Persist completed question/answer turns and retain only the five newest
  turns per session.
- Add focused unit tests using fakes; tests must not require MySQL, Milvus,
  Ollama, or local model files.

### Out of scope

- Changing SQL answer matching, retrieval, classification, prompts, or LLM
  generation behavior.
- Passing history into RAG prompts. The current RAG interface does not accept
  history, and adding it would change retrieval and answer behavior.
- Altering the application entry point in `main.py`.

## Components and Responsibilities

### `MySQLClient`

`core/sql/db.py` owns all database details. It will expose public methods to:

- create the `conversations` table if it is absent;
- append a completed conversation turn;
- fetch the newest five turns for one session, returned in chronological order;
- clear all turns for one session.

The table stores an auto-increment row id, `session_id`, `question`, `answer`,
and a creation timestamp. An index on `(session_id, created_at)` supports the
bounded-history lookup. SQL statements remain parameterized.

### `EducationQASystem`

`core/system.py` owns orchestration only. Its constructor accepts optional
instances of `MySqlQASystem`, `RAGSystem`, and `MySQLClient` for testability;
when they are omitted it constructs the project defaults. It initializes the
conversation table through the public database API.

The system regards the existing SQL fallback answer (`"在 sql 中没有找到答案"`)
as a miss. Any other SQL answer is a hit and is returned unchanged. This keeps
the existing SQL matching logic authoritative and avoids duplicating its BM25
threshold in the coordinator.

## Public API

`QAResponse` is an immutable response value containing:

- `session_id`: supplied session id or a newly generated UUID;
- `source`: `"sql"` or `"rag"`;
- `answer`: final non-streaming answer;
- `history`: the five most recent persisted turns after the current answer is
  saved.

`EducationQASystem` provides:

- `query(query, source_filter=None, session_id=None) -> QAResponse`
  for a complete answer;
- `stream_query(query, source_filter=None, session_id=None) ->
  tuple[str, str, Iterator[str]]`, returning session id, answer source, and an
  iterator of answer chunks;
- `get_session_history(session_id) -> list[dict[str, str]]`;
- `clear_session_history(session_id) -> bool`.

For a SQL hit, `stream_query` yields the single SQL answer chunk. For a RAG
miss, it delegates chunks to `RAGSystem.generate_answer_stream`.

## Request Flow

1. Resolve `session_id`; generate `uuid4()` when none is supplied.
2. Call `MySqlQASystem.query(query)`.
3. On SQL hit, use that answer and mark the source as `sql`.
4. On SQL miss, call either `RAGSystem.generate_answer` or
   `RAGSystem.generate_answer_stream`, forwarding the optional source filter,
   and mark the source as `rag`.
5. Persist the finished `(session_id, query, answer)` turn.
6. For the synchronous API, return a `QAResponse` with refreshed history.

For streaming, chunks are accumulated internally while being yielded. The turn
is persisted only after the iterator is exhausted successfully. If a caller
stops consuming early or RAG raises an exception, no partial answer is stored.
This prevents incomplete turns from becoming later session context.

## Error Handling

- Empty or whitespace-only queries raise `ValueError` before either backend is
  called.
- Backend exceptions are allowed to propagate; the coordinator does not mask
  existing SQL/RAG failure behavior.
- History methods delegate their return values to the database layer.

## Tests

Unit tests will cover:

- SQL-answer priority and persistence;
- RAG fallback with `source_filter` forwarding;
- streaming RAG chunks and persistence after completion;
- generated session ids;
- chronological five-turn history retention;
- history clearing;
- invalid empty queries.

