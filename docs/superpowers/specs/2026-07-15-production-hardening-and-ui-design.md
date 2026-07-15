# Production Hardening and Education UI Design

**Date:** 2026-07-15

## Goal

Make EduRAG safe and predictable for production use while replacing the
current dark chat interface with a responsive, accessible, light education
workspace. The existing SQL-first and RAG-fallback answer behavior remains
unchanged.

## Scope

### In scope

- Fix failing RAG workflow tests.
- Make command-line configuration and mock mode deterministic.
- Require an explicit demo/mock mode; production backend failures return 503.
- Protect FAQ create, update, and delete APIs with an administrator Bearer
  token stored only in an environment variable.
- Validate API pagination and request bodies, and avoid exposing internal
  exception text in SSE responses.
- Prevent duplicate primary keys in a single Milvus upsert batch.
- Replace tracked secrets and machine-specific configuration with a tracked
  `config.example.yaml` and an ignored local `config.yaml`.
- Add reproducible Python dependency metadata and focused automated tests.
- Redesign the static frontend as a light education workspace without adding a
  frontend framework.

### Out of scope

- End-user accounts, identity providers, roles beyond one administrator
  token, rate limiting, or multi-tenant data isolation.
- Changing FAQ retrieval thresholds, query classification, prompt content,
  embedding models, reranking, or RAG scoring methodology.
- Changing the persistence schema except where existing APIs already create
  and use conversation records.

## Runtime Modes and Configuration

The application has two explicit modes:

- **Production (default):** `EducationQASystem` must initialize all required
  backends. Any initialization failure leaves the API unavailable; QA and FAQ
  routes return HTTP 503 and `/health` returns `{"status": "degraded"}`.
- **Demo:** set `EDURAG_API_MOCK=true` or pass `--mock`. Only this mode creates
  `MockEducationQASystem`; every response identifies `source` as `mock` and
  the UI displays a persistent demo warning.

`main.py` must apply `--mock` before system initialization. `--config` must be
passed to the dependency layer and then to `EducationQASystem`/RAG factories,
so all backends use the selected configuration. Application startup uses the
FastAPI lifespan hook, and shutdown closes the MySQL connection when a real
system was initialized.

`config.yaml` becomes an ignored local file. A new committed
`config.example.yaml` contains only safe placeholders and relative paths.
All operational secrets and the FAQ administrator token are supplied through
environment variables. The repository adds a `requirements.txt` with the
project's direct runtime and test dependencies.

## API Behavior and Security

`GET /api/qa/*` and `POST /api/qa/ask*` remain public. `POST`, `PUT`, and
`DELETE /api/faq*` require an `Authorization: Bearer <token>` header whose
token securely matches `EDURAG_ADMIN_TOKEN`; absence or mismatch returns 401.
No token is embedded in the frontend.

FAQ pagination accepts `1 <= limit <= 100` and `offset >= 0`. FAQ creation
and updates reject whitespace-only questions and answers. Updates must change
at least one field.

The streaming endpoint validates the query and resolves the initial stream
before returning `StreamingResponse`. Validation failures return HTTP 400 and
backend unavailability returns 503. Once a stream starts, exceptions emit a
generic English SSE error message, while full details are kept in server logs.

## Vector Ingestion

Before embedding/upserting, `VectorStore.add_documents` deduplicates documents
by the existing deterministic document ID while preserving the first input
order. It embeds and upserts the unique documents only. This prevents Milvus
from rejecting one batch with duplicate primary keys.

## Frontend Design

The static frontend retains the existing API contracts and SSE protocol.

- Use a warm light background, deep navy text, restrained blue/teal accents,
  generous whitespace, and high-contrast focus/hover states.
- Desktop layout: narrow navigation rail for brand, new conversation, session
  list, FAQ prompts, and a visible runtime-mode indicator; a centered learning
  workspace for the conversation.
- Empty state: a concise welcome panel with three clickable suggested
  questions, explaining that FAQ is checked before the knowledge base.
- Conversation cards: distinguish learner and assistant content, show compact
  answer-source chips, preserve streaming typing feedback, and make error
  states understandable.
- Composer: attach the source filter as an optional compact control, offer an
  accessible send button, keyboard operation, and a non-blocking busy state.
- Mobile: collapse the navigation into an accessible drawer/toggle without
  losing sessions, FAQ suggestions, or service-mode visibility.
- All visible UI text and all new logs are English.

## Tests and Verification

Tests will cover:

- RAG CLI workflow tests updated to its actual logger initialization API.
- Production initialization failure, explicit mock behavior, `--mock` order,
  and selected config propagation.
- FAQ administrator authorization, validation, and pagination bounds.
- SSE HTTP 400 validation, generic streaming failure payload, and 503 backend
  state.
- Same-batch vector-ID deduplication.
- Existing SQL/RAG/core-system test regressions.
- A full `python -m pytest -q` run with no failures.

## Acceptance Criteria

- No plaintext database passwords or personal absolute model paths are tracked.
- Production API callers never receive mock answers after backend failure.
- FAQ mutations are inaccessible without the administrator token.
- The test suite passes in the `edurag` Conda environment.
- Duplicate documents no longer cause a duplicate-primary-key upsert batch.
- The frontend is visibly redesigned as a responsive, light education
  workspace and continues to support synchronous fallback and SSE streaming.
