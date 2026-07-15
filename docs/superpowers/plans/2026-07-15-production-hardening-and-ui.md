# Production Hardening and UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make EduRAG safe for production use and deliver a responsive light education workspace without changing SQL-first/RAG-fallback answer behavior.

**Architecture:** Configuration becomes an injected application dependency shared by `main.py`, FastAPI lifespan/dependencies, and `EducationQASystem`. API runtime mode is explicit: only an opt-in environment flag enables the in-memory demo implementation; production keeps the service unavailable on backend initialization failure. The existing static frontend is redesigned in place while preserving the QA/SSE contracts.

**Tech Stack:** Python 3.10+, FastAPI, Uvicorn, PyMySQL, Milvus, Ollama-compatible OpenAI client, pytest, vanilla HTML/CSS/JavaScript.

## Global Constraints

- Preserve existing SQL-first, SQL-miss-to-RAG, retrieval, prompt, classification, and evaluation behavior.
- Production is the default: backend initialization failure returns HTTP 503; only `EDURAG_API_MOCK=true` or `--mock` enables demo mode.
- FAQ mutation routes require `Authorization: Bearer <EDURAG_ADMIN_TOKEN>`; never expose this token to the frontend.
- All new visible UI strings and all new log messages are English.
- `config.yaml` is local and ignored; `config.example.yaml` contains no secret, no private path, and only portable placeholders.
- Do not add a frontend framework.
- Tests use fakes for MySQL, Redis, Milvus, Ollama, and models.

---

## File Structure

- Modify: `.gitignore`, `config.yaml`, `main.py`, `base/config.py` — safe configuration and runtime injection.
- Create: `config.example.yaml`, `requirements.txt`, `tests/test_api.py` — reproducible configuration and API contract tests.
- Modify: `api/app.py`, `api/deps.py`, `api/routes/faq.py`, `api/routes/qa.py`, `api/schemas.py` — FastAPI lifecycle, health, authorization, and error contracts.
- Modify: `core/system.py`, `core/rag/vector.py`, `core/rag/system.py`, `tests/test_rag_core.py`, `tests/test_rag_core.py`, `tests/test_main.py` — configuration injection, vector deduplication, and corrected CLI regression tests.
- Modify: `web/index.html`, `web/app.js`, `web/style.css` — light, responsive education workspace.

### Task 1: Make Configuration Safe and Reproducible

**Files:**
- Create: `config.example.yaml`, `requirements.txt`
- Modify: `.gitignore`, `base/config.py`, `main.py`, `tests/test_base_config.py`, `tests/test_main.py`

**Interfaces:**
- Produces `AppConfig.admin_token: str | None` from `EDURAG_ADMIN_TOKEN` only.
- Produces `configure_application(config: AppConfig) -> None` in `api.deps` for Task 2.

- [ ] **Step 1: Write failing configuration tests**

Add tests asserting that `load_config` resolves relative paths, environment variables override safe fields, the administrator token is loaded only from `EDURAG_ADMIN_TOKEN`, and `parse_args(["--mock"])` causes mock mode to be configured before initialization:

```python
def test_admin_token_is_read_from_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("EDURAG_ADMIN_TOKEN", "admin-secret")
    config = load_config(write_config(tmp_path))
    assert config.admin_token == "admin-secret"

def test_main_sets_mock_mode_before_initializing(monkeypatch):
    calls = []
    monkeypatch.setattr(main_module, "initialize_system", lambda: calls.append(os.environ.get("EDURAG_API_MOCK")))
    monkeypatch.setattr(main_module, "run_server", lambda **_: None)
    main_module.main(["--mock"])
    assert calls == ["true"]
```

- [ ] **Step 2: Run the focused tests and verify they fail**

Run: `conda run -n edurag python -m pytest tests/test_base_config.py tests/test_main.py -v`

Expected: FAIL because there is no administrator-token configuration and `main()` initializes before applying mock mode.

- [ ] **Step 3: Implement safe config and CLI ordering**

Add an `admin_token` field to `AppConfig`, sourced only from `os.environ.get("EDURAG_ADMIN_TOKEN")`; do not add it to YAML. In `main.main`, set `EDURAG_API_MOCK="true"` before `initialize_system`, retain the loaded `AppConfig`, and call `configure_application(config)` before Uvicorn starts:

```python
def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.mock:
        os.environ["EDURAG_API_MOCK"] = "true"
    config = initialize_app(args.config)
    from api.deps import configure_application
    configure_application(config)
    initialize_system()
    run_server(host=args.host, port=args.port, reload=args.reload)
```

Create `config.example.yaml` with `${EDURAG_MYSQL_PASSWORD}`, `${EDURAG_REDIS_PASSWORD}`, portable relative model paths, and comments naming required environment variables. Add `/config.yaml` to `.gitignore`; retain a local `config.yaml` in the workspace but remove it from Git tracking with `git rm --cached config.yaml`. Create `requirements.txt` from the direct imports used by the application and tests, with version constraints matching the tested environment.

- [ ] **Step 4: Run focused configuration tests**

Run: `conda run -n edurag python -m pytest tests/test_base_config.py tests/test_main.py -v`

Expected: PASS.

- [ ] **Step 5: Commit configuration work**

```bash
git add .gitignore config.example.yaml requirements.txt base/config.py main.py tests/test_base_config.py tests/test_main.py
git rm --cached config.yaml
git commit -m "feat: make application configuration safe"
```

### Task 2: Make API Runtime State, Authentication, and Errors Explicit

**Files:**
- Modify: `api/app.py`, `api/deps.py`, `api/routes/faq.py`, `api/routes/qa.py`, `api/schemas.py`, `core/system.py`
- Create: `tests/test_api.py`

**Interfaces:**
- Consumes `AppConfig` and Task 1 `configure_application(config)`.
- Produces `require_admin(authorization: str | None = Header(default=None)) -> None`.
- Produces `EducationQASystem.from_config(config: AppConfig) -> EducationQASystem`.

- [ ] **Step 1: Write failing API contract tests**

Create fakes and add these behavior tests:

```python
def test_production_backend_failure_returns_503(client_without_system):
    response = client_without_system.post("/api/qa/ask", json={"query": "hello"})
    assert response.status_code == 503

def test_demo_mode_requires_explicit_flag_and_identifies_mock(mock_client):
    response = mock_client.post("/api/qa/ask", json={"query": "hello"})
    assert response.status_code == 200
    assert response.json()["source"] == "mock"

def test_faq_mutation_requires_bearer_token(client, monkeypatch):
    monkeypatch.setenv("EDURAG_ADMIN_TOKEN", "token")
    response = client.post("/api/faq", json={"question": "Q", "answer": "A"})
    assert response.status_code == 401
    response = client.post("/api/faq", headers={"Authorization": "Bearer token"}, json={"question": "Q", "answer": "A"})
    assert response.status_code == 201

def test_stream_validation_returns_http_400(client):
    response = client.post("/api/qa/ask/stream", json={"query": "   "})
    assert response.status_code == 400
```

- [ ] **Step 2: Run API tests and verify they fail**

Run: `conda run -n edurag python -m pytest tests/test_api.py -v`

Expected: FAIL because the application automatically installs mock mode, FAQ writes have no authorization, and stream validation occurs after the response starts.

- [ ] **Step 3: Implement lifecycle and route contracts**

Replace automatic mock fallback in `api.deps._create_system` with explicit mode behavior:

```python
if _mock_enabled():
    _system = MockEducationQASystem()
    _is_mock = True
    return
try:
    _system = EducationQASystem.from_config(_config)
except Exception as exc:
    _system = None
    _init_error = type(exc).__name__
    log.exception("Education QA system initialization failed")
```

Use a FastAPI lifespan context in `api.app` to call `ensure_system()` at startup and close `system.mysql_client` at shutdown when it has `close`. Add `require_admin` using `hmac.compare_digest`, return 401 with `WWW-Authenticate: Bearer` for a missing or wrong token, and attach it as a dependency only to FAQ POST/PUT/DELETE. Validate `limit: int = Query(50, ge=1, le=100)` and `offset: int = Query(0, ge=0)`. Add a Pydantic validator that strips and rejects whitespace-only FAQ fields, and reject an update with no supplied fields.

For streaming, call `system.stream_query(req.query, source_filter=req.source_filter, session_id=req.session_id)` before constructing the generator; map `ValueError` to 400 and unavailable system to 503. Inside an already-started stream, log the exception and emit only:

```python
yield _sse({"type": "error", "message": "Answer generation failed."})
```

- [ ] **Step 4: Run API and core tests**

Run: `conda run -n edurag python -m pytest tests/test_api.py tests/test_core_system.py tests/test_main.py -v`

Expected: PASS.

- [ ] **Step 5: Commit API hardening**

```bash
git add api core/system.py tests/test_api.py tests/test_core_system.py tests/test_main.py
git commit -m "feat: harden API runtime and FAQ access"
```

### Task 3: Fix Vector Batch Deduplication and RAG CLI Regression

**Files:**
- Modify: `core/rag/vector.py`, `core/rag/system.py`, `tests/test_rag_core.py`

**Interfaces:**
- Produces `VectorStore.add_documents(documents)` that upserts each deterministic document ID at most once per call.

- [ ] **Step 1: Write failing regression tests**

Add a vector fake that records embedded strings and upsert rows, then assert duplicate documents generate one embedding/upsert row:

```python
def test_add_documents_deduplicates_primary_keys_in_one_batch(vector_store, document):
    vector_store.add_documents([document, document])
    assert vector_store.embedding_function.calls == [[document.page_content]]
    assert len(vector_store.client.upserted[0]["data"]) == 1
```

Update the CLI test to patch only symbols still owned by `core.rag.system`; remove the obsolete `setup_logger` monkeypatch and retain the assertions for training, indexing, stream, sync, and exit flow.

- [ ] **Step 2: Run failing focused tests**

Run: `conda run -n edurag python -m pytest tests/test_rag_core.py -v`

Expected: the existing CLI tests fail on the obsolete `setup_logger` patch, and the new duplicate test fails before deduplication.

- [ ] **Step 3: Implement minimal deduplication**

Compute the existing ID before collecting text/embedding inputs, retain the first `Document` for each unseen ID, and embed the resulting unique list:

```python
unique_documents = []
seen_ids = set()
for document in documents:
    document_id = self._document_id(document)
    if document_id not in seen_ids:
        seen_ids.add(document_id)
        unique_documents.append((document_id, document))
```

Build `texts`, embeddings, and upsert data from `unique_documents`, using the already-computed ID. Keep original input order and existing metadata validation.

- [ ] **Step 4: Run RAG regressions**

Run: `conda run -n edurag python -m pytest tests/test_rag_core.py tests/test_rag_utils.py -v`

Expected: PASS.

- [ ] **Step 5: Commit retrieval reliability fixes**

```bash
git add core/rag/vector.py tests/test_rag_core.py
git commit -m "fix: deduplicate vector upsert batches"
```

### Task 4: Redesign the Light Education Workspace

**Files:**
- Modify: `web/index.html`, `web/app.js`, `web/style.css`

**Interfaces:**
- Consumes existing `/health`, `/api/qa/ask`, `/api/qa/ask/stream`, session history, and FAQ list contracts.
- Produces an English, keyboard-accessible, responsive chat workspace with demo-state visibility.

- [ ] **Step 1: Create manual UI acceptance checklist**

Add this checklist as a comment at the top of `web/app.js` during implementation:

```javascript
// Manual acceptance: desktop rail + workspace; mobile drawer; light contrast;
// keyboard send; clear focus ring; demo warning; suggested prompts; source chips;
// streaming response; sync fallback; session history; FAQ prompt insertion.
```

- [ ] **Step 2: Verify the current page does not meet the target**

Run: `conda run -n edurag python main.py --mock --port 8001`

Expected: the current dark, Chinese interface lacks the target light workspace, visible demo warning, mobile navigation drawer, and suggested prompts.

- [ ] **Step 3: Replace the static layout and interactions**

Use semantic `<header>`, `<nav>`, `<main>`, `<section>`, and `<form>` elements. Add a mobile navigation toggle with `aria-expanded`, an always-visible runtime status banner, source-filter input, three suggested-prompt buttons, and a send button with an accessible busy state. Keep `textContent` for dynamic answer text and FAQ content; never insert model output with `innerHTML`.

Use CSS variables for a white/slate base, navy text, blue/teal accents, 44px minimum tap targets, `:focus-visible` rings, a 1024px desktop two-column layout, and a 760px drawer layout. The visual hierarchy must use cards and restrained borders rather than gradients as the primary decoration.

Update JavaScript to persist only session metadata in `localStorage`, load history on selection, display source chips from SSE metadata, pass a non-empty `source_filter` field, and expose a generic server error without retrying a failed stream as a second request after any token has arrived.

- [ ] **Step 4: Verify the static frontend manually**

Run: `conda run -n edurag python main.py --mock --port 8001`

Verify at `http://127.0.0.1:8001`: suggested prompts submit, stream text appears, the demo warning is visible, keyboard Enter sends, Shift+Enter adds a line, narrow viewport opens/closes navigation, and focus indicators are visible.

- [ ] **Step 5: Commit UI redesign**

```bash
git add web/index.html web/app.js web/style.css
git commit -m "feat: redesign education chat workspace"
```

### Task 5: Full Verification and Documentation

**Files:**
- Modify: `README.md` if present; otherwise create `README.md`

**Interfaces:**
- Documents setup using `config.example.yaml`, environment variables, mock mode, and protected FAQ write requests.

- [ ] **Step 1: Add concise setup and security documentation**

Document these commands:

```bash
cp config.example.yaml config.yaml
export EDURAG_MYSQL_PASSWORD='replace-me'
export EDURAG_REDIS_PASSWORD='replace-me'
export EDURAG_ADMIN_TOKEN='replace-with-a-long-random-token'
conda activate edurag
python -m pytest -q
python main.py --mock
```

State that `config.yaml` must not be committed and that production requires all configured backends.

- [ ] **Step 2: Run the complete suite**

Run: `conda run -n edurag python -m pytest -q`

Expected: PASS with zero test failures.

- [ ] **Step 3: Check repository safety**

Run: `git ls-files config.yaml && git check-ignore config.yaml && git diff --check`

Expected: no tracked `config.yaml`, `config.yaml` is ignored, and diff check has no output.

- [ ] **Step 4: Commit documentation and verification changes**

```bash
git add README.md
git commit -m "docs: document secure EduRAG setup"
```

## Self-Review

- Spec coverage: Tasks 1–2 cover mode selection, configuration propagation, 503 behavior, secret handling, FAQ authentication, validation, SSE errors, and lifecycle. Task 3 covers duplicate Milvus IDs and the failing CLI tests. Task 4 covers the full frontend design and accessibility requirements. Task 5 covers reproducibility and final verification.
- Implementation completeness: every task contains concrete files, tests, implementation guidance, commands, and expected outcomes.
- Type consistency: Task 1 configures the `AppConfig` consumed by Task 2; Task 2 preserves the existing QA/SSE contracts consumed by Task 4; Task 3 leaves retrieval interfaces unchanged.
