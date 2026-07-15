from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from fastapi import HTTPException

import api.deps as deps
from api.app import app
from core.system import EducationQASystem
import core.system as core_system


class FakeSystem:
    def __init__(self) -> None:
        self.mysql_client = None

    def query(self, query, source_filter=None, session_id=None):
        if not query.strip():
            raise ValueError("query must not be empty")
        return SimpleNamespace(
            session_id=session_id or "session-1",
            source="fake",
            answer="answer",
            history=[],
        )

    def stream_query(self, query, source_filter=None, session_id=None):
        if not query.strip():
            raise ValueError("query must not be empty")
        return session_id or "session-1", "fake", iter(("answer",))

    def get_session_history(self, session_id):
        return []

    def clear_session_history(self, session_id):
        return True


@pytest.fixture
def api_runtime(monkeypatch):
    monkeypatch.setattr(deps, "_system", None)
    monkeypatch.setattr(deps, "_init_error", None)
    monkeypatch.setattr(deps, "_is_mock", False)
    monkeypatch.setattr(deps, "_config", None)
    monkeypatch.setattr(deps, "_faq_store", deps.InMemoryFAQStore())
    monkeypatch.delenv("EDURAG_API_MOCK", raising=False)
    monkeypatch.delenv("EDURAG_ADMIN_TOKEN", raising=False)
    return deps


@pytest.fixture
def client(api_runtime):
    system = FakeSystem()
    system.mysql_client = api_runtime._faq_store
    api_runtime._system = system
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def mock_client(api_runtime, monkeypatch):
    monkeypatch.setenv("EDURAG_API_MOCK", "true")
    with TestClient(app) as test_client:
        yield test_client


def test_production_backend_failure_returns_503(api_runtime, monkeypatch):
    class ExplodingSystem:
        def __init__(self):
            raise OSError("backend unavailable")

        @classmethod
        def from_config(cls, config):
            raise OSError("backend unavailable")

    monkeypatch.setattr(core_system, "EducationQASystem", ExplodingSystem)
    api_runtime.configure_application(object())

    with TestClient(app) as client_without_system:
        response = client_without_system.post("/api/qa/ask", json={"query": "hello"})

    assert response.status_code == 503


def test_reload_worker_loads_selected_config_from_environment(
    api_runtime, monkeypatch, tmp_path
):
    config_path = tmp_path / "selected.yaml"
    config_path.write_text(
        "mysql:\n  host: reload-db\nredis: {}\nlog: {}\n",
        encoding="utf-8",
    )
    captured = []

    class CapturingSystem:
        @classmethod
        def from_config(cls, config):
            captured.append(config)
            return cls()

    monkeypatch.setattr(core_system, "EducationQASystem", CapturingSystem)
    monkeypatch.setenv("EDURAG_CONFIG_PATH", str(config_path))

    api_runtime.ensure_system()

    assert captured[0].mysql.host == "reload-db"


def test_demo_mode_requires_explicit_flag_and_identifies_mock(mock_client):
    response = mock_client.post("/api/qa/ask", json={"query": "hello"})

    assert response.status_code == 200
    assert response.json()["source"] == "mock"


def test_demo_mode_rejects_non_true_environment_values(api_runtime, monkeypatch):
    monkeypatch.setenv("EDURAG_API_MOCK", "yes")

    with TestClient(app) as test_client:
        response = test_client.post("/api/qa/ask", json={"query": "hello"})

    assert response.status_code == 503


def test_mock_mode_rejects_blank_queries(mock_client):
    response = mock_client.post("/api/qa/ask", json={"query": "   "})

    assert response.status_code == 400


def test_faq_mutation_requires_bearer_token(client, monkeypatch):
    monkeypatch.setenv("EDURAG_ADMIN_TOKEN", "token")

    response = client.post("/api/faq", json={"question": "Q", "answer": "A"})
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"

    response = client.post(
        "/api/faq",
        headers={"Authorization": "Bearer token"},
        json={"question": "Q", "answer": "A"},
    )
    assert response.status_code == 201


def test_stream_validation_returns_http_400(client):
    response = client.post("/api/qa/ask/stream", json={"query": "   "})

    assert response.status_code == 400


def test_sync_backend_error_returns_http_503(api_runtime):
    class FailingSystem(FakeSystem):
        def query(self, query, source_filter=None, session_id=None):
            raise RuntimeError("backend unavailable")

    api_runtime._system = FailingSystem()

    with TestClient(app, raise_server_exceptions=False) as test_client:
        response = test_client.post("/api/qa/ask", json={"query": "hello"})

    assert response.status_code == 503


def test_initial_stream_backend_error_returns_http_503(api_runtime):
    class FailingSystem(FakeSystem):
        def stream_query(self, query, source_filter=None, session_id=None):
            raise RuntimeError("backend unavailable")

    api_runtime._system = FailingSystem()

    with TestClient(app, raise_server_exceptions=False) as test_client:
        response = test_client.post("/api/qa/ask/stream", json={"query": "hello"})

    assert response.status_code == 503


@pytest.mark.parametrize("authorization", ["Bearer tést", b"Bearer token"])
def test_faq_auth_rejects_malformed_authorization_without_raising(
    monkeypatch, authorization
):
    monkeypatch.setenv("EDURAG_ADMIN_TOKEN", "token")

    with pytest.raises(HTTPException) as exc_info:
        deps.require_admin(authorization=authorization)

    assert exc_info.value.status_code == 401


def test_faq_validation_and_pagination_bounds(client, monkeypatch):
    monkeypatch.setenv("EDURAG_ADMIN_TOKEN", "token")
    headers = {"Authorization": "Bearer token"}

    assert client.get("/api/faq?limit=0").status_code == 422
    assert client.get("/api/faq?limit=101").status_code == 422
    assert client.get("/api/faq?offset=-1").status_code == 422
    assert client.post(
        "/api/faq", headers=headers, json={"question": "  ", "answer": "A"}
    ).status_code == 422
    assert client.put("/api/faq/1", headers=headers, json={}).status_code == 422


def test_started_stream_hides_internal_errors(api_runtime):
    class FailingStreamSystem(FakeSystem):
        def stream_query(self, query, source_filter=None, session_id=None):
            def chunks():
                yield "partial"
                raise RuntimeError("secret backend detail")

            return "session-1", "fake", chunks()

    api_runtime._system = FailingStreamSystem()

    with TestClient(app) as test_client:
        response = test_client.post("/api/qa/ask/stream", json={"query": "hello"})

    assert response.status_code == 200
    assert "secret backend detail" not in response.text
    assert '"message": "Answer generation failed."' in response.text


def test_stream_serializes_datetime_values_in_session_history(api_runtime):
    class DatetimeHistorySystem(FakeSystem):
        def get_session_history(self, session_id):
            return [
                {
                    "id": 1,
                    "session_id": session_id,
                    "question": "Who are you?",
                    "answer": "I am EduRAG.",
                    "created_at": datetime(2026, 7, 15, 19, 27, 40),
                }
            ]

    api_runtime._system = DatetimeHistorySystem()

    with TestClient(app) as test_client:
        response = test_client.post("/api/qa/ask/stream", json={"query": "hello"})

    assert response.status_code == 200
    assert '"type": "done"' in response.text
    assert '"created_at": "2026-07-15T19:27:40"' in response.text


def test_lifespan_closes_mysql_client(api_runtime):
    close_calls = []
    system = FakeSystem()
    system.mysql_client = SimpleNamespace(close=lambda: close_calls.append(True))
    api_runtime._system = system

    with TestClient(app):
        pass

    assert close_calls == [True]


def test_system_factory_uses_injected_config(monkeypatch):
    created = []

    class FakeMySQL:
        def __init__(self, config):
            created.append(("mysql", config))

        def create_conversation_table(self):
            created.append(("conversation_table",))

    class FakeRedis:
        def __init__(self, config):
            created.append(("redis", config))

    class FakeSQL:
        def __init__(self, *, mysql_client, redis_client):
            created.append(("sql", mysql_client, redis_client))
            self.mysql_client = mysql_client
            self.fallback_answer = "fallback"

    class FakeRAG:
        @classmethod
        def from_config(cls, config):
            created.append(("rag", config))
            return cls()

    config = object()
    monkeypatch.setattr(core_system, "MySQLClient", FakeMySQL)
    monkeypatch.setattr(core_system, "RedisClient", FakeRedis, raising=False)
    monkeypatch.setattr(core_system, "MySqlQASystem", FakeSQL)
    monkeypatch.setattr(core_system, "RAGSystem", FakeRAG)

    system = EducationQASystem.from_config(config)

    assert system.mysql_client is not None
    assert created[0] == ("mysql", config)
    assert created[1] == ("redis", config)
    assert created[2][0] == "sql"
    assert created[2][1] is system.mysql_client
    assert created[3] == ("rag", config)
    assert created[4] == ("conversation_table",)
