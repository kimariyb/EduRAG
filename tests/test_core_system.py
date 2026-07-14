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


class FakeNonStreamingRAG:
    def __init__(self):
        self.calls = []

    def generate_answer(self, query, source_filter=None):
        self.calls.append((query, source_filter))
        return "non-streaming rag answer"


class RaisingStreamRAG:
    def generate_answer_stream(self, query, source_filter=None):
        yield "partial "
        raise RuntimeError("stream failed")


def build(sql_answer="sql answer"):
    database = FakeDatabase()
    system = EducationQASystem(sql_system=FakeSQL(sql_answer), rag_system=FakeRAG(), mysql_client=database)
    return system, database


def test_sql_hit_has_priority_and_is_persisted():
    system, database = build()
    response = system.query("faq", session_id="s1")
    assert (response.session_id, response.source, response.answer) == ("s1", "sql", "sql answer")
    assert database.turns == [{"session_id": "s1", "question": "faq", "answer": "sql answer"}]


def test_generated_session_id_is_used_to_persist_the_turn():
    system, database = build()

    response = system.query("faq")

    assert response.session_id
    assert database.turns == [
        {"session_id": response.session_id, "question": "faq", "answer": "sql answer"}
    ]


def test_constructs_default_sql_system_with_supplied_mysql_client(monkeypatch):
    database = FakeDatabase()
    constructed_with = []

    class CapturingSQL(FakeSQL):
        def __init__(self, *, mysql_client):
            constructed_with.append(mysql_client)
            self.mysql_client = mysql_client
            super().__init__("sql answer")

    monkeypatch.setattr("core.system.MySqlQASystem", CapturingSQL)

    system = EducationQASystem(mysql_client=database, rag_system=FakeRAG())

    assert constructed_with == [database]
    assert system.sql_system.mysql_client is database
    assert system.mysql_client is database


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


def test_sql_stream_yields_one_answer_and_persists_after_consumption():
    system, database = build()

    session_id, source, chunks = system.stream_query("faq", session_id="s1")

    assert (session_id, source, database.turns) == ("s1", "sql", [])
    assert list(chunks) == ["sql answer"]
    assert database.turns == [
        {"session_id": "s1", "question": "faq", "answer": "sql answer"}
    ]


def test_abandoned_rag_stream_does_not_persist_a_partial_turn():
    system, database = build("在 sql 中没有找到答案")
    _, _, chunks = system.stream_query("question", session_id="s1")

    assert next(chunks) == "rag "
    chunks.close()

    assert database.turns == []


def test_raising_rag_stream_does_not_persist_a_partial_turn():
    database = FakeDatabase()
    system = EducationQASystem(
        sql_system=FakeSQL("在 sql 中没有找到答案"),
        rag_system=RaisingStreamRAG(),
        mysql_client=database,
    )
    _, _, chunks = system.stream_query("question", session_id="s1")

    with pytest.raises(RuntimeError, match="stream failed"):
        list(chunks)

    assert database.turns == []


def test_streaming_falls_back_to_generate_answer_when_rag_has_no_stream_method():
    database = FakeDatabase()
    rag_system = FakeNonStreamingRAG()
    system = EducationQASystem(
        sql_system=FakeSQL("在 sql 中没有找到答案"),
        rag_system=rag_system,
        mysql_client=database,
    )

    session_id, source, chunks = system.stream_query(
        "question", source_filter="python", session_id="s1"
    )

    assert (session_id, source, database.turns) == ("s1", "rag", [])
    assert list(chunks) == ["non-streaming rag answer"]
    assert rag_system.calls == [("question", "python")]
    assert database.turns == [
        {
            "session_id": "s1",
            "question": "question",
            "answer": "non-streaming rag answer",
        }
    ]


def test_history_is_limited_and_clearable_and_blank_query_is_rejected():
    system, _ = build()
    for index in range(6):
        system.query(f"q{index}", session_id="s1")
    assert [row["question"] for row in system.get_session_history("s1")] == ["q1", "q2", "q3", "q4", "q5"]
    assert system.clear_session_history("s1") is True
    assert system.get_session_history("s1") == []
    with pytest.raises(ValueError, match="query must not be empty"):
        system.query("  ")
