import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

from core.sql.retrieval import (
    BM25FAQRetriever,
    FAQRecord,
    softmax,
)


class FakeRedis:
    def __init__(self, initial=None):
        self.store = dict(initial or {})
        self.set_calls = []

    def get(self, key):
        return self.store.get(key)

    def set(self, key, value):
        self.store[key] = value
        self.set_calls.append((key, value))


class FakeMySQL:
    def __init__(self):
        self.question_calls = 0
        self.answer_calls = []
        self.questions = [
            {
                "id": 1,
                "question": "Python 如何创建虚拟环境",
                "subject": "Python学科",
            },
            {
                "id": 2,
                "question": "Redis 缓存如何设置过期时间",
                "subject": "Redis学科",
            },
            {
                "id": 3,
                "question": "MySQL 如何创建索引",
                "subject": "MySQL学科",
            },
        ]
        self.answers = {
            1: "使用 python -m venv .venv",
            2: "使用 expire 命令设置过期时间",
            3: "使用 CREATE INDEX 创建索引",
        }

    def fetch_faq_questions(self):
        self.question_calls += 1
        return list(self.questions)

    def fetch_faq_answer(self, question_id):
        self.answer_calls.append(question_id)
        return self.answers[question_id]


def _sample_records() -> list[FAQRecord]:
    return [
        FAQRecord(
            id=1,
            question="Python 如何创建虚拟环境",
            answer="使用 python -m venv .venv",
            subject="Python学科",
        ),
        FAQRecord(
            id=2,
            question="Redis 缓存如何设置过期时间",
            answer="使用 expire 命令设置过期时间",
            subject="Redis学科",
        ),
        FAQRecord(
            id=3,
            question="MySQL 如何创建索引",
            answer="使用 CREATE INDEX 创建索引",
            subject="MySQL学科",
        ),
    ]


def test_search_uses_bm25_okapi_and_softmax_scores():
    retriever = BM25FAQRetriever(_sample_records(), threshold=0.85)

    results = retriever.search("Redis 缓存如何设置过期时间", top_k=3)

    assert results[0].answer == "使用 expire 命令设置过期时间"
    assert results[0].score == pytest.approx(max(result.score for result in results))
    assert sum(result.score for result in results) == pytest.approx(1.0)
    assert results[0].score >= 0.85


def test_match_returns_none_when_top_softmax_score_is_below_threshold():
    retriever = BM25FAQRetriever(_sample_records(), threshold=0.85)

    result = retriever.match("Milvus 向量数据库怎么部署")

    assert result is None


def test_match_requires_top_softmax_score_to_be_greater_than_threshold():
    retriever = BM25FAQRetriever(_sample_records(), threshold=0.85)
    top_score = retriever.search("Redis 缓存如何设置过期时间", top_k=1)[0].score

    result = retriever.match("Redis 缓存如何设置过期时间", threshold=top_score)

    assert result is None


def test_softmax_normalizes_scores():
    probabilities = softmax([1.0, 2.0, 3.0])

    assert sum(probabilities) == pytest.approx(1.0)
    assert probabilities[2] > probabilities[1] > probabilities[0]


def test_from_csv_loads_default_chinese_faq_columns(tmp_path: Path):
    csv_path = tmp_path / "faq.csv"
    csv_path.write_text(
        "id,学科名称,问题,答案\n"
        "2,Redis学科,Redis 缓存如何设置过期时间,使用 expire 命令设置过期时间\n"
        "3,MySQL学科,MySQL 如何创建索引,使用 CREATE INDEX 创建索引\n"
        "4,Python学科,Python 如何创建虚拟环境,使用 python -m venv .venv\n",
        encoding="utf-8",
    )

    retriever = BM25FAQRetriever.from_csv(csv_path)
    result = retriever.match("Redis 缓存如何设置过期时间")

    assert result is not None
    assert result.id == "2"
    assert result.subject == "Redis学科"
    assert result.answer == "使用 expire 命令设置过期时间"


def test_from_backend_uses_cached_questions_and_tokens_before_mysql():
    cached_records = [
        {"id": 2, "question": "Redis 缓存如何设置过期时间", "subject": "Redis学科"}
    ]
    cached_tokens = [["redis", "缓存", "如何", "设置", "过期", "时间"]]
    redis_client = FakeRedis(
        {
            "faq:questions": json.dumps(cached_records, ensure_ascii=False),
            "faq:tokenized_questions": json.dumps(cached_tokens, ensure_ascii=False),
            "faq:answer:2": "缓存答案",
        }
    )
    mysql_client = FakeMySQL()

    retriever = BM25FAQRetriever.from_backend(redis_client, mysql_client)
    answer = retriever.answer("Redis 缓存如何设置过期时间")

    assert answer == "缓存答案"
    assert mysql_client.question_calls == 0
    assert mysql_client.answer_calls == []


def test_from_backend_loads_questions_from_mysql_and_caches_tokens_when_redis_misses():
    redis_client = FakeRedis()
    mysql_client = FakeMySQL()

    retriever = BM25FAQRetriever.from_backend(redis_client, mysql_client)

    assert mysql_client.question_calls == 1
    assert retriever.records[0].question == "Python 如何创建虚拟环境"
    assert "faq:questions" in redis_client.store
    assert "faq:tokenized_questions" in redis_client.store


def test_answer_fetches_from_mysql_and_caches_when_answer_cache_misses():
    redis_client = FakeRedis()
    mysql_client = FakeMySQL()
    retriever = BM25FAQRetriever.from_backend(redis_client, mysql_client)

    answer = retriever.answer("Python 如何创建虚拟环境")

    assert answer == "使用 python -m venv .venv"
    assert mysql_client.answer_calls == [1]
    assert redis_client.store["faq:answer:1"] == "使用 python -m venv .venv"


def test_answer_does_not_query_mysql_or_cache_when_match_is_not_reliable():
    redis_client = FakeRedis()
    mysql_client = FakeMySQL()
    retriever = BM25FAQRetriever.from_backend(redis_client, mysql_client)
    top_score = retriever.search("Python 如何创建虚拟环境", top_k=1)[0].score

    answer = retriever.answer("Python 如何创建虚拟环境", threshold=top_score)

    assert answer is None
    assert mysql_client.answer_calls == []
    assert "faq:answer:1" not in redis_client.store


def test_answer_does_not_cache_blank_mysql_answers():
    redis_client = FakeRedis()
    mysql_client = FakeMySQL()
    mysql_client.answers[1] = "   "
    retriever = BM25FAQRetriever.from_backend(redis_client, mysql_client)

    answer = retriever.answer("Python 如何创建虚拟环境")

    assert answer is None
    assert mysql_client.answer_calls == [1]
    assert "faq:answer:1" not in redis_client.store


def test_answer_does_not_cache_when_custom_threshold_allows_low_similarity_match():
    redis_client = FakeRedis()
    mysql_client = FakeMySQL()
    retriever = BM25FAQRetriever.from_backend(redis_client, mysql_client)

    answer = retriever.answer("缓存", threshold=0.5)

    assert answer == "使用 expire 命令设置过期时间"
    assert mysql_client.answer_calls == [2]
    assert "faq:answer:2" not in redis_client.store


def test_tokenizer_does_not_write_jieba_startup_logs():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "from core.sql.retrieval import tokenize; tokenize('Python 虚拟环境')",
        ],
        capture_output=True,
        check=True,
        text=True,
    )

    assert "Building prefix dict" not in completed.stderr
    assert "Prefix dict has been built successfully" not in completed.stderr


def test_retrieval_uses_project_logger_only_and_okapi():
    source = Path("core/sql/retrieval.py").read_text(encoding="utf-8")
    module = ast.parse(source)

    imports_base_logger = any(
        isinstance(node, ast.ImportFrom)
        and node.module == "base.logger"
        and any(alias.name == "logger" for alias in node.names)
        for node in module.body
    )
    imports_logging_module = any(
        isinstance(node, ast.Import) and any(alias.name == "logging" for alias in node.names)
        for node in module.body
    )
    imports_loguru_directly = any(
        isinstance(node, ast.ImportFrom) and node.module == "loguru"
        for node in module.body
    )
    imports_bm25_okapi = any(
        isinstance(node, ast.ImportFrom)
        and node.module == "rank_bm25"
        and any(alias.name == "BM25Okapi" for alias in node.names)
        for node in module.body
    )
    imports_bm25_plus = any(
        isinstance(node, ast.ImportFrom)
        and node.module == "rank_bm25"
        and any(alias.name == "BM25Plus" for alias in node.names)
        for node in module.body
    )
    class_names = {
        node.name
        for node in module.body
        if isinstance(node, ast.ClassDef)
    }
    imports_redis_client = any(
        isinstance(node, ast.ImportFrom)
        and node.module == "core.sql.cache"
        and any(alias.name == "RedisClient" for alias in node.names)
        for node in module.body
    )
    imports_mysql_client = any(
        isinstance(node, ast.ImportFrom)
        and node.module == "core.sql.db"
        and any(alias.name == "MySQLClient" for alias in node.names)
        for node in module.body
    )

    assert imports_base_logger
    assert not imports_logging_module
    assert not imports_loguru_directly
    assert imports_bm25_okapi
    assert not imports_bm25_plus
    assert "BM25OkapiSearch" not in class_names
    assert "RedisLike" not in class_names
    assert "MySQLLike" not in class_names
    assert imports_redis_client
    assert imports_mysql_client
