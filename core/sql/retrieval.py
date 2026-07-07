from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from rank_bm25 import BM25Okapi

from base.logger import logger
from core.sql.cache import RedisClient
from core.sql.constants import (
    DEFAULT_ANSWER_CACHE_KEY_TEMPLATE,
    DEFAULT_QUESTION_CACHE_KEY,
    DEFAULT_TOKEN_CACHE_KEY,
)
from core.sql.db import MySQLClient
from core.sql.utils import softmax, tokenize


Tokenizer = Callable[[str], list[str]]
DEFAULT_THRESHOLD = 0.85
DEFAULT_SOFTMAX_TEMPERATURE = 0.35

log = logger.bind(module=__name__)


@dataclass(frozen=True)
class FAQRecord:
    question: str
    answer: str | None = None
    id: str | int | None = None
    subject: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievalResult:
    question: str
    score: float
    bm25_score: float
    answer: str | None = None
    id: str | int | None = None
    subject: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class BM25FAQRetriever:
    def __init__(
        self,
        records: Sequence[FAQRecord | Mapping[str, Any]],
        *,
        threshold: float = DEFAULT_THRESHOLD,
        tokenizer: Tokenizer | None = None,
        tokenized_questions: Sequence[Sequence[str]] | None = None,
        softmax_temperature: float = DEFAULT_SOFTMAX_TEMPERATURE,
        redis_client: RedisClient | None = None,
        mysql_client: MySQLClient | None = None,
        answer_cache_key_template: str = DEFAULT_ANSWER_CACHE_KEY_TEMPLATE,
    ) -> None:
        if not records:
            raise ValueError("records cannot be empty")
        if softmax_temperature <= 0:
            raise ValueError("softmax_temperature must be greater than 0")

        self.records = [_coerce_record(record) for record in records]
        self.threshold = threshold
        self.tokenizer = tokenizer or tokenize
        self.softmax_temperature = softmax_temperature
        self.redis_client = redis_client
        self.mysql_client = mysql_client
        self.answer_cache_key_template = answer_cache_key_template

        if tokenized_questions is None:
            self.tokenized_questions = [
                self.tokenizer(record.question) for record in self.records
            ]
        else:
            self.tokenized_questions = [list(tokens) for tokens in tokenized_questions]
            if len(self.tokenized_questions) != len(self.records):
                raise ValueError("tokenized_questions length must match records length")

        self.bm25 = BM25Okapi(self.tokenized_questions)
        log.info(
            "BM25Okapi retriever initialized: records={}, threshold={}, temperature={}",
            len(self.records),
            self.threshold,
            self.softmax_temperature,
        )

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        *,
        question_column: str = "问题",
        answer_column: str = "答案",
        subject_column: str = "学科名称",
        id_column: str = "id",
        encoding: str = "utf-8",
        threshold: float = DEFAULT_THRESHOLD,
        tokenizer: Tokenizer | None = None,
        softmax_temperature: float = DEFAULT_SOFTMAX_TEMPERATURE,
    ) -> "BM25FAQRetriever":
        csv_path = Path(path)
        records: list[FAQRecord] = []

        with csv_path.open("r", encoding=encoding, newline="") as file:
            reader = csv.DictReader(file)
            for row in reader:
                records.append(
                    FAQRecord(
                        id=_row_value(row, id_column, "id", required=False),
                        question=_row_value(row, question_column, "question"),
                        answer=_row_value(row, answer_column, "answer", required=False),
                        subject=_row_value(row, subject_column, "subject", required=False),
                        metadata=dict(row),
                    )
                )

        log.info("Loaded FAQ questions from csv: path={}, count={}", csv_path, len(records))
        return cls(
            records,
            threshold=threshold,
            tokenizer=tokenizer,
            softmax_temperature=softmax_temperature,
        )

    @classmethod
    def from_backend(
        cls,
        redis_client: RedisClient,
        mysql_client: MySQLClient,
        *,
        question_cache_key: str = DEFAULT_QUESTION_CACHE_KEY,
        token_cache_key: str = DEFAULT_TOKEN_CACHE_KEY,
        answer_cache_key_template: str = DEFAULT_ANSWER_CACHE_KEY_TEMPLATE,
        threshold: float = DEFAULT_THRESHOLD,
        tokenizer: Tokenizer | None = None,
        softmax_temperature: float = DEFAULT_SOFTMAX_TEMPERATURE,
    ) -> "BM25FAQRetriever":
        active_tokenizer = tokenizer or tokenize
        cached_records = _cache_get_json(redis_client, question_cache_key)
        cached_tokens = _cache_get_json(redis_client, token_cache_key)
        log.info("Checked Redis FAQ question and token cache")

        if cached_records and cached_tokens:
            records = [_coerce_record(record) for record in cached_records]
            tokenized_questions = [list(tokens) for tokens in cached_tokens]
            if len(records) == len(tokenized_questions):
                log.info("Loaded FAQ questions and tokens from Redis cache")
                return cls(
                    records,
                    threshold=threshold,
                    tokenizer=active_tokenizer,
                    tokenized_questions=tokenized_questions,
                    softmax_temperature=softmax_temperature,
                    redis_client=redis_client,
                    mysql_client=mysql_client,
                    answer_cache_key_template=answer_cache_key_template,
                )

            log.warning(
                "Redis FAQ cache ignored because records/tokens length mismatch: records={}, tokens={}",
                len(records),
                len(tokenized_questions),
            )

        log.info("FAQ question cache miss; loading questions from MySQL")
        rows = mysql_client.fetch_faq_questions()
        log.info("Loaded FAQ question rows from MySQL: count={}", len(rows))
        records = [_coerce_record(row) for row in rows]
        tokenized_questions = [active_tokenizer(record.question) for record in records]
        log.info("Tokenized FAQ questions after MySQL load: count={}", len(tokenized_questions))

        _cache_set_json(redis_client, question_cache_key, [_record_to_cache(record) for record in records])
        _cache_set_json(redis_client, token_cache_key, tokenized_questions)
        log.info("Cached FAQ questions and tokens to Redis: count={}", len(records))

        return cls(
            records,
            threshold=threshold,
            tokenizer=active_tokenizer,
            tokenized_questions=tokenized_questions,
            softmax_temperature=softmax_temperature,
            redis_client=redis_client,
            mysql_client=mysql_client,
            answer_cache_key_template=answer_cache_key_template,
        )

    def search(self, query: str, *, top_k: int = 3) -> list[RetrievalResult]:
        if top_k <= 0:
            return []

        query_tokens = self.tokenizer(query)
        if not query_tokens:
            log.warning("Skip BM25 search because query produced no tokens")
            return []
        log.info("Tokenized query for BM25 search: tokens={}", len(query_tokens))

        bm25_scores = [float(score) for score in self.bm25.get_scores(query_tokens)]
        log.info("Calculated BM25Okapi scores: count={}", len(bm25_scores))
        normalized_scores = softmax(
            bm25_scores,
            temperature=self.softmax_temperature,
        )
        log.info("Normalized BM25 scores with Softmax: count={}", len(normalized_scores))
        results = [
            RetrievalResult(
                id=record.id,
                question=record.question,
                answer=record.answer,
                subject=record.subject,
                metadata=record.metadata,
                score=normalized_scores[index],
                bm25_score=bm25_scores[index],
            )
            for index, record in enumerate(self.records)
        ]

        results.sort(key=lambda result: (result.score, result.bm25_score), reverse=True)
        top_results = results[:top_k]
        if top_results:
            log.info(
                "BM25 search completed: query='{}', top_question_id={}, top_score={:.6f}",
                query,
                top_results[0].id,
                top_results[0].score,
            )
        return top_results

    def match(self, query: str, *, threshold: float | None = None) -> RetrievalResult | None:
        results = self.search(query, top_k=1)
        if not results:
            return None

        min_score = self.threshold if threshold is None else threshold
        best = results[0]
        if best.score <= min_score:
            log.info(
                "BM25 match is not reliable: question_id={}, score={:.6f}, threshold={}",
                best.id,
                best.score,
                min_score,
            )
            return None
        log.info(
            "BM25 reliable match found: question_id={}, score={:.6f}, threshold={}",
            best.id,
            best.score,
            min_score,
        )
        return best

    def answer(self, query: str, *, threshold: float | None = None) -> str | None:
        result = self.match(query, threshold=threshold)
        if result is None:
            return None

        answer = self._load_answer(result)
        if answer is None:
            log.warning("No answer found for matched question: question_id={}", result.id)
            return None
        return answer

    def _load_answer(self, result: RetrievalResult) -> str | None:
        if result.id is None:
            answer = _clean_answer(result.answer)
            if answer:
                log.info("Using in-memory FAQ answer without cache because question_id is missing")
                return answer
            return None

        can_cache_answer = result.score > DEFAULT_THRESHOLD
        cache_key = self.answer_cache_key_template.format(id=result.id)
        cached_answer = _cache_get_text(self.redis_client, cache_key)
        if cached_answer:
            log.info("Loaded FAQ answer from Redis cache: question_id={}", result.id)
            return cached_answer

        if self.mysql_client is not None:
            answer = _clean_answer(self.mysql_client.fetch_faq_answer(result.id))
            if answer:
                if can_cache_answer:
                    _cache_set_text(self.redis_client, cache_key, answer)
                    log.info("Loaded FAQ answer from MySQL and cached to Redis: question_id={}", result.id)
                else:
                    log.info(
                        "Loaded FAQ answer from MySQL without Redis cache: question_id={}, score={:.6f}",
                        result.id,
                        result.score,
                    )
                return answer
            log.warning("MySQL returned no reliable answer: question_id={}", result.id)

        answer = _clean_answer(result.answer)
        if answer:
            if can_cache_answer:
                _cache_set_text(self.redis_client, cache_key, answer)
                log.info("Cached in-memory FAQ answer to Redis: question_id={}", result.id)
            else:
                log.info(
                    "Used in-memory FAQ answer without Redis cache: question_id={}, score={:.6f}",
                    result.id,
                    result.score,
                )
            return answer

        return None


def _coerce_record(record: FAQRecord | Mapping[str, Any]) -> FAQRecord:
    if isinstance(record, FAQRecord):
        return record

    return FAQRecord(
        id=_first_value(record, "id", "question_id", "qa_id", "编号", required=False),
        question=_first_value(record, "问题", "question"),
        answer=_first_value(record, "答案", "answer", required=False),
        subject=_first_value(record, "学科名称", "subject", required=False),
        metadata=dict(record),
    )


def _row_value(
    row: Mapping[str, Any],
    primary_key: str,
    fallback_key: str,
    *,
    required: bool = True,
) -> str | None:
    return _first_value(row, primary_key, fallback_key, required=required)


def _first_value(
    row: Mapping[str, Any],
    *keys: str,
    required: bool = True,
) -> Any:
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip():
            return value

    if not required:
        return None

    key_names = ", ".join(keys)
    raise ValueError(f"missing required column: {key_names}")


def _record_to_cache(record: FAQRecord) -> dict[str, Any]:
    return {
        "id": record.id,
        "question": record.question,
        "subject": record.subject,
    }


def _clean_answer(answer: Any) -> str | None:
    if answer is None:
        return None
    text = str(answer).strip()
    if not text:
        return None
    return text


def _cache_get_json(redis_client: RedisClient | None, key: str) -> Any:
    text = _cache_get_text(redis_client, key)
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        log.warning("Ignoring invalid JSON from Redis cache: key={}", key)
        return None


def _cache_set_json(redis_client: RedisClient | None, key: str, value: Any) -> None:
    _cache_set_text(redis_client, key, json.dumps(value, ensure_ascii=False))


def _cache_get_text(redis_client: RedisClient | None, key: str) -> str | None:
    if redis_client is None:
        return None
    value = redis_client.get(key)
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, str):
        return value
    return str(value)


def _cache_set_text(redis_client: RedisClient | None, key: str, value: str) -> None:
    if redis_client is None:
        return
    redis_client.set(key, value)
