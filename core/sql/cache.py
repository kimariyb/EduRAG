from __future__ import annotations

import json
from typing import Any

import redis

from base.config import AppConfig, load_config
from base.logger import logger
from core.sql.utils import encode, decode, config_value
from core.sql.constants import (
    DEFAULT_ANSWER_CACHE_KEY_TEMPLATE,
    DEFAULT_QUESTION_CACHE_KEY,
    DEFAULT_TOKEN_CACHE_KEY,
)


log = logger.bind(module=__name__)


class RedisClient:
    def __init__(
        self,
        config: AppConfig | Any | None = None,
        *,
        client: Any | None = None,
        key_prefix: str = "",
    ) -> None:
        self.key_prefix = key_prefix.strip(":")
        self.client = client if client is not None else self._create_client(config)

    def get(self, key: str) -> str | None:
        value = self.client.get(self._key(key))
        decoded = decode(value)
        log.info("Read Redis key: key={}, hit={}", self._key(key), decoded is not None)
        return decoded

    def set(self, key: str, value: Any, ex: int | None = None) -> Any:
        payload = encode(value)
        result = self.client.set(self._key(key), payload, ex=ex)
        log.info("Wrote Redis key: key={}, ttl={}", self._key(key), ex)
        return result

    def delete(self, *keys: str) -> int:
        if not keys:
            return 0
        namespaced = [self._key(key) for key in keys]
        deleted = int(self.client.delete(*namespaced))
        log.info("Deleted Redis keys: count={}", deleted)
        return deleted

    def exists(self, key: str) -> bool:
        exists = bool(self.client.exists(self._key(key)))
        log.info("Checked Redis key existence: key={}, exists={}", self._key(key), exists)
        return exists

    def store_questions(
        self,
        questions: list[dict[str, Any]],
        *,
        key: str = DEFAULT_QUESTION_CACHE_KEY,
        ex: int | None = None,
    ) -> Any:
        log.info("Caching FAQ questions to Redis: count={}", len(questions))
        return self.set(key, json.dumps(questions, ensure_ascii=False), ex=ex)

    def get_questions(self, *, key: str = DEFAULT_QUESTION_CACHE_KEY) -> list[dict[str, Any]] | None:
        questions = self._get_json(key)
        log.info("Loaded FAQ questions from Redis: hit={}", questions is not None)
        return questions

    def store_tokenized_questions(
        self,
        tokenized_questions: list[list[str]],
        *,
        key: str = DEFAULT_TOKEN_CACHE_KEY,
        ex: int | None = None,
    ) -> Any:
        log.info("Caching FAQ question tokens to Redis: count={}", len(tokenized_questions))
        return self.set(key, json.dumps(tokenized_questions, ensure_ascii=False), ex=ex)

    def get_tokenized_questions(self, *, key: str = DEFAULT_TOKEN_CACHE_KEY) -> list[list[str]] | None:
        tokenized_questions = self._get_json(key)
        log.info("Loaded FAQ question tokens from Redis: hit={}", tokenized_questions is not None)
        return tokenized_questions

    def cache_answer(
        self,
        question_id: str | int,
        answer: str,
        *,
        key_template: str = DEFAULT_ANSWER_CACHE_KEY_TEMPLATE,
        ex: int | None = None,
    ) -> Any:
        key = key_template.format(id=question_id)
        log.info("Caching FAQ answer to Redis: question_id={}", question_id)
        return self.set(key, answer, ex=ex)

    def get_answer(
        self,
        question_id: str | int,
        *,
        key_template: str = DEFAULT_ANSWER_CACHE_KEY_TEMPLATE,
    ) -> str | None:
        key = key_template.format(id=question_id)
        answer = self.get(key)
        log.info("Loaded FAQ answer from Redis: question_id={}, hit={}", question_id, answer is not None)
        return answer

    def query_answer(
        self,
        question_id: str | int,
        *,
        key_template: str = DEFAULT_ANSWER_CACHE_KEY_TEMPLATE,
    ) -> str | None:
        return self.get_answer(question_id, key_template=key_template)

    def clear_faq_cache(self) -> int:
        keys = (DEFAULT_QUESTION_CACHE_KEY, DEFAULT_TOKEN_CACHE_KEY)
        deleted = self.delete(*keys)
        log.info("Cleared FAQ Redis cache: deleted={}", deleted)
        return deleted

    def _get_json(self, key: str) -> Any:
        text = self.get(key)
        if not text:
            return None
        try:
            value = json.loads(text)
            log.info("Decoded Redis JSON cache: key={}", key)
            return value
        except json.JSONDecodeError:
            log.warning("Ignoring invalid Redis JSON cache: key={}", key)
            return None

    def _key(self, key: str) -> str:
        if not self.key_prefix:
            return key
        namespaced_key = f"{self.key_prefix}:{key}"
        log.debug("Resolved Redis namespaced key: {}", namespaced_key)
        return namespaced_key

    def _create_client(self, config: AppConfig | Any | None) -> redis.Redis:
        if config is None:
            config = load_config()
        redis_config = config.redis if hasattr(config, "redis") else config

        client = redis.Redis(
            host=config_value(redis_config, "host", "localhost"),
            port=int(config_value(redis_config, "port", 6379)),
            db=int(config_value(redis_config, "db", 0)),
            password=config_value(redis_config, "password", None),
            decode_responses=bool(config_value(redis_config, "decode_responses", True)),
        )
        log.info("Created Redis client from config")
        return client








