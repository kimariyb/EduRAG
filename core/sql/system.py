from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

from base.logger import logger
from core.sql.cache import RedisClient
from core.sql.db import DEFAULT_SEED_CSV_PATH, MySQLClient
from core.sql.retrieval import DEFAULT_THRESHOLD, BM25FAQRetriever


log = logger.bind(module=__name__)


class MySqlQASystem:
    def __init__(
        self,
        *,
        mysql_client: MySQLClient | Any | None = None,
        redis_client: RedisClient | Any | None = None,
        retriever: BM25FAQRetriever | Any | None = None,
        threshold: float = DEFAULT_THRESHOLD,
        fallback_answer: str = "在 sql 中没有找到答案",
        seed_csv_path: str | Path | None = DEFAULT_SEED_CSV_PATH,
        auto_import: bool = True,
    ) -> None:
        self.threshold = threshold
        self.fallback_answer = fallback_answer
        self.mysql_client = mysql_client
        self.redis_client = redis_client
        self.seed_csv_path = Path(seed_csv_path) if seed_csv_path is not None else None
        self.auto_import = auto_import

        if retriever is not None:
            self.retriever = retriever
            log.info("MysqlQASystem initialized with injected retriever")
            return

        if self.mysql_client is None:
            self.mysql_client = MySQLClient(
                seed_csv_path=self.seed_csv_path,
                auto_import=self.auto_import,
            )
        log.info("MysqlQASystem MySQL client initialized")

        self.redis_client = self.redis_client or RedisClient()
        log.info("MysqlQASystem Redis client initialized")
        self.retriever = BM25FAQRetriever.from_backend(
            self.redis_client,
            self.mysql_client,
            threshold=self.threshold,
        )
        log.info("MysqlQASystem BM25 retriever initialized")

    def query(self, query: str) -> str:
        start = perf_counter()
        log.info("SQL QA query started: query='{}'", query)
        try:
            answer = self.retriever.answer(query, threshold=self.threshold)
            if answer:
                log.info("SQL QA query answered from BM25/MySQL path")
                return answer

            log.warning("SQL QA query missed: query='{}'", query)
            log.info("SQL QA query requires fallback outside SQL layer")
            return self.fallback_answer
        finally:
            duration_ms = (perf_counter() - start) * 1000
            log.info("SQL QA query finished: duration_ms={:.3f}", duration_ms)


if __name__ == "__main__":
    # 用于测试 MySqlQASystem
    from base.config import load_config

    config = load_config()
    mysql_client = MySQLClient(config)
    redis_client = RedisClient(config)

    retriever = BM25FAQRetriever.from_backend(redis_client, mysql_client)

    sys = MySqlQASystem(
        mysql_client=mysql_client,
        redis_client=redis_client,
        retriever=retriever)

    answer = sys.query("linux 查看指定进程的资源占用情况")
    print(answer)
