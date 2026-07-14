from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping, Sequence

import pymysql
from pymysql.cursors import DictCursor

from base.config import AppConfig, load_config
from base.logger import logger
from core.sql.utils import validate_identifier, config_value


log = logger.bind(module=__name__)
DEFAULT_SEED_CSV_PATH = Path(__file__).resolve().parents[2] / "data" / "subject_knowledge_qa.csv"


class MySQLClient:
    def __init__(
        self,
        config: AppConfig | Any | None = None,
        *,
        connection: Any | None = None,
        table_name: str = "faq",
        seed_csv_path: str | Path | None = DEFAULT_SEED_CSV_PATH,
        auto_import: bool = True,
    ) -> None:
        self.table_name = validate_identifier(table_name)
        self.seed_csv_path = Path(seed_csv_path) if seed_csv_path is not None else None
        self.auto_import = auto_import
        self.connection = connection if connection is not None else self._create_connection(config)
        if self.auto_import:
            self.prepare_faq_data()

    def prepare_faq_data(self) -> None:
        log.info("Preparing MySQL FAQ data")
        self.create_table()
        log.info("MySQL FAQ table prepared")

        existing_rows = self.list_faqs(limit=1)
        log.info("Checked MySQL FAQ seed state: has_data={}", bool(existing_rows))
        if existing_rows:
            log.info("Skipped FAQ CSV import because MySQL already has data")
            return

        if self.seed_csv_path is None:
            log.warning("Skipped FAQ CSV import because seed_csv_path is not configured")
            return
        if not self.seed_csv_path.exists():
            raise FileNotFoundError(f"FAQ seed csv not found: {self.seed_csv_path}")

        imported_count = self.import_faq_csv(self.seed_csv_path)
        log.info("Imported FAQ CSV into MySQL: path={}, count={}", self.seed_csv_path, imported_count)

    def import_faq_csv(self, csv_path: str | Path) -> int:
        imported_count = 0
        path = Path(csv_path)
        with path.open("r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            for row_number, row in enumerate(reader, start=2):
                question = _csv_value(row, "问题", "question")
                answer = _csv_value(row, "答案", "answer")
                subject = _csv_value(row, "学科名称", "subject", required=False)
                self.insert_faq(question=question, answer=answer, subject=subject)
                imported_count += 1
                log.debug("Imported FAQ CSV row into MySQL: row_number={}", row_number)
        return imported_count

    def create_table(self) -> None:
        sql = f"""
        CREATE TABLE IF NOT EXISTS `{self.table_name}` (
            id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
            subject VARCHAR(255) NULL,
            question TEXT NOT NULL,
            answer LONGTEXT NOT NULL,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            INDEX idx_subject (subject)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """
        self._execute(sql, commit=True)
        log.info("Ensured MySQL FAQ table exists: table={}", self.table_name)

    create_faq_table = create_table

    def insert_faq(self, *, question: str, answer: str, subject: str | None = None) -> int:
        sql = f"""
        INSERT INTO `{self.table_name}` (subject, question, answer)
        VALUES (%s, %s, %s)
        """
        cursor = self._execute(sql, (subject, question, answer), commit=True)
        log.info("Inserted FAQ into MySQL: question_id={}", cursor.lastrowid)
        return int(cursor.lastrowid)

    add_faq = insert_faq

    def get_faq(self, question_id: int | str) -> dict[str, Any] | None:
        sql = f"""
        SELECT id, subject, question, answer
        FROM `{self.table_name}`
        WHERE id = %s
        """
        row = self._fetch_one(sql, (question_id,))
        result = dict(row) if row else None
        log.info("Loaded FAQ from MySQL: question_id={}, hit={}", question_id, result is not None)
        return result

    def list_faqs(self, *, limit: int | None = None, offset: int = 0) -> list[dict[str, Any]]:
        sql = f"SELECT id, subject, question, answer FROM `{self.table_name}` ORDER BY id"
        params: tuple[Any, ...] | None = None
        if limit is not None:
            sql += " LIMIT %s OFFSET %s"
            params = (limit, offset)

        rows = [dict(row) for row in self._fetch_all(sql, params)]
        log.info("Listed FAQs from MySQL: count={}", len(rows))
        return rows

    def update_faq(
        self,
        question_id: int | str,
        *,
        question: str | None = None,
        answer: str | None = None,
        subject: str | None = None,
    ) -> None:
        updates: list[str] = []
        params: list[Any] = []
        for column, value in (
            ("subject", subject),
            ("question", question),
            ("answer", answer),
        ):
            if value is not None:
                updates.append(f"{column} = %s")
                params.append(value)

        if not updates:
            raise ValueError("at least one field must be provided for update")

        params.append(question_id)
        sql = f"UPDATE `{self.table_name}` SET {', '.join(updates)} WHERE id = %s"
        self._execute(sql, tuple(params), commit=True)
        log.info("Updated FAQ in MySQL: question_id={}", question_id)

    def delete_faq(self, question_id: int | str) -> None:
        sql = f"DELETE FROM `{self.table_name}` WHERE id = %s"
        self._execute(sql, (question_id,), commit=True)
        log.info("Deleted FAQ from MySQL: question_id={}", question_id)

    def fetch_faq_questions(self) -> list[dict[str, Any]]:
        sql = f"SELECT id, subject, question FROM `{self.table_name}` ORDER BY id"
        rows = [dict(row) for row in self._fetch_all(sql)]
        log.info("Loaded FAQ questions from MySQL: count={}", len(rows))
        return rows

    def fetch_faq_answer(self, question_id: int | str) -> str | None:
        sql = f"SELECT answer FROM `{self.table_name}` WHERE id = %s"
        row = self._fetch_one(sql, (question_id,))
        if not row:
            return None
        if isinstance(row, Mapping):
            answer = row.get("answer")
        else:
            answer = row[0]
        log.info("Loaded FAQ answer from MySQL: question_id={}, hit={}", question_id, answer is not None)
        return answer

    get_answer = fetch_faq_answer

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

    def append_conversation_turn(
        self,
        session_id: str,
        question: str,
        answer: str,
        table_name: str = "conversations",
    ) -> int:
        table = validate_identifier(table_name)
        cursor = self._execute(
            f"INSERT INTO `{table}` (session_id, question, answer) VALUES (%s, %s, %s)",
            (session_id, question, answer),
        )
        turn_id = int(cursor.lastrowid)
        self._execute(
            f"DELETE FROM `{table}` WHERE session_id = %s AND id NOT IN ("
            f"SELECT id FROM (SELECT id FROM `{table}` WHERE session_id = %s "
            "ORDER BY id DESC LIMIT %s) AS recent_turns)",
            (session_id, session_id, 5),
            commit=True,
        )
        return turn_id

    def fetch_recent_conversations(
        self,
        session_id: str,
        limit: int = 5,
        table_name: str = "conversations",
    ) -> list[dict[str, Any]]:
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

    def close(self) -> None:
        self.connection.close()
        log.info("Closed MySQL connection")

    def _execute(
        self,
        sql: str,
        params: Sequence[Any] | None = None,
        *,
        commit: bool = False,
    ) -> Any:
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(sql, params)
                if commit:
                    self.connection.commit()
                    log.info("Committed MySQL operation")
                log.debug("Executed MySQL statement")
                return cursor
        except Exception:
            if commit:
                self.connection.rollback()
            log.exception("MySQL operation failed")
            raise

    def _fetch_all(
        self,
        sql: str,
        params: Sequence[Any] | None = None,
    ) -> list[Any]:
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(sql, params)
                rows = list(cursor.fetchall())
                log.debug("Fetched MySQL rows: count={}", len(rows))
                return rows
        except Exception:
            log.exception("MySQL query failed")
            raise

    def _fetch_one(
        self,
        sql: str,
        params: Sequence[Any] | None = None,
    ) -> Any | None:
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(sql, params)
                row = cursor.fetchone()
                log.debug("Fetched one MySQL row: hit={}", row is not None)
                return row
        except Exception:
            log.exception("MySQL query failed")
            raise

    def _create_connection(self, config: AppConfig | Any | None) -> Any:
        if config is None:
            config = load_config()
        mysql_config = config.mysql if hasattr(config, "mysql") else config

        connection = pymysql.connect(
            host=config_value(mysql_config, "host", "localhost"),
            port=int(config_value(mysql_config, "port", 3306)),
            user=config_value(mysql_config, "username", config_value(mysql_config, "user", "root")),
            password=config_value(mysql_config, "password", ""),
            database=config_value(mysql_config, "database", "edurag"),
            charset=config_value(mysql_config, "charset", "utf8mb4"),
            cursorclass=DictCursor,
            autocommit=False,
        )
        log.info("Created MySQL connection from config")
        return connection


def _csv_value(
    row: Mapping[str, Any],
    primary_key: str,
    fallback_key: str,
    *,
    required: bool = True,
) -> str | None:
    for key in (primary_key, fallback_key):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    if not required:
        return None
    raise ValueError(f"missing required csv column: {primary_key}, {fallback_key}")
