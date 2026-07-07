from pathlib import Path

from core.sql.db import MySQLClient


class FakeCursor:
    def __init__(self, rows=None, row=None, lastrowid=99):
        self.rows = rows or []
        self.row = row
        self.lastrowid = lastrowid
        self.executed = []

    def execute(self, sql, params=None):
        self.executed.append((sql, params))
        return 1

    def fetchall(self):
        return self.rows

    def fetchone(self):
        return self.row

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class FakeConnection:
    def __init__(self, cursor):
        self.cursor_obj = cursor
        self.commit_count = 0
        self.rollback_count = 0
        self.close_count = 0

    def cursor(self):
        return self.cursor_obj

    def commit(self):
        self.commit_count += 1

    def rollback(self):
        self.rollback_count += 1

    def close(self):
        self.close_count += 1


class ClosingCursor(FakeCursor):
    def __init__(self, rows=None):
        super().__init__(rows=rows)
        self.closed = False

    def fetchall(self):
        if self.closed:
            raise RuntimeError("cursor already closed")
        return super().fetchall()

    def __exit__(self, exc_type, exc, tb):
        self.closed = True
        return False


def test_mysql_client_create_table_executes_schema_sql():
    cursor = FakeCursor()
    connection = FakeConnection(cursor)
    client = MySQLClient(connection=connection, table_name="faq", auto_import=False)

    client.create_table()

    sql, params = cursor.executed[0]
    assert "CREATE TABLE IF NOT EXISTS `faq`" in sql
    assert params is None
    assert connection.commit_count == 1


def test_mysql_client_insert_faq_commits_and_returns_id():
    cursor = FakeCursor(lastrowid=7)
    connection = FakeConnection(cursor)
    client = MySQLClient(connection=connection, auto_import=False)

    question_id = client.insert_faq(
        question="Redis 缓存如何设置过期时间",
        answer="使用 expire 命令",
        subject="Redis学科",
    )

    sql, params = cursor.executed[0]
    assert "INSERT INTO `faq`" in sql
    assert params == ("Redis学科", "Redis 缓存如何设置过期时间", "使用 expire 命令")
    assert question_id == 7
    assert connection.commit_count == 1


def test_mysql_client_fetch_faq_questions_returns_rows_without_answers():
    rows = [{"id": 1, "question": "Python 如何创建虚拟环境", "subject": "Python学科"}]
    cursor = FakeCursor(rows=rows)
    connection = FakeConnection(cursor)
    client = MySQLClient(connection=connection, auto_import=False)

    result = client.fetch_faq_questions()

    sql, params = cursor.executed[0]
    assert "SELECT id, subject, question" in sql
    assert "answer" not in sql.lower()
    assert params is None
    assert result == rows


def test_mysql_client_fetch_faq_answer_returns_answer_text():
    cursor = FakeCursor(row={"answer": "使用 CREATE INDEX 创建索引"})
    connection = FakeConnection(cursor)
    client = MySQLClient(connection=connection, auto_import=False)

    answer = client.fetch_faq_answer(3)

    sql, params = cursor.executed[0]
    assert "SELECT answer" in sql
    assert params == (3,)
    assert answer == "使用 CREATE INDEX 创建索引"


def test_mysql_client_update_delete_and_get_faq():
    cursor = FakeCursor(row={"id": 2, "subject": "Redis学科", "question": "Redis Q", "answer": "Redis A"})
    connection = FakeConnection(cursor)
    client = MySQLClient(connection=connection, auto_import=False)

    client.update_faq(2, question="Redis 新问题", answer="Redis 新答案", subject="Redis")
    client.delete_faq(2)
    row = client.get_faq(2)

    executed_sql = [sql for sql, _ in cursor.executed]
    assert "UPDATE `faq`" in executed_sql[0]
    assert "DELETE FROM `faq`" in executed_sql[1]
    assert "SELECT id, subject, question, answer" in executed_sql[2]
    assert row == {"id": 2, "subject": "Redis学科", "question": "Redis Q", "answer": "Redis A"}
    assert connection.commit_count == 2


def test_mysql_client_fetches_rows_before_cursor_context_closes():
    cursor = ClosingCursor(rows=[{"id": 1, "question": "Python", "subject": "Python学科"}])
    connection = FakeConnection(cursor)
    client = MySQLClient(connection=connection, auto_import=False)

    rows = client.fetch_faq_questions()

    assert rows == [{"id": 1, "question": "Python", "subject": "Python学科"}]
    assert cursor.closed


class SequenceConnection:
    def __init__(self, cursors):
        self.cursors = list(cursors)
        self.commit_count = 0
        self.rollback_count = 0

    def cursor(self):
        return self.cursors.pop(0)

    def commit(self):
        self.commit_count += 1

    def rollback(self):
        self.rollback_count += 1


def test_mysql_client_auto_creates_table_and_imports_seed_csv_when_empty(tmp_path: Path):
    seed_csv = tmp_path / "faq.csv"
    seed_csv.write_text(
        "学科名称,问题,答案\n"
        "Python学科,Python 如何创建虚拟环境,使用 python -m venv .venv\n"
        "Redis学科,Redis 缓存如何设置过期时间,使用 expire 命令设置过期时间\n",
        encoding="utf-8",
    )
    create_cursor = FakeCursor()
    empty_check_cursor = FakeCursor(rows=[])
    insert_python_cursor = FakeCursor(lastrowid=1)
    insert_redis_cursor = FakeCursor(lastrowid=2)
    connection = SequenceConnection(
        [create_cursor, empty_check_cursor, insert_python_cursor, insert_redis_cursor]
    )

    MySQLClient(connection=connection, seed_csv_path=seed_csv)

    assert "CREATE TABLE IF NOT EXISTS `faq`" in create_cursor.executed[0][0]
    assert "SELECT id, subject, question, answer FROM `faq`" in empty_check_cursor.executed[0][0]
    assert insert_python_cursor.executed[0][1] == (
        "Python学科",
        "Python 如何创建虚拟环境",
        "使用 python -m venv .venv",
    )
    assert insert_redis_cursor.executed[0][1] == (
        "Redis学科",
        "Redis 缓存如何设置过期时间",
        "使用 expire 命令设置过期时间",
    )
    assert connection.commit_count == 3


def test_mysql_client_auto_import_skips_seed_csv_when_table_has_data(tmp_path: Path):
    seed_csv = tmp_path / "faq.csv"
    seed_csv.write_text(
        "学科名称,问题,答案\n"
        "Redis学科,Redis 缓存如何设置过期时间,使用 expire 命令设置过期时间\n",
        encoding="utf-8",
    )
    create_cursor = FakeCursor()
    existing_check_cursor = FakeCursor(
        rows=[{"id": 1, "subject": "Python学科", "question": "已有问题", "answer": "已有答案"}]
    )
    connection = SequenceConnection([create_cursor, existing_check_cursor])

    MySQLClient(connection=connection, seed_csv_path=seed_csv)

    assert "CREATE TABLE IF NOT EXISTS `faq`" in create_cursor.executed[0][0]
    assert "SELECT id, subject, question, answer FROM `faq`" in existing_check_cursor.executed[0][0]
    assert connection.commit_count == 1
