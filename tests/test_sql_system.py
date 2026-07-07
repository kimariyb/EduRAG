from core.sql.system import MySqlQASystem


class FakeRetriever:
    def __init__(self, answer):
        self.answer_value = answer
        self.queries = []

    def answer(self, query, threshold=None):
        self.queries.append((query, threshold))
        return self.answer_value


class FakeLog:
    def __init__(self):
        self.records = []

    def info(self, message, *args):
        self.records.append(("info", message, args))

    def warning(self, message, *args):
        self.records.append(("warning", message, args))


class RealCaseRedis:
    def __init__(self):
        self.store = {}

    def get(self, key):
        return self.store.get(key)

    def set(self, key, value):
        self.store[key] = value


class RealCaseMySQL:
    def __init__(self):
        self.create_table_calls = 0
        self.insert_calls = []
        self.question_calls = 0
        self.answer_calls = []
        self.questions = [
            {
                "id": 101,
                "subject": "Python学科",
                "question": "linux 查看指定进程的资源占用情况",
            },
            {
                "id": 102,
                "subject": "Python学科",
                "question": 'no module name "MySQLdb"',
            },
            {
                "id": 103,
                "subject": "Python学科",
                "question": "OSError:mysql_config not found",
            },
        ]
        self.answers = {
            101: 'top -p `pgrep python | tr "\\n" "," | sed \'s/,$//\'`',
            102: (
                "pip install PyMySQL，将数据库连接改为 "
                "mysql+pymysql://username:password@server/db，接下来的操作就一切正常了。"
            ),
            103: "yum install mysql-devel gcc gcc-devel python-deve",
        }

    def create_table(self):
        self.create_table_calls += 1

    def list_faqs(self, limit=None, offset=0):
        return list(self.questions[:limit])

    def insert_faq(self, *, question, answer, subject=None):
        self.insert_calls.append((subject, question, answer))
        return len(self.questions) + len(self.insert_calls)

    def fetch_faq_questions(self):
        self.question_calls += 1
        return list(self.questions)

    def fetch_faq_answer(self, question_id):
        self.answer_calls.append(question_id)
        return self.answers[question_id]


def test_sql_system_query_returns_answer_and_logs_elapsed_time(monkeypatch):
    fake_log = FakeLog()
    monkeypatch.setattr("core.sql.system.log", fake_log)
    retriever = FakeRetriever("缓存答案")
    system = MySqlQASystem(retriever=retriever, threshold=0.85)

    answer = system.query("Redis 缓存如何设置过期时间")

    assert answer == "缓存答案"
    assert retriever.queries == [("Redis 缓存如何设置过期时间", 0.85)]
    assert any("duration_ms" in record[1] for record in fake_log.records)


def test_sql_system_query_returns_fallback_and_logs_elapsed_time(monkeypatch):
    fake_log = FakeLog()
    monkeypatch.setattr("core.sql.system.log", fake_log)
    system = MySqlQASystem(retriever=FakeRetriever(None), fallback_answer="未命中")

    answer = system.query("未知问题")

    assert answer == "未命中"
    assert any(level == "warning" for level, _, _ in fake_log.records)
    assert any("duration_ms" in record[1] for record in fake_log.records)


def test_mysql_system_answers_real_python_mysql_dependency_case():
    redis_client = RealCaseRedis()
    mysql_client = RealCaseMySQL()
    system = MySqlQASystem(mysql_client=mysql_client, redis_client=redis_client)

    answer = system.query('no module name "MySQLdb"')
    cached_answer = system.query('no module name "MySQLdb"')

    assert answer == (
        "pip install PyMySQL，将数据库连接改为 "
        "mysql+pymysql://username:password@server/db，接下来的操作就一切正常了。"
    )
    assert cached_answer == answer
    assert mysql_client.create_table_calls == 0
    assert mysql_client.insert_calls == []
    assert mysql_client.question_calls == 1
    assert mysql_client.answer_calls == [102]
    assert redis_client.store["faq:answer:102"] == answer
