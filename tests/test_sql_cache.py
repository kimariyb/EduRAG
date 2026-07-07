import json

from core.sql.cache import RedisClient


class FakeRedisConnection:
    def __init__(self):
        self.store = {}
        self.set_calls = []
        self.delete_calls = []

    def get(self, key):
        return self.store.get(key)

    def set(self, key, value, ex=None):
        self.store[key] = value
        self.set_calls.append((key, value, ex))
        return True

    def delete(self, *keys):
        self.delete_calls.append(keys)
        removed = 0
        for key in keys:
            if key in self.store:
                removed += 1
                del self.store[key]
        return removed

    def exists(self, key):
        return int(key in self.store)


def test_redis_client_stores_and_loads_questions_as_json():
    connection = FakeRedisConnection()
    client = RedisClient(client=connection)
    questions = [{"id": 1, "question": "Python 如何创建虚拟环境", "subject": "Python学科"}]

    client.store_questions(questions)

    assert json.loads(connection.store["faq:questions"]) == questions
    assert client.get_questions() == questions


def test_redis_client_stores_and_loads_tokenized_questions():
    connection = FakeRedisConnection()
    client = RedisClient(client=connection)
    tokens = [["Python", "虚拟环境"], ["Redis", "缓存"]]

    client.store_tokenized_questions(tokens)

    assert client.get_tokenized_questions() == tokens


def test_redis_client_caches_and_queries_answer():
    connection = FakeRedisConnection()
    client = RedisClient(client=connection)

    client.cache_answer(12, "缓存答案")

    assert connection.store["faq:answer:12"] == "缓存答案"
    assert client.get_answer(12) == "缓存答案"
    assert client.query_answer(12) == "缓存答案"


def test_redis_client_decodes_json_bytes():
    connection = FakeRedisConnection()
    connection.store["faq:questions"] = b'[{"id": 1, "question": "Redis"}]'
    client = RedisClient(client=connection)

    assert client.get_questions() == [{"id": 1, "question": "Redis"}]


def test_redis_client_delete_and_exists_delegate_to_connection():
    connection = FakeRedisConnection()
    connection.store["faq:answer:1"] = "answer"
    client = RedisClient(client=connection)

    assert client.exists("faq:answer:1")
    assert client.delete("faq:answer:1") == 1
    assert not client.exists("faq:answer:1")
