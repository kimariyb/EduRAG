import ast
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch
from langchain_core.documents import Document
from scipy.sparse import csr_array

import core.rag.system as rag_system_module
from base.config import load_config
from core.rag.constants import (
    BACKTRACKING_RETRIEVAL_STRATEGY,
    DIRECT_RETRIEVAL_STRATEGY,
    GENERAL_KNOWLEDGE_CATEGORY,
    HYDE_RETRIEVAL_STRATEGY,
    PROFESSIONAL_CONSULTATION_CATEGORY,
    SUBQUERY_RETRIEVAL_STRATEGY,
)
from core.rag.llm import (
    DEFAULT_SYSTEM_PROMPT,
    ChatLLM,
    create_openai_client,
)
from core.rag.parser import (
    _resolve_chunk_overlap,
    load_document_from_dir,
    parse_document_from_dir,
)
from core.rag.prompt import RAGPrompts
from core.rag.query import QueryClassifier
from core.rag.retrieval import StrategySelector
from core.rag.system import RAGSystem
from core.rag.vector import VectorStore


class FakeTokenizer:
    def __call__(self, texts, **kwargs):
        count = len(texts) if isinstance(texts, list) else 1
        return {
            "input_ids": torch.ones((count, 2), dtype=torch.long),
            "attention_mask": torch.ones((count, 2), dtype=torch.long),
        }

    def save_pretrained(self, path):
        self.saved_path = path


class FakeClassifierModel:
    def __init__(self, prediction=0):
        self.prediction = prediction
        self.device = None
        self.config = SimpleNamespace()

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        return self

    def __call__(self, **kwargs):
        logits = (
            torch.tensor([[0.1, 0.9]])
            if self.prediction == 1
            else torch.tensor([[0.9, 0.1]])
        )
        return SimpleNamespace(logits=logits)

    def save_pretrained(self, path):
        self.saved_path = path


class StaticQueryClassifier:
    def __init__(self, category):
        self.category = category

    def predict_category(self, query):
        return self.category


class StaticStrategySelector:
    def __init__(self, strategy=DIRECT_RETRIEVAL_STRATEGY):
        self.strategy = strategy
        self.calls = []

    def select_strategy(self, query):
        self.calls.append(query)
        return self.strategy


class FakeVectorStore:
    def __init__(self):
        self.calls = []

    def hybrid_search_with_rerank(self, query, k, source_filter=None):
        self.calls.append((query, k, source_filter))
        return [Document(page_content=f"context:{query}")]


class FakeStreamingLLM:
    def __init__(self, *, sync_response="generated query", chunks=None):
        self.sync_response = sync_response
        self.chunks = chunks or ["streamed ", "answer"]
        self.sync_calls = []
        self.stream_calls = []

    def __call__(self, prompt):
        self.sync_calls.append(prompt)
        return self.sync_response

    def stream(self, prompt):
        self.stream_calls.append(prompt)
        yield from self.chunks


class FakeEmbedding:
    dim = {"dense": 2}

    def __init__(self):
        self.calls = []

    def __call__(self, texts):
        self.calls.append(texts)
        return {
            "dense": np.array([[index, index + 1] for index in range(len(texts))]),
            "sparse": csr_array(
                [
                    [0.25, 0.75]
                    for _ in texts
                ]
            ),
        }


class FakeReranker:
    def predict(self, pairs):
        return [0.1, 0.9][: len(pairs)]


class FakeMilvusClient:
    def __init__(self):
        self.upsert_calls = []
        self.search_calls = []

    def upsert(self, **kwargs):
        self.upsert_calls.append(kwargs)

    def hybrid_search(self, **kwargs):
        self.search_calls.append(kwargs)
        return [[
            {
                "entity": {
                    "text": "child-a",
                    "parent_id": "a",
                    "parent_content": "parent-a",
                    "source": "ai",
                    "timestamp": "now",
                }
            },
            {
                "entity": {
                    "text": "child-b",
                    "parent_id": "b",
                    "parent_content": "parent-b",
                    "source": "ai",
                    "timestamp": "now",
                }
            },
        ]]


def test_prompt_templates_have_clean_boundaries_and_expected_variables():
    prompt = RAGPrompts.rag_prompt()
    classification_prompt = RAGPrompts.query_classification_prompt()

    assert prompt.template.startswith("You are an education support assistant")
    assert not prompt.template.endswith(" ")
    assert set(prompt.input_variables) == {"context", "phone", "question"}
    assert classification_prompt.template.startswith(
        "You are a precise query classification system"
    )
    assert classification_prompt.input_variables == ["query"]


def test_all_prompt_templates_are_written_in_english():
    templates = [
        RAGPrompts.query_classification_prompt(),
        RAGPrompts.rag_prompt(),
        RAGPrompts.hyde_prompt(),
        RAGPrompts.subquery_prompt(),
        RAGPrompts.backtracking_prompt(),
        StrategySelector._get_strategy_prompt(),
    ]
    chinese_range = range(0x4E00, 0xA000)

    for template in templates:
        assert not any(
            ord(character) in chinese_range
            for character in template.template
        )
    assert not any(
        ord(character) in chinese_range
        for character in DEFAULT_SYSTEM_PROMPT
    )


@pytest.mark.parametrize(
    ("prediction", "expected"),
    [
        (0, GENERAL_KNOWLEDGE_CATEGORY),
        (1, PROFESSIONAL_CONSULTATION_CATEGORY),
    ],
)
def test_query_classifier_predicts_with_injected_dependencies(
    prediction,
    expected,
):
    classifier = QueryClassifier(
        tokenizer=FakeTokenizer(),
        model=FakeClassifierModel(prediction),
        device="cpu",
    )

    assert classifier.predict_category("测试问题") == expected


def test_query_classifier_rejects_unknown_training_label():
    classifier = QueryClassifier(
        tokenizer=FakeTokenizer(),
        model=FakeClassifierModel(),
        device="cpu",
    )

    with pytest.raises(ValueError, match="unsupported query label"):
        classifier.preprocess_data(["query"], ["unknown"])


def test_query_classifier_accepts_legacy_chinese_training_labels():
    classifier = QueryClassifier(
        tokenizer=FakeTokenizer(),
        model=FakeClassifierModel(),
        device="cpu",
    )

    _, labels = classifier.preprocess_data(
        ["first", "second"],
        ["通用知识", "专业咨询"],
    )

    assert labels == [0, 1]


def test_query_classifier_from_config_uses_model_settings():
    config = load_config()
    classifier = QueryClassifier.from_config(
        config,
        tokenizer=FakeTokenizer(),
        model=FakeClassifierModel(),
    )

    assert classifier.model_path == Path(config.rag.query_model_path)
    assert classifier.base_model == config.rag.query_base_model
    assert classifier.training_data_path == Path(
        config.rag.query_training_data_path
    )
    assert classifier.device == torch.device(config.rag.model_device)
    assert classifier.model.device == torch.device(config.rag.model_device)
    assert classifier.model.config.id2label == {
        0: GENERAL_KNOWLEDGE_CATEGORY,
        1: PROFESSIONAL_CONSULTATION_CATEGORY,
    }
    assert classifier.model.config.label2id == {
        GENERAL_KNOWLEDGE_CATEGORY: 0,
        PROFESSIONAL_CONSULTATION_CATEGORY: 1,
    }
    assert (
        classifier.model.config.problem_type
        == "single_label_classification"
    )
    assert "{query}" in (
        classifier.model.config.query_classification_prompt
    )


def test_strategy_selector_normalizes_llm_output():
    completion = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="建议使用：子查询检索")
            )
        ]
    )
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return completion

    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=create),
        )
    )
    config = load_config()
    selector = StrategySelector(
        config,
        client=client,
        model="test-model",
    )

    assert selector.select_strategy("复杂问题") == SUBQUERY_RETRIEVAL_STRATEGY
    assert calls[0]["max_tokens"] == config.llm.max_tokens
    assert calls[0]["reasoning_effort"] == config.llm.reasoning_effort


def test_chat_llm_uses_typed_llm_config():
    completion = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=" answer "))]
    )
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return completion

    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=create),
        )
    )
    config = load_config()
    llm = ChatLLM(config, client=client)

    assert llm("prompt") == "answer"
    assert calls[0]["model"] == config.llm.model
    assert calls[0]["temperature"] == config.llm.temperature
    assert calls[0]["max_tokens"] == config.llm.max_tokens
    assert calls[0]["reasoning_effort"] == config.llm.reasoning_effort


def test_chat_llm_streams_non_empty_content_fragments():
    chunks = [
        SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content="first "))]
        ),
        SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content=None))]
        ),
        SimpleNamespace(choices=[]),
        SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content="second"))]
        ),
    ]
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return iter(chunks)

    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=create),
        )
    )
    config = load_config()
    llm = ChatLLM(config, client=client)

    assert list(llm.stream("prompt")) == ["first ", "second"]
    assert calls[0]["stream"] is True
    assert calls[0]["max_tokens"] == config.llm.max_tokens
    assert calls[0]["reasoning_effort"] == config.llm.reasoning_effort


def test_openai_compatible_client_uses_ollama_config(monkeypatch):
    calls = {}
    http_client_calls = []
    openai_module = ModuleType("openai")

    class FakeDefaultHttpxClient:
        def __init__(self, **kwargs):
            http_client_calls.append(kwargs)

    class FakeOpenAI:
        def __init__(self, **kwargs):
            calls.update(kwargs)

    openai_module.DefaultHttpxClient = FakeDefaultHttpxClient
    openai_module.OpenAI = FakeOpenAI
    monkeypatch.setitem(sys.modules, "openai", openai_module)
    config = load_config()

    client = create_openai_client(config)

    assert isinstance(client, FakeOpenAI)
    assert calls["api_key"] == "ollama"
    assert calls["base_url"] == "http://localhost:11434/v1"
    assert isinstance(calls["http_client"], FakeDefaultHttpxClient)
    assert http_client_calls == [{"trust_env": False}]


def test_rag_system_general_query_skips_retrieval():
    store = FakeVectorStore()
    selector = StaticStrategySelector()
    system = RAGSystem(
        store,
        lambda prompt: "general answer",
        query_classifier=StaticQueryClassifier(GENERAL_KNOWLEDGE_CATEGORY),
        strategy_selector=selector,
    )

    assert system.generate_answer("通用问题") == "general answer"
    assert store.calls == []
    assert selector.calls == []


def test_rag_system_returns_english_fallback_answer():
    def failing_llm(prompt):
        raise RuntimeError("unavailable")

    system = RAGSystem(
        FakeVectorStore(),
        failing_llm,
        query_classifier=StaticQueryClassifier(GENERAL_KNOWLEDGE_CATEGORY),
        strategy_selector=StaticStrategySelector(),
        customer_service_phone="12345",
    )

    assert system.generate_answer("question") == (
        "Sorry, we could not process your general knowledge question. "
        "Please contact customer service at 12345."
    )


def test_rag_system_streams_final_answer_without_retrieval_for_general_query():
    store = FakeVectorStore()
    llm = FakeStreamingLLM()
    selector = StaticStrategySelector()
    system = RAGSystem(
        store,
        llm,
        query_classifier=StaticQueryClassifier(GENERAL_KNOWLEDGE_CATEGORY),
        strategy_selector=selector,
    )

    assert list(system.generate_answer_stream("general question")) == [
        "streamed ",
        "answer",
    ]
    assert store.calls == []
    assert selector.calls == []
    assert llm.sync_calls == []
    assert len(llm.stream_calls) == 1


def test_rag_system_streaming_keeps_hyde_generation_synchronous():
    store = FakeVectorStore()
    llm = FakeStreamingLLM(sync_response="generated search query")
    system = RAGSystem(
        store,
        llm,
        query_classifier=StaticQueryClassifier(
            PROFESSIONAL_CONSULTATION_CATEGORY
        ),
        strategy_selector=StaticStrategySelector(HYDE_RETRIEVAL_STRATEGY),
        retrieval_k=4,
    )

    assert "".join(
        system.generate_answer_stream(
            "course question",
            source_filter="ai",
        )
    ) == "streamed answer"
    assert len(llm.sync_calls) == 1
    assert len(llm.stream_calls) == 1
    assert store.calls == [("generated search query", 4, "ai")]
    assert "context:generated search query" in llm.stream_calls[0]


def test_rag_system_streaming_supports_synchronous_llm_callables():
    system = RAGSystem(
        FakeVectorStore(),
        lambda prompt: "synchronous answer",
        query_classifier=StaticQueryClassifier(GENERAL_KNOWLEDGE_CATEGORY),
        strategy_selector=StaticStrategySelector(),
    )

    assert list(system.generate_answer_stream("question")) == [
        "synchronous answer"
    ]


def test_rag_system_streaming_yields_english_fallback_on_generation_error():
    class FailingStreamingLLM:
        def __call__(self, prompt):
            raise AssertionError("the synchronous path must not be used")

        def stream(self, prompt):
            raise RuntimeError("unavailable")
            yield

    system = RAGSystem(
        FakeVectorStore(),
        FailingStreamingLLM(),
        query_classifier=StaticQueryClassifier(GENERAL_KNOWLEDGE_CATEGORY),
        strategy_selector=StaticStrategySelector(),
        customer_service_phone="12345",
    )

    assert list(system.generate_answer_stream("question")) == [
        "Sorry, we could not process your general knowledge question. "
        "Please contact customer service at 12345."
    ]


def test_rag_system_returns_stable_identity_answer_without_calling_llm():
    class FailingLLM:
        def __call__(self, prompt):
            raise AssertionError("identity answers must not call the LLM")

    system = RAGSystem(
        FakeVectorStore(),
        FailingLLM(),
        query_classifier=StaticQueryClassifier(GENERAL_KNOWLEDGE_CATEGORY),
        strategy_selector=StaticStrategySelector(),
    )

    assert system.generate_answer("你是谁？") == (
        "我是 EduRAG，一个面向 IT 教育培训的智能问答助手。"
        "我可以回答 FAQ、课程与培训咨询，并基于知识库提供相关帮助。"
    )


def test_rag_system_streams_stable_identity_answer_without_calling_llm():
    class FailingLLM:
        def __call__(self, prompt):
            raise AssertionError("identity answers must not call the LLM")

        def stream(self, prompt):
            raise AssertionError("identity answers must not stream from the LLM")
            yield

    system = RAGSystem(
        FakeVectorStore(),
        FailingLLM(),
        query_classifier=StaticQueryClassifier(GENERAL_KNOWLEDGE_CATEGORY),
        strategy_selector=StaticStrategySelector(),
    )

    assert list(system.generate_answer_stream("你是谁？")) == [
        "我是 EduRAG，一个面向 IT 教育培训的智能问答助手。"
        "我可以回答 FAQ、课程与培训咨询，并基于知识库提供相关帮助。"
    ]


def test_rag_system_from_config_uses_rag_settings():
    config = load_config()
    store = FakeVectorStore()
    classifier = StaticQueryClassifier(GENERAL_KNOWLEDGE_CATEGORY)
    selector = StaticStrategySelector()

    system = RAGSystem.from_config(
        config,
        vector_store=store,
        llm=lambda prompt: "answer",
        query_classifier=classifier,
        strategy_selector=selector,
    )

    assert system.vector_store is store
    assert system.query_classifier is classifier
    assert system.strategy_selector is selector
    assert system.retrieval_k == config.rag.retrieval_k
    assert system.candidate_m == config.rag.candidate_m
    assert system.customer_service_phone == config.rag.customer_service_phone


@pytest.mark.parametrize(
    ("stream_enabled", "expected_method"),
    [(True, "stream"), (False, "sync")],
)
def test_system_main_builds_and_runs_complete_workflow(
    monkeypatch,
    tmp_path,
    capsys,
    stream_enabled,
    expected_method,
):
    knowledge_base = tmp_path / "ai_data"
    knowledge_base.mkdir()
    model_path = tmp_path / "query_classifier"
    config = load_config()
    config = replace(
        config,
        llm=replace(config.llm, stream=stream_enabled),
        rag=replace(
            config.rag,
            knowledge_base_path=str(knowledge_base),
            query_model_path=str(model_path),
        ),
    )
    calls = {
        "trained": 0,
        "indexed": [],
        "queries": [],
    }
    classifier = SimpleNamespace(
        train_model=lambda: calls.__setitem__("trained", 1)
    )
    vector_store = SimpleNamespace(
        add_documents=lambda documents: calls["indexed"].extend(documents)
    )
    rag_system = SimpleNamespace(
        generate_answer=lambda query, source_filter=None: (
            calls["queries"].append(("sync", query, source_filter))
            or "answer"
        ),
        generate_answer_stream=lambda query, source_filter=None: (
            calls["queries"].append(("stream", query, source_filter))
            or iter(["ans", "wer"])
        ),
    )
    documents = [Document(page_content="knowledge")]
    inputs = iter(["course question", "exit"])

    monkeypatch.setattr(rag_system_module, "load_config", lambda: config)
    monkeypatch.setattr(
        rag_system_module,
        "parse_document_from_dir",
        lambda path, config=None: documents,
    )
    monkeypatch.setattr(
        rag_system_module,
        "QueryClassifier",
        SimpleNamespace(from_config=lambda config: classifier),
    )
    monkeypatch.setattr(
        rag_system_module,
        "VectorStore",
        SimpleNamespace(from_config=lambda config: vector_store),
    )
    monkeypatch.setattr(
        rag_system_module,
        "RAGSystem",
        SimpleNamespace(from_config=lambda config, **kwargs: rag_system),
    )
    monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))

    rag_system_module.main()

    assert calls["trained"] == 1
    assert calls["indexed"] == documents
    assert calls["queries"] == [
        (expected_method, "course question", "ai")
    ]
    assert capsys.readouterr().out == "answer\n"


def test_rag_system_direct_retrieval_propagates_source_filter():
    store = FakeVectorStore()
    captured_prompts = []

    def llm(prompt):
        captured_prompts.append(prompt)
        return "rag answer"

    system = RAGSystem(
        store,
        llm,
        query_classifier=StaticQueryClassifier(
            PROFESSIONAL_CONSULTATION_CATEGORY
        ),
        strategy_selector=StaticStrategySelector(),
        retrieval_k=7,
        candidate_m=2,
    )

    assert system.generate_answer("专业问题", source_filter="ai") == "rag answer"
    assert store.calls == [("专业问题", 7, "ai")]
    assert "context:专业问题" in captured_prompts[-1]


def test_rag_system_returns_answer_with_actual_retrieval_trace():
    store = FakeVectorStore()
    system = RAGSystem(
        store,
        lambda prompt: "traced answer",
        query_classifier=StaticQueryClassifier(
            PROFESSIONAL_CONSULTATION_CATEGORY
        ),
        strategy_selector=StaticStrategySelector(
            DIRECT_RETRIEVAL_STRATEGY
        ),
        retrieval_k=4,
    )

    result = system.generate_answer_with_trace(
        "课程问题",
        source_filter="ai",
    )

    assert result.answer == "traced answer"
    assert result.category == PROFESSIONAL_CONSULTATION_CATEGORY
    assert result.strategy == DIRECT_RETRIEVAL_STRATEGY
    assert [document.page_content for document in result.documents] == [
        "context:课程问题"
    ]
    assert store.calls == [("课程问题", 4, "ai")]


@pytest.mark.parametrize(
    "strategy",
    [HYDE_RETRIEVAL_STRATEGY, BACKTRACKING_RETRIEVAL_STRATEGY],
)
def test_rag_system_generated_query_strategies_propagate_source_filter(strategy):
    store = FakeVectorStore()
    system = RAGSystem(
        store,
        lambda prompt: "generated query",
        query_classifier=StaticQueryClassifier(
            PROFESSIONAL_CONSULTATION_CATEGORY
        ),
        strategy_selector=StaticStrategySelector(strategy),
        retrieval_k=4,
    )

    documents = system.retrieve_and_merge(
        "original query",
        source_filter="ai",
        strategy=strategy,
    )

    assert documents[0].page_content == "context:generated query"
    assert store.calls == [("generated query", 4, "ai")]


def test_rag_system_subqueries_merge_duplicate_documents():
    store = FakeVectorStore()
    system = RAGSystem(
        store,
        lambda prompt: "first\nfirst",
        query_classifier=StaticQueryClassifier(
            PROFESSIONAL_CONSULTATION_CATEGORY
        ),
        strategy_selector=StaticStrategySelector(SUBQUERY_RETRIEVAL_STRATEGY),
        retrieval_k=2,
    )

    documents = system.retrieve_and_merge(
        "original query",
        source_filter="ai",
        strategy=SUBQUERY_RETRIEVAL_STRATEGY,
    )

    assert [document.page_content for document in documents] == ["context:first"]
    assert store.calls == [("first", 2, "ai"), ("first", 2, "ai")]


def test_vector_store_upserts_documents_and_reranks_search_results():
    client = FakeMilvusClient()
    store = VectorStore(
        "knowledge",
        "localhost",
        19530,
        "default",
        client=client,
        embedding_function=FakeEmbedding(),
        reranker=FakeReranker(),
        auto_prepare=False,
    )
    documents = [
        Document(
            page_content="child",
            metadata={
                "parent_id": "parent-1",
                "parent_content": "parent",
                "source": "ai",
                "timestamp": "now",
            },
        )
    ]

    store.add_documents(documents)
    results = store.hybrid_search_with_rerank(
        "query",
        k=5,
        source_filter='ai"course',
    )

    upserted = client.upsert_calls[0]
    assert upserted["collection_name"] == "knowledge"
    assert upserted["data"][0]["dense_vector"] == [0.0, 1.0]
    assert upserted["data"][0]["sparse_vector"] == {0: 0.25, 1: 0.75}
    assert [document.page_content for document in results] == [
        "parent-b",
        "parent-a",
    ]
    assert VectorStore._source_filter_expression('ai"course') == (
        'source == "ai\\"course"'
    )


def test_vector_store_deduplicates_primary_keys_in_one_batch():
    client = FakeMilvusClient()
    embedding = FakeEmbedding()
    store = VectorStore(
        "knowledge",
        "localhost",
        19530,
        "default",
        client=client,
        embedding_function=embedding,
        reranker=FakeReranker(),
        auto_prepare=False,
    )
    document = Document(
        page_content="duplicate child",
        metadata={
            "id": "child-1",
            "parent_id": "parent-1",
            "parent_content": "parent",
            "source": "ai",
        },
    )

    store.add_documents([document, document])

    assert embedding.calls == [[document.page_content]]
    assert len(client.upsert_calls[0]["data"]) == 1


def test_vector_store_disambiguates_duplicate_text_with_chunk_metadata():
    client = FakeMilvusClient()
    store = VectorStore(
        "knowledge",
        "localhost",
        19530,
        "default",
        client=client,
        embedding_function=FakeEmbedding(),
        reranker=FakeReranker(),
        auto_prepare=False,
    )
    documents = [
        Document(
            page_content="repeated boundary text",
            metadata={
                "id": f"doc_0_parent_{index}_child_0",
                "parent_id": f"doc_0_parent_{index}",
                "parent_content": f"parent {index}",
                "source": "ai",
                "timestamp": "now",
            },
        )
        for index in range(2)
    ]

    store.add_documents(documents)
    first_ids = [row["id"] for row in client.upsert_calls[0]["data"]]
    store.add_documents(documents)
    second_ids = [row["id"] for row in client.upsert_calls[1]["data"]]

    assert len(set(first_ids)) == len(documents)
    assert second_ids == first_ids


def test_vector_store_from_config_uses_milvus_and_rag_settings():
    config = load_config()
    store = VectorStore.from_config(
        config,
        client=FakeMilvusClient(),
        embedding_function=FakeEmbedding(),
        reranker=FakeReranker(),
        auto_prepare=False,
    )

    assert store.collection_name == config.milvus.collection
    assert store.host == config.milvus.host
    assert store.port == config.milvus.port
    assert store.database == config.milvus.database
    assert store.candidate_m == config.rag.candidate_m


def test_vector_store_from_config_loads_configured_local_models(monkeypatch):
    config = load_config()
    calls = {}

    def create_embedding(**kwargs):
        calls["embedding"] = kwargs
        return FakeEmbedding()

    def create_reranker(model_path, **kwargs):
        calls["reranker"] = (model_path, kwargs)
        return FakeReranker()

    monkeypatch.setattr(
        "core.rag.vector.BGEM3EmbeddingFunction",
        create_embedding,
    )
    monkeypatch.setattr("core.rag.vector.CrossEncoder", create_reranker)

    VectorStore.from_config(
        config,
        client=FakeMilvusClient(),
        auto_prepare=False,
    )

    assert calls["embedding"]["model_name"] == config.rag.embedding_model_path
    assert calls["embedding"]["device"] == config.rag.model_device
    assert calls["reranker"] == (
        config.rag.reranker_model_path,
        {"device": config.rag.model_device},
    )


def test_parser_loads_documents_deterministically(tmp_path):
    (tmp_path / "b.txt").write_text("B", encoding="utf-8")
    (tmp_path / "a.txt").write_text("A", encoding="utf-8")
    (tmp_path / "ignored.bin").write_bytes(b"ignored")

    class FakeLoader:
        def __init__(self, path, encoding=None):
            self.path = path
            self.encoding = encoding

        def load(self):
            return [Document(page_content=Path(self.path).read_text())]

    documents = load_document_from_dir(
        tmp_path,
        loader_registry={".txt": FakeLoader},
    )

    assert [document.page_content for document in documents] == ["A", "B"]
    assert all(document.metadata["source"] == tmp_path.name for document in documents)
    assert all("timestamp" in document.metadata for document in documents)


def test_default_parser_loaders_read_plain_text_and_markdown(tmp_path):
    (tmp_path / "notes.md").write_text("# Notes\nMarkdown", encoding="utf-8")
    (tmp_path / "plain.txt").write_text("Plain text", encoding="utf-8")

    documents = load_document_from_dir(tmp_path)

    assert [document.page_content for document in documents] == [
        "# Notes\nMarkdown",
        "Plain text",
    ]


def test_parser_resolves_ratio_overlap_and_builds_parent_metadata(
    monkeypatch,
):
    document = Document(
        page_content="甲乙丙丁。戊己庚辛。壬癸子丑。",
        metadata={"file_path": "sample.txt"},
    )
    monkeypatch.setattr(
        "core.rag.parser.load_document_from_dir",
        lambda path: [document],
    )

    chunks = parse_document_from_dir(
        "unused",
        parent_chunk_size=8,
        child_chunk_size=4,
        chunk_overlap=0.25,
    )

    assert _resolve_chunk_overlap(100, 0.25) == 25
    assert chunks
    assert all("parent_id" in chunk.metadata for chunk in chunks)
    assert all("parent_content" in chunk.metadata for chunk in chunks)
    assert all("id" in chunk.metadata for chunk in chunks)


def test_parser_uses_chunk_settings_from_config(monkeypatch):
    config = load_config()
    config = replace(
        config,
        rag=replace(
            config.rag,
            parent_chunk_size=8,
            child_chunk_size=4,
            chunk_overlap=0,
        ),
    )
    document = Document(
        page_content="甲乙丙丁戊己庚辛",
        metadata={"file_path": "sample.txt"},
    )
    monkeypatch.setattr(
        "core.rag.parser.load_document_from_dir",
        lambda path: [document],
    )

    chunks = parse_document_from_dir("unused", config=config)

    assert [chunk.page_content for chunk in chunks] == ["甲乙丙丁", "戊己庚辛"]


def test_all_rag_log_message_templates_are_english():
    rag_root = Path("core/rag")
    chinese_range = range(0x4E00, 0xA000)

    for path in rag_root.rglob("*.py"):
        module = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(module):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr not in {
                "debug",
                "info",
                "warning",
                "error",
                "exception",
                "critical",
            }:
                continue
            if not node.args or not isinstance(node.args[0], ast.Constant):
                continue
            message = node.args[0].value
            if isinstance(message, str):
                assert not any(ord(character) in chinese_range for character in message), (
                    f"non-English log template in {path}: {message}"
                )
