import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from langchain_core.documents import Document
from ragas.metrics import (
    ContextPrecision,
    ContextRecall,
    Faithfulness,
    ResponseRelevancy,
)

import eval.rag as rag_eval
from core.rag.constants import (
    DIRECT_RETRIEVAL_STRATEGY,
    GENERAL_KNOWLEDGE_CATEGORY,
    PROFESSIONAL_CONSULTATION_CATEGORY,
)
from core.rag.system import RAGAnswer
from eval.datasets import TestSample as Sample


def make_ragas_sample():
    return SimpleNamespace(
        user_input="What is an N-gram language model?",
        retrieved_contexts=["An N-gram models a fixed-length token window."],
        response="It models a fixed-length token window.",
        reference="An N-gram models a fixed-length token window.",
    )


def make_test_sample(index=1):
    return Sample(
        context=f"reference context {index}",
        question=f"question {index}",
        answer=f"reference {index}",
        source_doc="knowledge.md",
    )


def make_prediction(index=1):
    return rag_eval.RAGPrediction(
        question=f"question {index}",
        reference=f"reference {index}",
        retrieved_contexts=(f"retrieved {index}",),
        response=f"response {index}",
        category=PROFESSIONAL_CONSULTATION_CATEGORY,
        strategy=DIRECT_RETRIEVAL_STRATEGY,
        source_doc="knowledge.md",
    )


def make_llm_config():
    return SimpleNamespace(
        llm=SimpleNamespace(
            model="qwen3.5:4b-mlx",
            api_key="ollama",
            base_url="http://localhost:11434/v1",
            temperature=0.0,
            max_tokens=128,
            reasoning_effort="none",
        )
    )


class FakeEmbeddingFunction:
    def __call__(self, texts):
        return {
            "dense": np.asarray(
                [[float(index), 1.0] for index, _ in enumerate(texts)]
            )
        }


def test_select_evaluation_predictions_requires_successful_retrieval():
    eligible = make_prediction()
    general = replace(
        eligible,
        question="general question",
        category=GENERAL_KNOWLEDGE_CATEGORY,
        strategy=None,
        retrieved_contexts=(),
    )
    no_context = replace(
        eligible,
        question="missing context",
        retrieved_contexts=(),
    )

    assert rag_eval.select_evaluation_predictions(
        [general, no_context, eligible]
    ) == [eligible]


def test_build_metrics_uses_native_ragas_classes():
    embedding_function = FakeEmbeddingFunction()

    metrics = rag_eval.build_metrics(
        make_llm_config(),
        embedding_function,
    )

    assert [type(metric) for metric in metrics] == [
        ContextPrecision,
        ContextRecall,
        ResponseRelevancy,
        Faithfulness,
    ]
    assert metrics[2].embeddings.embedding_function is embedding_function
    assert metrics[2].strictness == 1


def test_build_ragas_llm_supports_ragas_legacy_prompt_api():
    llm = rag_eval.build_ragas_llm(make_llm_config())

    assert hasattr(llm, "run_config")
    assert callable(llm.generate)


def test_rag_prediction_uses_actual_contexts_from_answer_trace():
    result = RAGAnswer(
        answer="generated answer",
        category=PROFESSIONAL_CONSULTATION_CATEGORY,
        strategy=DIRECT_RETRIEVAL_STRATEGY,
        documents=(
            Document(page_content="retrieved one"),
            Document(page_content="retrieved two"),
        ),
    )

    prediction = rag_eval.RAGPrediction.from_sample(
        make_test_sample(),
        result,
    )

    assert prediction.retrieved_contexts == (
        "retrieved one",
        "retrieved two",
    )
    assert prediction.response == "generated answer"
    assert rag_eval.RAGPrediction.from_dict(prediction.to_dict()) == prediction


def test_generate_predictions_resumes_successful_rows(tmp_path):
    output_path = tmp_path / "predictions.jsonl"
    existing = make_prediction(1)
    output_path.write_text(
        json.dumps(existing.to_dict(), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    calls = []

    class FakeRAGSystem:
        def generate_answer_with_trace(self, question, source_filter=None):
            calls.append((question, source_filter))
            return RAGAnswer(
                answer="response 2",
                category=PROFESSIONAL_CONSULTATION_CATEGORY,
                strategy=DIRECT_RETRIEVAL_STRATEGY,
                documents=(Document(page_content="retrieved 2"),),
            )

    predictions = rag_eval.generate_predictions(
        [make_test_sample(1), make_test_sample(2)],
        FakeRAGSystem(),
        output_path=output_path,
        source_filter="ai",
    )

    assert predictions == [existing, make_prediction(2)]
    assert calls == [("question 2", "ai")]
    assert len(output_path.read_text(encoding="utf-8").splitlines()) == 2


def test_build_evaluation_dataset_uses_standard_ragas_columns():
    dataset = rag_eval.build_evaluation_dataset([make_prediction()])

    assert dataset.to_list() == [
        {
            "user_input": "question 1",
            "retrieved_contexts": ["retrieved 1"],
            "response": "response 1",
            "reference": "reference 1",
        }
    ]


def test_evaluation_record_persists_native_float_scores():
    scores = {
        "context_precision": 0.75,
        "context_recall": 1.0,
        "answer_relevancy": 0.8,
        "faithfulness": 0.5,
    }
    record = rag_eval.RAGEvaluationRecord(make_prediction(), scores)

    assert rag_eval.RAGEvaluationRecord.from_dict(record.to_dict()) == record


@pytest.mark.parametrize(
    "score",
    [float("nan"), float("inf"), -float("inf")],
)
def test_evaluation_record_rejects_non_finite_scores(score):
    scores = {name: 0.5 for name in rag_eval.METRIC_NAMES}
    scores["faithfulness"] = score

    with pytest.raises(ValueError, match="finite"):
        rag_eval.RAGEvaluationRecord(make_prediction(), scores)


def test_evaluate_predictions_uses_native_ragas_scores(
    monkeypatch,
    tmp_path,
):
    calls = []
    metrics = [
        SimpleNamespace(name=name)
        for name in rag_eval.METRIC_NAMES
    ]
    expected_scores = {
        "context_precision": 0.75,
        "context_recall": 1.0,
        "answer_relevancy": 0.8,
        "faithfulness": 0.5,
    }

    def fake_ragas_evaluate(dataset, active_metrics, **kwargs):
        calls.append((dataset, active_metrics, kwargs))
        return SimpleNamespace(scores=[expected_scores])

    monkeypatch.setattr(rag_eval, "ragas_evaluate", fake_ragas_evaluate)
    output_path = tmp_path / "evaluations.jsonl"

    records = rag_eval.evaluate_predictions(
        [make_prediction()],
        metrics,
        output_path=output_path,
        max_workers=1,
        timeout=120,
        max_retries=2,
    )

    assert len(calls) == 1
    assert calls[0][2]["raise_exceptions"] is True
    assert calls[0][2]["run_config"].max_workers == 1
    assert records[0].error is None
    assert records[0].metrics == expected_scores
    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert saved["metrics"] == expected_scores


def test_summarize_evaluations_reports_excluded_predictions():
    completed = rag_eval.RAGEvaluationRecord(
        prediction=make_prediction(1),
        metrics={
            "context_precision": 0.8,
            "context_recall": 0.6,
            "answer_relevancy": 0.7,
            "faithfulness": 0.9,
        },
    )
    failed = rag_eval.RAGEvaluationRecord(
        prediction=make_prediction(2),
        metrics={},
        error="judge failed",
    )

    summary = rag_eval.summarize_evaluations(
        [completed, failed],
        input_count=10,
        eligible_count=2,
    )

    assert summary["input_predictions"] == 10
    assert summary["eligible_predictions"] == 2
    assert summary["excluded_predictions"] == 8
    assert summary["completed"] == 1
    assert summary["failed"] == 1
    assert summary["metrics"]["faithfulness"] == {
        "mean": 0.9,
        "count": 1,
    }
    assert summary["overall_mean"] == 0.75


def test_main_runs_only_retrieval_predictions_through_native_ragas(
    monkeypatch,
    tmp_path,
):
    config = SimpleNamespace(
        llm=SimpleNamespace(model="qwen3.5:4b-mlx"),
        eval=SimpleNamespace(
            filtered_samples_path=str(tmp_path / "filtered.jsonl"),
            rag_predictions_path=str(tmp_path / "predictions.jsonl"),
            rag_evaluation_path=str(tmp_path / "evaluations.jsonl"),
            rag_summary_path=str(tmp_path / "summary.json"),
            ragas_max_workers=1,
            ragas_timeout=120,
            critique_max_retries=2,
        ),
    )
    samples = [make_test_sample()]
    embedding_function = object()
    rag_system = SimpleNamespace(
        vector_store=SimpleNamespace(
            embedding_function=embedding_function,
        )
    )
    eligible_prediction = make_prediction()
    general_prediction = replace(
        make_prediction(2),
        category=GENERAL_KNOWLEDGE_CATEGORY,
        strategy=None,
        retrieved_contexts=(),
    )
    predictions = [eligible_prediction, general_prediction]
    eligible_predictions = [eligible_prediction]
    metrics = [object()]
    records = [object()]
    summary = {"completed": 1, "failed": 0}
    calls = {}

    monkeypatch.setattr(rag_eval, "load_config", lambda: config, raising=False)
    monkeypatch.setattr(
        rag_eval,
        "load_test_samples",
        lambda path: calls.update(samples_path=path) or samples,
        raising=False,
    )
    monkeypatch.setattr(
        rag_eval,
        "_build_rag_system",
        lambda active_config: (rag_system, "ai"),
        raising=False,
    )
    monkeypatch.setattr(
        rag_eval,
        "generate_predictions",
        lambda active_samples, active_system, **kwargs: (
            calls.update(
                predictions=(active_samples, active_system, kwargs)
            )
            or predictions
        ),
    )
    monkeypatch.setattr(
        rag_eval,
        "select_evaluation_predictions",
        lambda active_predictions: (
            calls.update(selected_predictions=active_predictions)
            or eligible_predictions
        ),
    )
    monkeypatch.setattr(
        rag_eval,
        "build_metrics",
        lambda active_config, active_embeddings: (
            calls.update(
                metric_dependencies=(active_config, active_embeddings)
            )
            or metrics
        ),
    )
    monkeypatch.setattr(
        rag_eval,
        "evaluate_predictions",
        lambda active_predictions, active_metrics, **kwargs: (
            calls.update(
                evaluations=(active_predictions, active_metrics, kwargs)
            )
            or records
        ),
    )
    monkeypatch.setattr(
        rag_eval,
        "summarize_evaluations",
        lambda active_records, **kwargs: (
            calls.update(summary_records=(active_records, kwargs))
            or summary
        ),
    )
    monkeypatch.setattr(
        rag_eval,
        "write_summary",
        lambda path, data: calls.update(summary_output=(path, data)),
        raising=False,
    )

    rag_eval.main()

    assert calls["samples_path"] == config.eval.filtered_samples_path
    assert calls["predictions"] == (
        samples,
        rag_system,
        {
            "output_path": config.eval.rag_predictions_path,
            "source_filter": "ai",
        },
    )
    assert calls["selected_predictions"] == predictions
    assert calls["metric_dependencies"] == (
        config,
        embedding_function,
    )
    assert calls["evaluations"] == (
        eligible_predictions,
        metrics,
        {
            "output_path": config.eval.rag_evaluation_path,
            "max_workers": 1,
            "timeout": 120,
            "max_retries": 2,
        },
    )
    assert calls["summary_records"] == (
        records,
        {"input_count": 2, "eligible_count": 1},
    )
    assert calls["summary_output"] == (
        config.eval.rag_summary_path,
        summary,
    )
