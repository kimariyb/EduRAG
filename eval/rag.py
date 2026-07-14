from __future__ import annotations

import asyncio
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from langchain_openai import ChatOpenAI
from openai import DefaultAsyncHttpxClient, DefaultHttpxClient
from ragas import EvaluationDataset, evaluate as ragas_evaluate
from ragas.embeddings.base import BaseRagasEmbeddings
from ragas.llms import BaseRagasLLM, LangchainLLMWrapper
from ragas.metrics import (
    ContextPrecision,
    ContextRecall,
    Faithfulness,
    ResponseRelevancy,
)
from ragas.metrics.base import SingleTurnMetric
from ragas.run_config import RunConfig

from base.config import AppConfig, load_config
from base.logger import logger
from core.rag.parser import parse_document_from_dir
from core.rag.query import QueryClassifier
from core.rag.constants import PROFESSIONAL_CONSULTATION_CATEGORY
from core.rag.system import RAGAnswer, RAGSystem
from core.rag.vector import VectorStore
from eval.datasets import (
    TestSample,
    load_test_samples,
    normalize_question,
)


METRIC_NAMES = (
    "context_precision",
    "context_recall",
    "answer_relevancy",
    "faithfulness",
)
log = logger.bind(module=__name__)


class BGEM3RagasEmbeddings(BaseRagasEmbeddings):
    """Expose an existing BGE-M3 embedding function to Ragas."""

    def __init__(self, embedding_function: Any) -> None:
        super().__init__()
        if not callable(embedding_function):
            raise ValueError("embedding_function must be callable")
        self.embedding_function = embedding_function

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        embeddings = self.embedding_function(texts)
        dense_vectors = embeddings.get("dense")
        if dense_vectors is None or len(dense_vectors) != len(texts):
            raise ValueError("BGE-M3 did not return one dense vector per text")
        return [
            [float(value) for value in vector]
            for vector in dense_vectors
        ]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    async def aembed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        return await asyncio.to_thread(self.embed_documents, texts)

    async def aembed_query(self, text: str) -> list[float]:
        return await asyncio.to_thread(self.embed_query, text)


def build_ragas_llm(config: AppConfig) -> BaseRagasLLM:
    """Create the Ragas LLM adapter for the configured Ollama model."""
    chat_model = ChatOpenAI(
        model=config.llm.model,
        api_key=config.llm.api_key,
        base_url=config.llm.base_url,
        temperature=config.llm.temperature,
        max_completion_tokens=config.llm.max_tokens,
        reasoning_effort=config.llm.reasoning_effort,
        http_client=DefaultHttpxClient(trust_env=False),
        http_async_client=DefaultAsyncHttpxClient(trust_env=False),
        http_socket_options=(),
    )
    return LangchainLLMWrapper(chat_model)


def build_metrics(
    config: AppConfig,
    embedding_function: Any,
) -> list[SingleTurnMetric]:
    """Build the four native Ragas metrics with shared local models."""
    llm = build_ragas_llm(config)
    embeddings = BGEM3RagasEmbeddings(embedding_function)
    return [
        ContextPrecision(llm=llm),
        ContextRecall(llm=llm),
        ResponseRelevancy(llm=llm, embeddings=embeddings, strictness=1),
        Faithfulness(llm=llm),
    ]


@dataclass(frozen=True)
class RAGPrediction:
    """One RAG answer and the actual contexts used to produce it."""

    question: str
    reference: str
    retrieved_contexts: tuple[str, ...]
    response: str
    category: str
    strategy: str | None
    source_doc: str
    error: str | None = None

    def __post_init__(self) -> None:
        for field_name in ("question", "reference", "source_doc"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} cannot be empty")
            object.__setattr__(self, field_name, value.strip())

        contexts = self.retrieved_contexts
        if not isinstance(contexts, (list, tuple)) or not all(
            isinstance(context, str) and context.strip()
            for context in contexts
        ):
            raise ValueError("retrieved_contexts must contain non-empty strings")
        object.__setattr__(
            self,
            "retrieved_contexts",
            tuple(context.strip() for context in contexts),
        )

        if not isinstance(self.response, str):
            raise ValueError("response must be a string")
        object.__setattr__(self, "response", self.response.strip())
        if not isinstance(self.category, str) or not self.category.strip():
            raise ValueError("category cannot be empty")
        object.__setattr__(self, "category", self.category.strip())
        if self.strategy is not None:
            if not isinstance(self.strategy, str) or not self.strategy.strip():
                raise ValueError("strategy must be a non-empty string or None")
            object.__setattr__(self, "strategy", self.strategy.strip())
        if self.error is None:
            if not self.response:
                raise ValueError("a successful prediction requires a response")
        elif not isinstance(self.error, str) or not self.error.strip():
            raise ValueError("error must be a non-empty string or None")
        else:
            object.__setattr__(self, "error", self.error.strip())

    @classmethod
    def from_sample(
        cls,
        sample: TestSample,
        result: RAGAnswer,
    ) -> "RAGPrediction":
        return cls(
            question=sample.question,
            reference=sample.answer,
            retrieved_contexts=tuple(
                document.page_content for document in result.documents
            ),
            response=result.answer,
            category=result.category,
            strategy=result.strategy,
            source_doc=sample.source_doc,
        )

    @classmethod
    def from_error(
        cls,
        sample: TestSample,
        error: str,
    ) -> "RAGPrediction":
        return cls(
            question=sample.question,
            reference=sample.answer,
            retrieved_contexts=(),
            response="",
            category="unknown",
            strategy=None,
            source_doc=sample.source_doc,
            error=error,
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RAGPrediction":
        contexts = data.get("retrieved_contexts")
        if not isinstance(contexts, list):
            raise ValueError("retrieved_contexts must be a JSON array")
        return cls(
            question=data.get("question", ""),
            reference=data.get("reference", ""),
            retrieved_contexts=tuple(contexts),
            response=data.get("response", ""),
            category=data.get("category", ""),
            strategy=data.get("strategy"),
            source_doc=data.get("source_doc", ""),
            error=data.get("error"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "reference": self.reference,
            "retrieved_contexts": list(self.retrieved_contexts),
            "response": self.response,
            "category": self.category,
            "strategy": self.strategy,
            "source_doc": self.source_doc,
            "error": self.error,
        }


def select_evaluation_predictions(
    predictions: Sequence[RAGPrediction],
) -> list[RAGPrediction]:
    """Return successful professional predictions with actual context."""
    return [
        prediction
        for prediction in predictions
        if prediction.error is None
        and prediction.category == PROFESSIONAL_CONSULTATION_CATEGORY
        and bool(prediction.retrieved_contexts)
    ]


@dataclass(frozen=True)
class RAGEvaluationRecord:
    """Persisted Ragas scores for one prediction."""

    prediction: RAGPrediction
    metrics: Mapping[str, float]
    error: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.prediction, RAGPrediction):
            raise ValueError("prediction must be a RAGPrediction")
        if not isinstance(self.metrics, Mapping):
            raise ValueError("metrics must be a mapping")
        normalized_metrics: dict[str, float] = {}
        for name, score in self.metrics.items():
            if name not in METRIC_NAMES:
                raise ValueError(f"unsupported metric name: {name}")
            if isinstance(score, bool):
                raise ValueError("metric scores must be numeric")
            try:
                numeric_score = float(score)
            except (TypeError, ValueError) as exc:
                raise ValueError("metric scores must be numeric") from exc
            if not math.isfinite(numeric_score):
                raise ValueError("metric scores must be finite")
            normalized_metrics[name] = numeric_score
        object.__setattr__(self, "metrics", normalized_metrics)
        if self.error is None:
            if set(normalized_metrics) != set(METRIC_NAMES):
                raise ValueError(
                    "a completed evaluation requires every metric"
                )
        elif not isinstance(self.error, str) or not self.error.strip():
            raise ValueError("error must be a non-empty string or None")
        else:
            if normalized_metrics:
                raise ValueError("a failed evaluation cannot contain metrics")
            object.__setattr__(self, "error", self.error.strip())

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RAGEvaluationRecord":
        raw_metrics = data.get("metrics")
        if not isinstance(raw_metrics, Mapping):
            raise ValueError("metrics must be a JSON object")
        return cls(
            prediction=RAGPrediction.from_dict(data),
            metrics={
                name: value
                for name, value in raw_metrics.items()
            },
            error=data.get("evaluation_error"),
        )

    def to_dict(self) -> dict[str, Any]:
        data = self.prediction.to_dict()
        data.update(
            {
                "metrics": dict(self.metrics),
                "evaluation_error": self.error,
            }
        )
        return data


def _write_jsonl(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        for row in rows:
            json.dump(dict(row), file, ensure_ascii=False)
            file.write("\n")
        file.flush()
    temporary_path.replace(output_path)


def _append_jsonl(path: str | Path, row: Mapping[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as file:
        json.dump(dict(row), file, ensure_ascii=False)
        file.write("\n")
        file.flush()


def _load_jsonl(path: str | Path) -> list[Mapping[str, Any]]:
    input_path = Path(path)
    if not input_path.exists():
        return []
    rows: list[Mapping[str, Any]] = []
    with input_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, Mapping):
                    raise ValueError("JSONL row must be an object")
                rows.append(row)
            except (json.JSONDecodeError, ValueError):
                log.warning(
                    "Invalid RAG evaluation row skipped: path={}, line={}",
                    input_path,
                    line_number,
                )
    return rows


def load_predictions(path: str | Path) -> list[RAGPrediction]:
    predictions: list[RAGPrediction] = []
    for row in _load_jsonl(path):
        try:
            predictions.append(RAGPrediction.from_dict(row))
        except ValueError:
            log.warning("Invalid RAG prediction row skipped: path={}", path)
    return predictions


def load_evaluation_records(
    path: str | Path,
) -> list[RAGEvaluationRecord]:
    records: list[RAGEvaluationRecord] = []
    for row in _load_jsonl(path):
        try:
            records.append(RAGEvaluationRecord.from_dict(row))
        except ValueError:
            log.warning("Invalid RAG evaluation record skipped: path={}", path)
    return records


def generate_predictions(
    samples: Sequence[TestSample],
    rag_system: Any,
    *,
    output_path: str | Path,
    source_filter: str | None = None,
) -> list[RAGPrediction]:
    """Generate or resume RAG predictions in source-dataset order."""
    existing = {
        normalize_question(prediction.question): prediction
        for prediction in load_predictions(output_path)
        if prediction.error is None
    }
    predictions: list[RAGPrediction] = []
    for index, sample in enumerate(samples, start=1):
        key = normalize_question(sample.question)
        prediction = existing.get(key)
        if prediction is None:
            try:
                result = rag_system.generate_answer_with_trace(
                    sample.question,
                    source_filter=source_filter,
                )
                prediction = RAGPrediction.from_sample(sample, result)
            except Exception as exc:
                error = str(exc).strip() or type(exc).__name__
                log.exception(
                    "RAG prediction failed: sample={}/{}, error={}",
                    index,
                    len(samples),
                    error,
                )
                prediction = RAGPrediction.from_error(sample, error)
            _append_jsonl(output_path, prediction.to_dict())
        predictions.append(prediction)
        log.info(
            "RAG prediction progress: completed={}, total={}",
            index,
            len(samples),
        )

    _write_jsonl(
        output_path,
        [prediction.to_dict() for prediction in predictions],
    )
    return predictions


def build_evaluation_dataset(
    predictions: Sequence[RAGPrediction],
) -> EvaluationDataset:
    """Convert successful predictions to the standard Ragas schema."""
    rows = [
        {
            "user_input": prediction.question,
            "retrieved_contexts": list(prediction.retrieved_contexts),
            "response": prediction.response,
            "reference": prediction.reference,
        }
        for prediction in predictions
        if prediction.error is None
    ]
    if not rows:
        raise ValueError("no successful RAG predictions to evaluate")
    return EvaluationDataset.from_list(rows)


def evaluate_predictions(
    predictions: Sequence[RAGPrediction],
    metrics: Sequence[SingleTurnMetric],
    *,
    output_path: str | Path,
    max_workers: int,
    timeout: int,
    max_retries: int,
) -> list[RAGEvaluationRecord]:
    """Evaluate predictions one at a time for resumable Ragas checkpoints."""
    existing = {
        normalize_question(record.prediction.question): record
        for record in load_evaluation_records(output_path)
        if record.error is None
    }
    records: list[RAGEvaluationRecord] = []
    run_config = RunConfig(
        timeout=timeout,
        max_retries=1,
        max_workers=max_workers,
    )

    for index, prediction in enumerate(predictions, start=1):
        key = normalize_question(prediction.question)
        record = existing.get(key)
        if record is None and prediction.error is not None:
            record = RAGEvaluationRecord(
                prediction=prediction,
                metrics={},
                error=f"RAG prediction failed: {prediction.error}",
            )
        if record is None:
            last_error = "Ragas evaluation failed"
            for attempt in range(1, max_retries + 1):
                try:
                    dataset = build_evaluation_dataset([prediction])
                    result = ragas_evaluate(
                        dataset,
                        list(metrics),
                        run_config=run_config,
                        raise_exceptions=True,
                        show_progress=False,
                    )
                    row_scores = result.scores[0]
                    native_scores: dict[str, float] = {}
                    for name in METRIC_NAMES:
                        if name not in row_scores:
                            raise ValueError(
                                f"Ragas result is missing metric: {name}"
                            )
                        numeric_score = float(row_scores[name])
                        if not math.isfinite(numeric_score):
                            raise ValueError(
                                f"metric returned a non-finite score: {name}"
                            )
                        native_scores[name] = numeric_score
                    record = RAGEvaluationRecord(
                        prediction=prediction,
                        metrics=native_scores,
                    )
                    break
                except Exception as exc:
                    last_error = str(exc).strip() or type(exc).__name__
                    log.warning(
                        "Ragas evaluation attempt failed: sample={}/{}, "
                        "attempt={}/{}, error={}",
                        index,
                        len(predictions),
                        attempt,
                        max_retries,
                        last_error,
                    )
            if record is None:
                record = RAGEvaluationRecord(
                    prediction=prediction,
                    metrics={},
                    error=last_error,
                )
            _append_jsonl(output_path, record.to_dict())

        records.append(record)
        log.info(
            "Ragas evaluation progress: completed={}, total={}, failed={}",
            index,
            len(predictions),
            sum(item.error is not None for item in records),
        )

    _write_jsonl(output_path, [record.to_dict() for record in records])
    return records


def summarize_evaluations(
    records: Sequence[RAGEvaluationRecord],
    *,
    input_count: int,
    eligible_count: int,
) -> dict[str, Any]:
    """Compute transparent aggregate statistics from completed records."""
    if input_count < 0 or eligible_count < 0:
        raise ValueError("evaluation counts cannot be negative")
    if eligible_count > input_count:
        raise ValueError("eligible_count cannot exceed input_count")
    if eligible_count != len(records):
        raise ValueError("eligible_count must match evaluation records")

    completed = [record for record in records if record.error is None]
    metric_summaries: dict[str, dict[str, int | float | None]] = {}
    all_scores: list[float] = []
    for name in METRIC_NAMES:
        scores = [record.metrics[name] for record in completed]
        all_scores.extend(scores)
        metric_summaries[name] = {
            "mean": (
                round(sum(scores) / len(scores), 4) if scores else None
            ),
            "count": len(scores),
        }
    return {
        "input_predictions": input_count,
        "eligible_predictions": eligible_count,
        "excluded_predictions": input_count - eligible_count,
        "completed": len(completed),
        "failed": len(records) - len(completed),
        "metrics": metric_summaries,
        "overall_mean": (
            round(sum(all_scores) / len(all_scores), 4)
            if all_scores
            else None
        ),
    }


def write_summary(path: str | Path, summary: Mapping[str, Any]) -> None:
    """Atomically write the aggregate evaluation report."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(dict(summary), file, ensure_ascii=False, indent=2)
        file.write("\n")
        file.flush()
    temporary_path.replace(output_path)


def _build_rag_system(config: AppConfig) -> tuple[RAGSystem, str]:
    """Build and populate the same RAG workflow used by the CLI."""
    model_path = Path(config.rag.query_model_path)
    query_classifier = QueryClassifier.from_config(config)
    if not model_path.is_dir():
        log.info(
            "Fine-tuned query classifier not found; training started: path={}",
            model_path,
        )
        query_classifier.train_model()

    knowledge_base_path = Path(config.rag.knowledge_base_path)
    documents = parse_document_from_dir(knowledge_base_path, config=config)
    vector_store = VectorStore.from_config(config)
    vector_store.add_documents(documents)
    rag_system = RAGSystem.from_config(
        config,
        vector_store=vector_store,
        query_classifier=query_classifier,
    )
    source_filter = knowledge_base_path.name.removesuffix("_data")
    log.info(
        "RAG evaluation workflow ready: documents={}, source_filter={}",
        len(documents),
        source_filter,
    )
    return rag_system, source_filter


def main() -> None:
    """Evaluate the RAG system on the quality-filtered local dataset."""
    config = load_config()
    if config.llm.model != "qwen3.5:4b-mlx":
        raise ValueError(
            "RAG evaluation requires llm.model=qwen3.5:4b-mlx so every "
            "LLM role uses the same model"
        )

    samples = load_test_samples(config.eval.filtered_samples_path)
    if not samples:
        raise ValueError("quality-filtered RAG evaluation dataset is empty")

    rag_system, source_filter = _build_rag_system(config)
    predictions = generate_predictions(
        samples,
        rag_system,
        output_path=config.eval.rag_predictions_path,
        source_filter=source_filter,
    )
    eligible_predictions = select_evaluation_predictions(predictions)
    if not eligible_predictions:
        raise ValueError("no retrieval predictions are eligible for Ragas")
    log.info(
        "Selected retrieval predictions for Ragas: input={}, "
        "eligible={}, excluded={}",
        len(predictions),
        len(eligible_predictions),
        len(predictions) - len(eligible_predictions),
    )
    metrics = build_metrics(
        config,
        rag_system.vector_store.embedding_function,
    )
    records = evaluate_predictions(
        eligible_predictions,
        metrics,
        output_path=config.eval.rag_evaluation_path,
        max_workers=config.eval.ragas_max_workers,
        timeout=config.eval.ragas_timeout,
        max_retries=config.eval.critique_max_retries,
    )
    summary = summarize_evaluations(
        records,
        input_count=len(predictions),
        eligible_count=len(eligible_predictions),
    )
    write_summary(config.eval.rag_summary_path, summary)
    log.info(
        "RAG evaluation completed: input={}, eligible={}, completed={}, "
        "failed={}, "
        "predictions_path={}, evaluations_path={}, summary_path={}",
        len(predictions),
        len(eligible_predictions),
        summary["completed"],
        summary["failed"],
        config.eval.rag_predictions_path,
        config.eval.rag_evaluation_path,
        config.eval.rag_summary_path,
    )


if __name__ == "__main__":
    main()
