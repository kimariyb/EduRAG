from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

from base.config import AppConfig, load_config
from base.logger import logger
from core.rag.constants import (
    BACKTRACKING_RETRIEVAL_STRATEGY,
    DEFAULT_CANDIDATE_M,
    DEFAULT_CUSTOMER_SERVICE_PHONE,
    DEFAULT_RETRIEVAL_K,
    DIRECT_RETRIEVAL_STRATEGY,
    GENERAL_KNOWLEDGE_CATEGORY,
    HYDE_RETRIEVAL_STRATEGY,
    QUERY_CATEGORY_LOG_NAMES,
    RETRIEVAL_STRATEGY_LOG_NAMES,
    SUBQUERY_RETRIEVAL_STRATEGY,
    normalize_query_category,
    normalize_retrieval_strategy,
)
from core.rag.prompt import RAGPrompts
from core.rag.query import QueryClassifier
from core.rag.retrieval import StrategySelector
from core.rag.parser import parse_document_from_dir
from core.rag.vector import VectorStore


log = logger.bind(module=__name__)
LLMCallable = Callable[[str], str]


@dataclass(frozen=True)
class RAGAnswer:
    """A final answer together with the retrieval decisions that produced it."""

    answer: str
    category: str
    strategy: str | None
    documents: tuple[Document, ...]


@dataclass(frozen=True)
class _PreparedRAGQuery:
    prompt: str
    category: str
    strategy: str | None
    documents: tuple[Document, ...]


class RAGSystem:
    def __init__(
        self,
        vector_store: Any,
        llm: LLMCallable,
        *,
        query_classifier: Any | None = None,
        strategy_selector: Any | None = None,
        rag_prompt: PromptTemplate | None = None,
        retrieval_k: int = DEFAULT_RETRIEVAL_K,
        candidate_m: int = DEFAULT_CANDIDATE_M,
        customer_service_phone: str = DEFAULT_CUSTOMER_SERVICE_PHONE,
    ) -> None:
        if retrieval_k <= 0:
            raise ValueError("retrieval_k must be greater than 0")
        if candidate_m <= 0:
            raise ValueError("candidate_m must be greater than 0")

        self.vector_store = vector_store
        self.llm = llm
        self.rag_prompt = rag_prompt or RAGPrompts.rag_prompt()
        self.query_classifier = query_classifier or QueryClassifier.from_config(
            load_config()
        )
        self.strategy_selector = strategy_selector or StrategySelector()
        self.retrieval_k = retrieval_k
        self.candidate_m = candidate_m
        self.customer_service_phone = customer_service_phone
        log.info(
            "RAG system initialized: retrieval_k={}, candidate_m={}",
            self.retrieval_k,
            self.candidate_m,
        )

    @classmethod
    def from_config(
        cls,
        config: AppConfig | None = None,
        *,
        vector_store: Any | None = None,
        llm: LLMCallable | None = None,
        query_classifier: Any | None = None,
        strategy_selector: Any | None = None,
        rag_prompt: PromptTemplate | None = None,
    ) -> "RAGSystem":
        active_config = config or load_config()
        if vector_store is None:
            from core.rag.vector import VectorStore

            vector_store = VectorStore.from_config(active_config)
        if llm is None:
            from core.rag.llm import ChatLLM

            llm = ChatLLM(active_config)
        return cls(
            vector_store,
            llm,
            query_classifier=(
                query_classifier or QueryClassifier.from_config(active_config)
            ),
            strategy_selector=(
                strategy_selector or StrategySelector(active_config)
            ),
            rag_prompt=rag_prompt,
            retrieval_k=active_config.rag.retrieval_k,
            candidate_m=active_config.rag.candidate_m,
            customer_service_phone=(
                active_config.rag.customer_service_phone
            ),
        )

    def _search(
        self,
        query: str,
        source_filter: str | None = None,
    ) -> list[Document]:
        return self.vector_store.hybrid_search_with_rerank(
            query,
            k=self.retrieval_k,
            source_filter=source_filter,
        )

    def _retrieve_with_hyde(
        self,
        query: str,
        source_filter: str | None = None,
    ) -> list[Document]:
        log.info("HyDE retrieval started")
        try:
            prompt = RAGPrompts.hyde_prompt().format(query=query)
            hypothetical_answer = self.llm(prompt).strip()
            if not hypothetical_answer:
                log.warning("HyDE generation returned empty text")
                return []
            documents = self._search(hypothetical_answer, source_filter)
            log.info("HyDE retrieval completed: documents={}", len(documents))
            return documents
        except Exception:
            log.exception("HyDE retrieval failed")
            return []

    def _retrieve_with_subqueries(
        self,
        query: str,
        source_filter: str | None = None,
    ) -> list[Document]:
        log.info("Subquery retrieval started")
        try:
            prompt = RAGPrompts.subquery_prompt().format(query=query)
            generated_text = self.llm(prompt).strip()
            subqueries = [
                item.strip()
                for item in generated_text.splitlines()
                if item.strip()
            ]
            if not subqueries:
                log.warning("Subquery generation returned no usable queries")
                return []

            documents: list[Document] = []
            for subquery in subqueries:
                matches = self._search(subquery, source_filter)
                documents.extend(matches)
                log.info(
                    "Subquery retrieval completed: documents={}",
                    len(matches),
                )

            unique_documents = list(
                {document.page_content: document for document in documents}.values()
            )
            log.info(
                "Subquery results merged: total={}, unique={}",
                len(documents),
                len(unique_documents),
            )
            return unique_documents
        except Exception:
            log.exception("Subquery retrieval failed")
            return []

    def _retrieve_with_backtracking(
        self,
        query: str,
        source_filter: str | None = None,
    ) -> list[Document]:
        log.info("Backtracking retrieval started")
        try:
            prompt = RAGPrompts.backtracking_prompt().format(query=query)
            simplified_query = self.llm(prompt).strip()
            if not simplified_query:
                log.warning("Backtracking generation returned empty text")
                return []
            documents = self._search(simplified_query, source_filter)
            log.info(
                "Backtracking retrieval completed: documents={}",
                len(documents),
            )
            return documents
        except Exception:
            log.exception("Backtracking retrieval failed")
            return []

    def retrieve_and_merge(
        self,
        query: str,
        source_filter: str | None = None,
        strategy: str | None = None,
    ) -> list[Document]:
        selected_strategy = normalize_retrieval_strategy(
            strategy or self.strategy_selector.select_strategy(query)
        )
        if selected_strategy == BACKTRACKING_RETRIEVAL_STRATEGY:
            documents = self._retrieve_with_backtracking(query, source_filter)
        elif selected_strategy == SUBQUERY_RETRIEVAL_STRATEGY:
            documents = self._retrieve_with_subqueries(query, source_filter)
        elif selected_strategy == HYDE_RETRIEVAL_STRATEGY:
            documents = self._retrieve_with_hyde(query, source_filter)
        else:
            selected_strategy = DIRECT_RETRIEVAL_STRATEGY
            log.info("Direct retrieval started")
            documents = self._search(query, source_filter)

        context_documents = documents[: self.candidate_m]
        log.info(
            "Retrieval completed: strategy={}, candidates={}, selected={}",
            RETRIEVAL_STRATEGY_LOG_NAMES[selected_strategy],
            len(documents),
            len(context_documents),
        )
        return context_documents

    def _prepare_answer_prompt(
        self,
        query: str,
        source_filter: str | None = None,
    ) -> _PreparedRAGQuery:
        """Classify, retrieve, and format the final answer prompt."""
        category = normalize_query_category(
            self.query_classifier.predict_category(query)
        )
        log.info(
            "RAG query classified: category={}",
            QUERY_CATEGORY_LOG_NAMES.get(category, "unknown"),
        )

        context_documents: list[Document] = []
        selected_strategy: str | None = None
        if category == GENERAL_KNOWLEDGE_CATEGORY:
            log.info("Using direct LLM path for general knowledge query")
        else:
            try:
                selected_strategy = normalize_retrieval_strategy(
                    self.strategy_selector.select_strategy(query)
                )
                context_documents = self.retrieve_and_merge(
                    query,
                    source_filter=source_filter,
                    strategy=selected_strategy,
                )
            except Exception:
                log.exception("RAG retrieval failed; using empty context")
            log.info(
                "RAG context prepared: documents={}",
                len(context_documents),
            )

        context = "\n\n".join(
            document.page_content for document in context_documents
        )
        prompt = self.rag_prompt.format(
            context=context,
            question=query,
            phone=self.customer_service_phone,
        )
        return _PreparedRAGQuery(
            prompt=prompt,
            category=category,
            strategy=selected_strategy,
            documents=tuple(context_documents),
        )

    def generate_answer_with_trace(
        self,
        query: str,
        source_filter: str | None = None,
    ) -> RAGAnswer:
        """Generate one answer and retain the actual retrieval trace."""
        start = perf_counter()
        log.info("RAG query started: source_filter={}", source_filter)
        prepared = self._prepare_answer_prompt(query, source_filter)
        try:
            try:
                answer = self.llm(prepared.prompt)
            except Exception:
                log.exception("Final answer generation failed")
                answer = self._fallback_answer(prepared.category)
            return RAGAnswer(
                answer=answer,
                category=prepared.category,
                strategy=prepared.strategy,
                documents=prepared.documents,
            )
        finally:
            duration_ms = (perf_counter() - start) * 1000
            log.info("RAG query finished: duration_ms={:.3f}", duration_ms)

    def generate_answer(
        self,
        query: str,
        source_filter: str | None = None,
    ) -> str:
        return self.generate_answer_with_trace(
            query,
            source_filter,
        ).answer

    def generate_answer_stream(
        self,
        query: str,
        source_filter: str | None = None,
    ) -> Iterator[str]:
        """Yield the final answer while keeping internal RAG calls synchronous."""
        start = perf_counter()
        log.info("RAG streaming query started: source_filter={}", source_filter)
        try:
            prepared = self._prepare_answer_prompt(
                query,
                source_filter,
            )
            try:
                stream = getattr(self.llm, "stream", None)
                if callable(stream):
                    yield from stream(prepared.prompt)
                else:
                    yield self.llm(prepared.prompt)
            except Exception:
                log.exception("Streaming final answer generation failed")
                yield self._fallback_answer(prepared.category)
        finally:
            duration_ms = (perf_counter() - start) * 1000
            log.info(
                "RAG streaming query finished: duration_ms={:.3f}",
                duration_ms,
            )

    def _fallback_answer(self, category: str) -> str:
        category_name = (
            "general knowledge"
            if normalize_query_category(category) == GENERAL_KNOWLEDGE_CATEGORY
            else "professional consultation"
        )
        return (
            f"Sorry, we could not process your {category_name} question. "
            "Please contact customer service at "
            f"{self.customer_service_phone}."
        )


def main() -> None:
    """Build the local RAG workflow and start an interactive query loop."""
    config = load_config()

    model_path = Path(config.rag.query_model_path)
    should_train_classifier = not model_path.is_dir()
    query_classifier = QueryClassifier.from_config(config)
    if should_train_classifier:
        log.info(
            "Fine-tuned query classifier not found; training started: path={}",
            model_path,
        )
        query_classifier.train_model()

    knowledge_base_path = Path(config.rag.knowledge_base_path)
    documents = parse_document_from_dir(
        knowledge_base_path,
        config=config,
    )
    vector_store = VectorStore.from_config(config)
    vector_store.add_documents(documents)

    rag_system = RAGSystem.from_config(
        config,
        vector_store=vector_store,
        query_classifier=query_classifier,
    )
    source_filter = knowledge_base_path.name.removesuffix("_data")
    log.info(
        "RAG workflow ready: documents={}, source_filter={}",
        len(documents),
        source_filter,
    )

    while True:
        try:
            query = input("Query (type 'exit' to quit): ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            log.info("RAG workflow stopped")
            break

        if query.lower() in {"exit", "quit"}:
            log.info("RAG workflow stopped")
            break
        if not query:
            continue

        if config.llm.stream:
            for fragment in rag_system.generate_answer_stream(
                query,
                source_filter=source_filter,
            ):
                print(fragment, end="", flush=True)
            print()
        else:
            answer = rag_system.generate_answer(
                query,
                source_filter=source_filter,
            )
            print(answer)


if __name__ == "__main__":
    main()
