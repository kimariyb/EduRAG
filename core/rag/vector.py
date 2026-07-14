from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from langchain_core.documents import Document
from pymilvus import AnnSearchRequest, DataType, MilvusClient, WeightedRanker
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from sentence_transformers import CrossEncoder

from base.config import AppConfig
from base.logger import logger
from core.rag.constants import (
    DEFAULT_CANDIDATE_M,
    DEFAULT_EMBEDDING_MODEL_PATH,
    DEFAULT_MODEL_DEVICE,
    DEFAULT_RERANKER_MODEL_PATH,
    DEFAULT_RETRIEVAL_K,
    DENSE_SEARCH_WEIGHT,
    SPARSE_SEARCH_WEIGHT,
)


log = logger.bind(module=__name__)
OUTPUT_FIELDS = (
    "text",
    "parent_id",
    "parent_content",
    "source",
    "timestamp",
)


class VectorStore:
    def __init__(
        self,
        collection: str,
        host: str,
        port: int | str,
        database: str,
        *,
        client: Any | None = None,
        embedding_function: Any | None = None,
        reranker: Any | None = None,
        candidate_m: int = DEFAULT_CANDIDATE_M,
        auto_prepare: bool = True,
        embedding_model_path: str = DEFAULT_EMBEDDING_MODEL_PATH,
        reranker_model_path: str = DEFAULT_RERANKER_MODEL_PATH,
        model_device: str = DEFAULT_MODEL_DEVICE,
    ) -> None:
        if candidate_m <= 0:
            raise ValueError("candidate_m must be greater than 0")

        self.collection = collection
        self.collection_name = collection
        self.host = host
        self.port = port
        self.database = database
        self.candidate_m = candidate_m
        self.embedding_function = embedding_function or BGEM3EmbeddingFunction(
            model_name=embedding_model_path,
            use_fp16=False,
            device=model_device,
        )
        self.embedding = self.embedding_function
        self.reranker = reranker or CrossEncoder(
            reranker_model_path,
            device=model_device,
        )
        self.dense_dim = int(self.embedding_function.dim["dense"])
        self.client = client or MilvusClient(
            uri=self._build_uri(host, port),
            db_name=database,
        )

        if auto_prepare:
            self._create_or_load_collection()
        log.info(
            "Vector store initialized: collection={}, database={}",
            self.collection_name,
            self.database,
        )

    @classmethod
    def from_config(
        cls,
        config: AppConfig,
        *,
        client: Any | None = None,
        embedding_function: Any | None = None,
        reranker: Any | None = None,
        auto_prepare: bool = True,
    ) -> "VectorStore":
        return cls(
            collection=config.milvus.collection,
            host=config.milvus.host,
            port=config.milvus.port,
            database=config.milvus.database,
            client=client,
            embedding_function=embedding_function,
            reranker=reranker,
            candidate_m=config.rag.candidate_m,
            auto_prepare=auto_prepare,
            embedding_model_path=config.rag.embedding_model_path,
            reranker_model_path=config.rag.reranker_model_path,
            model_device=config.rag.model_device,
        )

    @staticmethod
    def _build_uri(host: str, port: int | str) -> str:
        if host.startswith(("http://", "https://")):
            return f"{host.rstrip('/')}:{port}"
        return f"http://{host}:{port}"

    def _create_or_load_collection(self) -> None:
        if not self.client.has_collection(self.collection_name):
            schema = self.client.create_schema(
                auto_id=False,
                enable_dynamic_field=True,
            )
            schema.add_field(
                field_name="id",
                datatype=DataType.VARCHAR,
                is_primary=True,
                max_length=100,
            )
            schema.add_field(
                field_name="text",
                datatype=DataType.VARCHAR,
                max_length=65535,
            )
            schema.add_field(
                field_name="dense_vector",
                datatype=DataType.FLOAT_VECTOR,
                dim=self.dense_dim,
            )
            schema.add_field(
                field_name="sparse_vector",
                datatype=DataType.SPARSE_FLOAT_VECTOR,
            )
            schema.add_field(
                field_name="parent_id",
                datatype=DataType.VARCHAR,
                max_length=100,
            )
            schema.add_field(
                field_name="parent_content",
                datatype=DataType.VARCHAR,
                max_length=65535,
            )
            schema.add_field(
                field_name="source",
                datatype=DataType.VARCHAR,
                max_length=50,
            )
            schema.add_field(
                field_name="timestamp",
                datatype=DataType.VARCHAR,
                max_length=50,
            )

            index_params = self.client.prepare_index_params()
            index_params.add_index(
                field_name="dense_vector",
                index_name="dense_index",
                index_type="IVF_FLAT",
                metric_type="IP",
                params={"nlist": 128},
            )
            index_params.add_index(
                field_name="sparse_vector",
                index_name="sparse_index",
                index_type="SPARSE_INVERTED_INDEX",
                metric_type="IP",
                params={"drop_ratio_build": 0.2},
            )
            self.client.create_collection(
                collection_name=self.collection_name,
                schema=schema,
                index_params=index_params,
            )
            log.info("Created Milvus collection: collection={}", self.collection_name)
        else:
            log.info("Using existing Milvus collection: collection={}", self.collection_name)

        self.client.load_collection(collection_name=self.collection_name)
        log.info("Loaded Milvus collection: collection={}", self.collection_name)

    def add_documents(self, documents: Sequence[Document]) -> None:
        if not documents:
            log.warning("Skipped vector upsert because document list is empty")
            return

        texts = [document.page_content for document in documents]
        embeddings = self.embedding_function(texts)
        data: list[dict[str, Any]] = []
        for index, document in enumerate(documents):
            parent_id = document.metadata.get("parent_id")
            parent_content = document.metadata.get("parent_content")
            if not parent_id or parent_content is None:
                raise ValueError(
                    "document metadata must include parent_id and parent_content"
                )

            data.append(
                {
                    "id": self._document_id(document),
                    "text": document.page_content,
                    "dense_vector": self._dense_vector(
                        embeddings["dense"][index]
                    ),
                    "sparse_vector": self._sparse_vector(
                        embeddings["sparse"][index : index + 1]
                    ),
                    "parent_id": str(parent_id),
                    "parent_content": str(parent_content),
                    "source": str(document.metadata.get("source", "unknown")),
                    "timestamp": str(
                        document.metadata.get("timestamp", "unknown")
                    ),
                }
            )

        self.client.upsert(collection_name=self.collection_name, data=data)
        log.info(
            "Upserted documents into Milvus: collection={}, count={}",
            self.collection_name,
            len(data),
        )

    def hybrid_search_with_rerank(
        self,
        query: str,
        k: int = DEFAULT_RETRIEVAL_K,
        source_filter: str | None = None,
    ) -> list[Document]:
        if k <= 0:
            return []

        query_embeddings = self.embedding_function([query])
        dense_query_vector = self._dense_vector(query_embeddings["dense"][0])
        sparse_query_vector = self._sparse_vector(
            query_embeddings["sparse"][0:1]
        )
        filter_expression = self._source_filter_expression(source_filter)

        dense_request = AnnSearchRequest(
            data=[dense_query_vector],
            anns_field="dense_vector",
            param={"metric_type": "IP", "params": {"nprobe": 10}},
            limit=k,
            expr=filter_expression,
        )
        sparse_request = AnnSearchRequest(
            data=[sparse_query_vector],
            anns_field="sparse_vector",
            param={"metric_type": "IP", "params": {}},
            limit=k,
            expr=filter_expression,
        )
        search_results = self.client.hybrid_search(
            collection_name=self.collection_name,
            reqs=[dense_request, sparse_request],
            ranker=WeightedRanker(
                DENSE_SEARCH_WEIGHT,
                SPARSE_SEARCH_WEIGHT,
            ),
            limit=k,
            output_fields=list(OUTPUT_FIELDS),
        )
        results = search_results[0] if search_results else []

        sub_chunks = [
            self._doc_from_hit(hit.get("entity", {}))
            for hit in results
        ]
        parent_documents = self._get_unique_parent_docs(sub_chunks)
        log.info(
            "Milvus hybrid search completed: hits={}, parents={}",
            len(sub_chunks),
            len(parent_documents),
        )
        if len(parent_documents) < 2:
            return parent_documents[: self.candidate_m]

        pairs = [[query, document.page_content] for document in parent_documents]
        scores = self.reranker.predict(pairs)
        ranked_documents = [
            document
            for _, document in sorted(
                zip(scores, parent_documents),
                key=lambda item: float(item[0]),
                reverse=True,
            )
        ]
        log.info("Reranked parent documents: count={}", len(ranked_documents))
        return ranked_documents[: self.candidate_m]

    @staticmethod
    def _dense_vector(vector: Any) -> list[float]:
        values = vector.tolist() if hasattr(vector, "tolist") else vector
        return [float(value) for value in values]

    @staticmethod
    def _document_id(document: Document) -> str:
        identity = json.dumps(
            [
                document.metadata.get("source"),
                document.metadata.get("id"),
                document.metadata.get("parent_id"),
                document.page_content,
            ],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return hashlib.md5(
            identity.encode("utf-8"),
            usedforsecurity=False,
        ).hexdigest()

    @staticmethod
    def _sparse_vector(row: Any) -> dict[int, float]:
        return {
            int(index): float(value)
            for index, value in zip(row.indices, row.data)
        }

    @staticmethod
    def _source_filter_expression(source_filter: str | None) -> str:
        if not source_filter:
            return ""
        return f"source == {json.dumps(source_filter, ensure_ascii=False)}"

    @staticmethod
    def _get_unique_parent_docs(
        sub_chunks: Sequence[Document],
    ) -> list[Document]:
        parent_contents: set[str] = set()
        unique_documents: list[Document] = []
        for chunk in sub_chunks:
            parent_content = str(
                chunk.metadata.get(
                    "parent_content",
                    chunk.page_content,
                )
            )
            if not parent_content or parent_content in parent_contents:
                continue
            unique_documents.append(
                Document(
                    page_content=parent_content,
                    metadata=dict(chunk.metadata),
                )
            )
            parent_contents.add(parent_content)
        return unique_documents

    @staticmethod
    def _doc_from_hit(hit: Mapping[str, Any]) -> Document:
        return Document(
            page_content=str(hit.get("text") or ""),
            metadata={
                "parent_id": hit.get("parent_id"),
                "parent_content": hit.get("parent_content"),
                "source": hit.get("source"),
                "timestamp": hit.get("timestamp"),
            },
        )
