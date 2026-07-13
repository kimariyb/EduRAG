from __future__ import annotations

import os
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters.markdown import MarkdownTextSplitter

from base.config import AppConfig
from base.logger import logger
from core.rag.constants import (
    DEFAULT_CHILD_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_PARENT_CHUNK_SIZE,
)
from core.rag.utils.document_loaders import (
    DOCXLoader,
    IMGLoader,
    PDFLoader,
    PlainTextLoader,
    PPTXLoader,
)
from core.rag.utils.text_splitters import ChineseRecursiveTextSplitter


log = logger.bind(module=__name__)

DOCUMENT_LOADERS: dict[str, Any] = {
    ".txt": PlainTextLoader,
    ".pdf": PDFLoader,
    ".docx": DOCXLoader,
    ".pptx": PPTXLoader,
    ".jpg": IMGLoader,
    ".png": IMGLoader,
    ".md": PlainTextLoader,
}

# Preserve the original public name used by existing callers.
document_loaders = DOCUMENT_LOADERS


def load_document_from_dir(
    dir_path: str | Path,
    *,
    loader_registry: Mapping[str, Any] | None = None,
) -> list[Document]:
    directory = Path(dir_path)
    if not directory.is_dir():
        raise NotADirectoryError(f"document directory not found: {directory}")

    registry = loader_registry or DOCUMENT_LOADERS
    source = directory.name.removesuffix("_data")
    documents: list[Document] = []
    log.info("Document directory loading started: path={}", directory)

    for root, directories, filenames in os.walk(directory):
        directories.sort()
        for filename in sorted(filenames):
            file_path = Path(root) / filename
            extension = file_path.suffix.lower()
            loader_class = registry.get(extension)
            if loader_class is None:
                log.warning("Unsupported document type skipped: path={}", file_path)
                continue

            try:
                loader = (
                    loader_class(str(file_path), encoding="utf-8")
                    if extension in {".txt", ".md"}
                    else loader_class(str(file_path))
                )
                loaded_documents = loader.load()
                timestamp = datetime.now(timezone.utc).isoformat()
                for document in loaded_documents:
                    document.metadata.update(
                        {
                            "source": source,
                            "file_path": str(file_path),
                            "timestamp": timestamp,
                        }
                    )
                documents.extend(loaded_documents)
                log.info(
                    "Document loaded: path={}, documents={}",
                    file_path,
                    len(loaded_documents),
                )
            except Exception:
                log.exception("Document loading failed: path={}", file_path)

    log.info(
        "Document directory loading completed: path={}, documents={}",
        directory,
        len(documents),
    )
    return documents


def parse_document_from_dir(
    dir_path: str | Path,
    parent_chunk_size: int | None = None,
    child_chunk_size: int | None = None,
    chunk_overlap: int | float | None = None,
    *,
    config: AppConfig | None = None,
) -> list[Document]:
    parent_chunk_size = parent_chunk_size or (
        config.rag.parent_chunk_size if config else DEFAULT_PARENT_CHUNK_SIZE
    )
    child_chunk_size = child_chunk_size or (
        config.rag.child_chunk_size if config else DEFAULT_CHILD_CHUNK_SIZE
    )
    if chunk_overlap is None:
        chunk_overlap = (
            config.rag.chunk_overlap if config else DEFAULT_CHUNK_OVERLAP
        )
    parent_overlap = _resolve_chunk_overlap(parent_chunk_size, chunk_overlap)
    child_overlap = _resolve_chunk_overlap(child_chunk_size, chunk_overlap)
    documents = load_document_from_dir(dir_path)
    log.info("Document parsing started: documents={}", len(documents))

    parent_splitter = ChineseRecursiveTextSplitter(
        chunk_size=parent_chunk_size,
        chunk_overlap=parent_overlap,
    )
    child_splitter = ChineseRecursiveTextSplitter(
        chunk_size=child_chunk_size,
        chunk_overlap=child_overlap,
    )
    markdown_parent_splitter = MarkdownTextSplitter(
        chunk_size=parent_chunk_size,
        chunk_overlap=parent_overlap,
    )
    markdown_child_splitter = MarkdownTextSplitter(
        chunk_size=child_chunk_size,
        chunk_overlap=child_overlap,
    )

    child_chunks: list[Document] = []
    for document_index, document in enumerate(documents):
        file_path = str(document.metadata.get("file_path", ""))
        is_markdown = Path(file_path).suffix.lower() == ".md"
        active_parent_splitter = (
            markdown_parent_splitter if is_markdown else parent_splitter
        )
        active_child_splitter = (
            markdown_child_splitter if is_markdown else child_splitter
        )
        log.info(
            "Splitting document: path={}, splitter={}",
            file_path,
            "markdown" if is_markdown else "chinese_recursive",
        )

        parent_documents = active_parent_splitter.split_documents([document])
        for parent_index, parent_document in enumerate(parent_documents):
            parent_id = f"doc_{document_index}_parent_{parent_index}"
            parent_document.metadata.update(
                {
                    "parent_id": parent_id,
                    "parent_content": parent_document.page_content,
                }
            )
            sub_chunks = active_child_splitter.split_documents(
                [parent_document]
            )
            for child_index, child_chunk in enumerate(sub_chunks):
                child_chunk.metadata.update(
                    {
                        "parent_id": parent_id,
                        "parent_content": parent_document.page_content,
                        "id": f"{parent_id}_child_{child_index}",
                    }
                )
                child_chunks.append(child_chunk)

    log.info("Document parsing completed: child_chunks={}", len(child_chunks))
    return child_chunks


def _resolve_chunk_overlap(
    chunk_size: int,
    chunk_overlap: int | float,
) -> int:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative")

    overlap = (
        int(chunk_size * chunk_overlap)
        if isinstance(chunk_overlap, float) and chunk_overlap < 1
        else int(chunk_overlap)
    )
    if overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    return overlap
