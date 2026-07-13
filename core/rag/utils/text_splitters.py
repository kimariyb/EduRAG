from __future__ import annotations

import re
from typing import Any

from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)
from typing_extensions import Literal

from base.config import AppConfig
from base.logger import logger
from core.rag.constants import DEFAULT_SEGMENTER_DEVICE


log = logger.bind(module=__name__)

MODEL_PATH = {
    "segment_model": {
        "ali_model": "iic/nlp_bert_document-segmentation_chinese-base",
    }
}

# Order separators from coarse to fine to preserve paragraph and sentence boundaries.
DEFAULT_CHINESE_SEPARATORS = (
    "\n\n",
    "\n",
    r"[。！？]",
    r"[.!?]\s",
    r"(?:；|;\s)",
    r"(?:，|,\s)",
    "",
)


def _split_text_with_regex_from_end(
    text: str,
    separator: str,
    keep_separator: bool | Literal["start", "end"],
) -> list[str]:
    if not separator:
        return list(text)
    if not keep_separator:
        return [part for part in re.split(separator, text) if part]

    split_parts = re.split(f"({separator})", text)
    # True keeps punctuation at the end; "start" moves it to the next segment.
    if keep_separator == "start":
        splits = [
            "".join(pair)
            for pair in zip(split_parts[1::2], split_parts[2::2])
        ]
        if len(split_parts) % 2 == 0:
            splits.append(split_parts[-1])
        splits.insert(0, split_parts[0])
    else:
        splits = [
            "".join(pair)
            for pair in zip(split_parts[0::2], split_parts[1::2])
        ]
        if len(split_parts) % 2 == 1:
            splits.append(split_parts[-1])
    return [part for part in splits if part]


class ChineseRecursiveTextSplitter(RecursiveCharacterTextSplitter):
    def __init__(
        self,
        separators: list[str] | None = None,
        keep_separator: bool | Literal["start", "end"] = True,
        is_separator_regex: bool = True,
        **kwargs: Any,
    ) -> None:
        active_separators = separators or list(DEFAULT_CHINESE_SEPARATORS)
        super().__init__(
            separators=active_separators,
            keep_separator=keep_separator,
            is_separator_regex=is_separator_regex,
            **kwargs,
        )

    def _split_text(self, text: str, separators: list[str]) -> list[str]:
        final_chunks: list[str] = []
        separator = separators[-1]
        new_separators: list[str] = []

        # Select the first matching level, then recurse with finer separators.
        for index, candidate in enumerate(separators):
            pattern = (
                candidate
                if self._is_separator_regex
                else re.escape(candidate)
            )
            if not candidate:
                separator = candidate
                break
            if re.search(pattern, text):
                separator = candidate
                new_separators = separators[index + 1 :]
                break

        pattern = separator if self._is_separator_regex else re.escape(separator)
        splits = _split_text_with_regex_from_end(
            text,
            pattern,
            self._keep_separator,
        )

        good_splits: list[str] = []
        # A regex is a matching rule and must not be reinserted as plain text.
        merge_separator = (
            ""
            if self._keep_separator or self._is_separator_regex
            else separator
        )
        for split in splits:
            if self._length_function(split) < self._chunk_size:
                good_splits.append(split)
                continue

            if good_splits:
                final_chunks.extend(
                    self._merge_splits(good_splits, merge_separator)
                )
                good_splits = []
            if new_separators:
                final_chunks.extend(self._split_text(split, new_separators))
            else:
                final_chunks.append(split)

        if good_splits:
            final_chunks.extend(
                self._merge_splits(good_splits, merge_separator)
            )

        return [
            re.sub(r"\n{2,}", "\n", chunk.strip())
            for chunk in final_chunks
            if chunk.strip()
        ]


class AliTextSplitter(CharacterTextSplitter):
    def __init__(
        self,
        separator: str = "\n\n",
        is_separator_regex: bool = False,
        pdf: bool = False,
        device: str = DEFAULT_SEGMENTER_DEVICE,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            separator=separator,
            is_separator_regex=is_separator_regex,
            **kwargs,
        )
        self.pdf = pdf
        try:
            from modelscope.pipelines import pipeline

            self.pipeline = pipeline(
                task="document-segmentation",
                model=MODEL_PATH["segment_model"]["ali_model"],
                device=device,
            )
        except Exception as exc:
            log.warning(
                "Failed to initialize ModelScope document segmentation: {}",
                exc,
            )
            self.pipeline = None

    @classmethod
    def from_config(
        cls,
        config: AppConfig,
        **kwargs: Any,
    ) -> "AliTextSplitter":
        return cls(device=config.rag.segmenter_device, **kwargs)

    def split_text(self, text: str) -> list[str]:
        if self.pdf:
            # Normalize excess whitespace without destroying PDF paragraphs.
            text = re.sub(r"\n{3,}", "\n\n", text)
            text = re.sub(r"[^\S\r\n]+", " ", text)

        if self.pipeline is None:
            log.warning(
                "ModelScope pipeline is unavailable; using character splitting"
            )
            return super().split_text(text)

        try:
            result = self.pipeline(documents=text)
            # ModelScope separates document segments with a newline and tab.
            return [
                segment.strip()
                for segment in result["text"].split("\n\t")
                if segment.strip()
            ]
        except Exception:
            log.exception(
                "ModelScope document segmentation failed; using character splitting"
            )
            return super().split_text(text)
