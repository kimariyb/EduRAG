from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, description="用户提问内容")
    session_id: str | None = Field(
        default=None, description="会话 ID；不传则服务端自动新建"
    )
    source_filter: str | None = Field(
        default=None, description="RAG 检索来源过滤（可选）"
    )


class AskResponse(BaseModel):
    session_id: str
    # 答案来源：sql（命中 FAQ）/ rag（知识库检索）/ mock（演示模式）
    source: str
    answer: str
    history: list[dict[str, Any]] = Field(default_factory=list)


class HistoryResponse(BaseModel):
    session_id: str
    history: list[dict[str, Any]] = Field(default_factory=list)


class ClearResponse(BaseModel):
    session_id: str
    cleared: bool


class FAQCreate(BaseModel):
    question: str = Field(..., min_length=1)
    answer: str = Field(..., min_length=1)
    subject: str | None = None


class FAQUpdate(BaseModel):
    question: str | None = None
    answer: str | None = None
    subject: str | None = None


class FAQListResponse(BaseModel):
    items: list[dict[str, Any]] = Field(default_factory=list)
