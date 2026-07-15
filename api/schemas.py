from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


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

    @field_validator("question", "answer", "subject")
    @classmethod
    def strip_and_require_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("FAQ fields must not be blank")
        return value


class FAQUpdate(BaseModel):
    question: str | None = None
    answer: str | None = None
    subject: str | None = None

    @field_validator("question", "answer", "subject")
    @classmethod
    def strip_and_require_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("FAQ fields must not be blank")
        return value

    @model_validator(mode="after")
    def require_a_field(self) -> "FAQUpdate":
        if self.question is None and self.answer is None and self.subject is None:
            raise ValueError("at least one FAQ field must be supplied")
        return self


class FAQListResponse(BaseModel):
    items: list[dict[str, Any]] = Field(default_factory=list)
