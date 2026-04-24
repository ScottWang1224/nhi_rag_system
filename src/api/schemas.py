from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User query.")
    top_k: int | None = Field(default=None, ge=1, le=10)


class ReferenceLink(BaseModel):
    rank: int
    title: str
    url: str
    source: str | None = None


class ChatResponse(BaseModel):
    query: str
    answer: str
    references: list[ReferenceLink] = Field(default_factory=list)
