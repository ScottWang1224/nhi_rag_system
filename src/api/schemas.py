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
    request_id: str
    query: str
    answer: str
    references: list[ReferenceLink] = Field(default_factory=list)


class QueryLogReviewUpdate(BaseModel):
    human_rating: str | None = Field(default=None, max_length=32)
    human_note: str | None = Field(default=None, max_length=2000)


class QueryLogFeedbackUpdate(BaseModel):
    user_feedback: str = Field(..., pattern="^(helpful|not_helpful)$")
