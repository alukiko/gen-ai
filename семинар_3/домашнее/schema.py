from __future__ import annotations

from datetime import date, datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


Aspect = Literal["performance", "design", "support", "price", "ads", "reliability"]
Sentiment = Literal["positive", "negative", "neutral"]


class Issue(BaseModel):
    category: Aspect
    severity: int = Field(ge=1, le=5)
    quote: str = Field(min_length=5)


class Review(BaseModel):
    review_id: str
    author: str
    rating: int = Field(ge=1, le=5)
    date: date
    app_version: Optional[str] = None
    text: str = Field(min_length=10)
    issues: list[Issue] = Field(default_factory=list)

    @field_validator("date", mode="before")
    @classmethod
    def parse_and_check_date(cls, value: object) -> date:
        if isinstance(value, date):
            parsed = value
        elif isinstance(value, str):
            raw = value.strip()
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00")).date()
        else:
            raise ValueError("date must be an ISO date string")
        if parsed > date.today():
            raise ValueError("review date cannot be in the future")
        return parsed


class AspectSentiment(BaseModel):
    aspect: Aspect
    sentiment: Sentiment
    quote: str = Field(min_length=5)
    confidence: float = Field(ge=0, le=1)


class ReviewSentiment(BaseModel):
    review_id: str
    author: str
    rating: int = Field(ge=1, le=5)
    aspects: list[AspectSentiment] = Field(default_factory=list)


class ChunkSummary(BaseModel):
    chunk_id: int
    review_ids: list[str] = Field(min_length=1)
    key_points: list[str] = Field(min_length=2, max_length=8)
    recurring_issues: list[str] = Field(min_length=1, max_length=8)
    sentiment: Literal["positive", "negative", "mixed"]
    evidence_quotes: list[str] = Field(min_length=1, max_length=8)


class ReviewSummary(BaseModel):
    headline: str
    key_findings: list[str] = Field(min_length=3, max_length=8)
    action_items: list[str] = Field(min_length=2, max_length=6)
    risks: list[str] = Field(default_factory=list, max_length=6)


class ActionVerdict(BaseModel):
    action: str
    support: Literal["supported", "weakly_supported", "not_supported"]
    evidence: list[str] = Field(default_factory=list)
    comment: str


class JudgeReport(BaseModel):
    verdicts: list[ActionVerdict]
    overall_score: float = Field(ge=0, le=1)
    summary: str
