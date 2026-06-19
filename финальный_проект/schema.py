from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


Weekday = Literal[
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]


class TweetInput(BaseModel):
    text: str = Field(min_length=5, max_length=500)
    weekday: Weekday
    hour: int = Field(ge=0, le=23)


class TweetVariant(BaseModel):
    text: str = Field(min_length=5, max_length=280)
    weekday: Weekday
    hour: int = Field(ge=0, le=23)
    rationale: str = Field(min_length=5, max_length=500)

    @field_validator("text")
    @classmethod
    def tweet_must_fit_x_limit(cls, value: str) -> str:
        if len(value) > 280:
            raise ValueError("Tweet text must fit the 280 character X/Twitter limit")
        if "\n\n" in value:
            raise ValueError("Tweet text must be one compact post, not a thread")
        return value.strip()


class Prediction(BaseModel):
    engagement_score: float = Field(ge=0, le=10)
    weekday: Weekday
    hour: int = Field(ge=0, le=23)
    text_length: int = Field(ge=0, le=280)


class RetrievedExample(BaseModel):
    tweet_id: str
    text: str
    weekday: str
    hour: int
    engagement_score: float
    similarity: float


class HallucinationReport(BaseModel):
    ghost_numbers: int = Field(ge=0)
    ghost_quotes: int = Field(ge=0)
    checked_numbers: list[str]
    checked_quotes: list[str]
    notes: list[str]


class JudgeResult(BaseModel):
    score: int = Field(ge=1, le=5)
    passes: bool
    reason: str


class OptimizationResult(BaseModel):
    original: TweetInput
    original_prediction: Prediction
    best_variant: TweetVariant
    best_prediction: Prediction
    retrieved_examples: list[RetrievedExample]
    tool_trace: list[dict]
    hallucination_report: HallucinationReport
    judge: JudgeResult
    techniques_used: list[str]

