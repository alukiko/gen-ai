from __future__ import annotations

import json
import re
from statistics import mode

from pydantic import BaseModel, Field

from llm_client import structured_completion
from model import predict_engagement
from rag import retrieve_examples
from schema import (
    HallucinationReport,
    JudgeResult,
    OptimizationResult,
    RetrievedExample,
    TweetInput,
    TweetVariant,
)


class RewritePlan(BaseModel):
    variants: list[TweetVariant] = Field(min_length=1, max_length=4)


class JudgePayload(BaseModel):
    score: int = Field(ge=1, le=5)
    passes: bool
    reason: str


def _hashtags(text: str) -> list[str]:
    return re.findall(r"#[A-Za-z0-9_]+", text)


def _fallback_variants(tweet: TweetInput, examples: list[RetrievedExample]) -> list[TweetVariant]:
    common_day = mode([e.weekday for e in examples]) if examples else tweet.weekday
    best_hour = int(max(examples, key=lambda e: e.engagement_score).hour) if examples else tweet.hour
    base = re.sub(r"\s+", " ", tweet.text).strip()
    tags = _hashtags(base)
    if not tags:
        tags = ["#AI", "#Cloud", "#Tech"]
    cta = "What would you build with it?"
    variants = [
        TweetVariant(
            text=f"{base} {cta} {' '.join(tags[:3])}"[:280],
            weekday=common_day,  # type: ignore[arg-type]
            hour=best_hour,
            rationale="Adds a direct question and uses timing from similar historical posts.",
        ),
        TweetVariant(
            text=f"New update: {base} Join the discussion and share your take. {' '.join(tags[:2])}"[:280],
            weekday=tweet.weekday,
            hour=best_hour,
            rationale="Frames the post as news and adds a clear discussion prompt.",
        ),
        TweetVariant(
            text=f"{base} Save this for your next project and tell us what should come next. {' '.join(tags[:2])}"[:280],
            weekday=common_day,  # type: ignore[arg-type]
            hour=18 if best_hour < 8 else best_hour,
            rationale="Uses a practical CTA and avoids very low attention hours.",
        ),
    ]
    return variants


def _llm_variants(tweet: TweetInput, examples: list[RetrievedExample]) -> list[TweetVariant]:
    prompt = {
        "task": "Rewrite a tweet to improve predicted engagement.",
        "constraints": [
            "Return 3 variants.",
            "Each variant must be <= 280 characters.",
            "Use only Monday-Sunday weekday names and hour 0-23.",
            "Do not invent measured scores.",
        ],
        "input": tweet.model_dump(),
        "retrieved_examples": [e.model_dump() for e in examples],
    }
    plan = structured_completion(
        [
            {
                "role": "user",
                "content": json.dumps(prompt, ensure_ascii=False),
            }
        ],
        RewritePlan,
        max_retries=2,
        temperature=0.4,
    )
    return plan.variants


def _hallucination_report(result_text: str, allowed_texts: list[str], allowed_numbers: list[float | int]) -> HallucinationReport:
    numbers = re.findall(r"\b\d+(?:\.\d+)?\b", result_text)
    allowed = {str(int(n)) if float(n).is_integer() else str(n) for n in map(float, allowed_numbers)}
    ghost_numbers = sum(1 for n in numbers if n not in allowed and not (0 <= float(n) <= 280))
    ignored_json_keys = {"original", "best_variant", "scores", "best_time"}
    quotes = [q for q in re.findall(r'"([^"]{8,})"', result_text) if q not in ignored_json_keys]
    ghost_quotes = sum(1 for q in quotes if not any(q in text for text in allowed_texts))
    notes = []
    if ghost_numbers:
        notes.append("Some numeric claims were not produced by tools or validators.")
    if ghost_quotes:
        notes.append("Some quoted text was not found in the original, variants, or retrieved examples.")
    if not notes:
        notes.append("No unsupported numbers or quotes found.")
    return HallucinationReport(
        ghost_numbers=ghost_numbers,
        ghost_quotes=ghost_quotes,
        checked_numbers=numbers,
        checked_quotes=quotes,
        notes=notes,
    )


def _fallback_judge(original_score: float, best_score: float, hallucinations: HallucinationReport) -> JudgeResult:
    improvement = best_score - original_score
    passes = improvement >= 0 and hallucinations.ghost_numbers == 0 and hallucinations.ghost_quotes == 0
    score = 5 if improvement >= 1.0 and passes else 4 if passes else 3
    return JudgeResult(
        score=score,
        passes=passes,
        reason=f"Heuristic judge: score delta={improvement:.2f}, ghost numbers={hallucinations.ghost_numbers}, ghost quotes={hallucinations.ghost_quotes}.",
    )


def _llm_judge(payload: dict) -> JudgeResult:
    judged = structured_completion(
        [
            {
                "role": "user",
                "content": (
                    "Judge whether the tweet optimization is useful and grounded in the tool outputs. "
                    "Score 1-5. Return JSON only.\n"
                    + json.dumps(payload, ensure_ascii=False)
                ),
            }
        ],
        JudgePayload,
        max_retries=1,
        temperature=0.0,
    )
    return JudgeResult(**judged.model_dump())


def optimize_tweet(tweet: TweetInput, use_llm: bool = True) -> OptimizationResult:
    trace: list[dict] = []
    examples = retrieve_examples(tweet.text, k=4)
    original_prediction = predict_engagement(tweet.text, tweet.weekday, tweet.hour)
    trace.append({"tool": "predict_engagement", "input": tweet.model_dump(), "output": original_prediction.model_dump()})

    try:
        variants = _llm_variants(tweet, examples) if use_llm else _fallback_variants(tweet, examples)
    except Exception as exc:
        trace.append({"tool": "llm_rewriter", "error": str(exc), "fallback": "deterministic_variants"})
        variants = _fallback_variants(tweet, examples)
    variants.append(
        TweetVariant(
            text=tweet.text[:280],
            weekday=tweet.weekday,
            hour=tweet.hour,
            rationale="Baseline candidate: keep the original if rewrites reduce predicted engagement.",
        )
    )

    scored = []
    for variant in variants[:4]:
        prediction = predict_engagement(variant.text, variant.weekday, variant.hour)
        trace.append({"tool": "predict_engagement", "input": variant.model_dump(), "output": prediction.model_dump()})
        scored.append((variant, prediction))

    best_variant, best_prediction = max(scored, key=lambda item: item[1].engagement_score)
    result_text = (
        f"Original score {original_prediction.engagement_score}. "
        f"Best score {best_prediction.engagement_score}. "
        f"Best time {best_variant.weekday} {best_variant.hour}. "
        f"Original text: {tweet.text}. Best text: {best_variant.text}."
    )
    allowed_texts = [tweet.text, best_variant.text] + [v.text for v, _ in scored] + [e.text for e in examples]
    allowed_numbers = [
        original_prediction.engagement_score,
        best_prediction.engagement_score,
        tweet.hour,
        best_variant.hour,
        len(tweet.text),
        len(best_variant.text),
        280,
        10,
    ]
    hallucinations = _hallucination_report(result_text, allowed_texts, allowed_numbers)
    judge_payload = {
        "original": tweet.model_dump(),
        "original_prediction": original_prediction.model_dump(),
        "best_variant": best_variant.model_dump(),
        "best_prediction": best_prediction.model_dump(),
        "hallucination_report": hallucinations.model_dump(),
        "trace": trace,
    }
    try:
        judge = _llm_judge(judge_payload) if use_llm else _fallback_judge(
            original_prediction.engagement_score,
            best_prediction.engagement_score,
            hallucinations,
        )
    except Exception as exc:
        trace.append({"tool": "llm_judge", "error": str(exc), "fallback": "heuristic_judge"})
        judge = _fallback_judge(
            original_prediction.engagement_score,
            best_prediction.engagement_score,
            hallucinations,
        )

    return OptimizationResult(
        original=tweet,
        original_prediction=original_prediction,
        best_variant=best_variant,
        best_prediction=best_prediction,
        retrieved_examples=examples,
        tool_trace=trace,
        hallucination_report=hallucinations,
        judge=judge,
        techniques_used=[
            "RAG over historical tweets",
            "agent with engagement prediction tool",
            "structured output with Pydantic validators",
            "LLM-as-judge with heuristic fallback",
            "hallucination check for ghost numbers and quotes",
        ],
    )
