from __future__ import annotations

import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, TypeVar

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pydantic import ValidationError

from llm_client import get_model, make_client
from prompts import ASPECTS_SYSTEM, CHUNK_SYSTEM, IE_SYSTEM, JUDGE_SYSTEM, REDUCE_SYSTEM
from schema import ChunkSummary, JudgeReport, Review, ReviewSentiment, ReviewSummary

MODEL = get_model()
ASPECTS = ["performance", "design", "support", "price", "ads", "reliability"]
T = TypeVar("T")


class UsageTracker:
    def __init__(self) -> None:
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.prompt_cache_hit_tokens = 0

    def add(self, completion: Any) -> None:
        self.calls += 1
        usage = getattr(completion, "usage", None)
        if usage is None:
            return
        self.prompt_tokens += int(getattr(usage, "prompt_tokens", 0) or 0)
        self.completion_tokens += int(getattr(usage, "completion_tokens", 0) or 0)
        self.total_tokens += int(getattr(usage, "total_tokens", 0) or 0)
        details = getattr(usage, "prompt_tokens_details", None)
        cached = getattr(details, "cached_tokens", 0) if details is not None else 0
        self.prompt_cache_hit_tokens += int(cached or 0)

    def cost_usd(self) -> float:
        in_price = float(os.environ.get("LLM_INPUT_PRICE_PER_1M", "0"))
        out_price = float(os.environ.get("LLM_OUTPUT_PRICE_PER_1M", "0"))
        return (
            self.prompt_tokens / 1_000_000 * in_price
            + self.completion_tokens / 1_000_000 * out_price
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "prompt_cache_hit_tokens": self.prompt_cache_hit_tokens,
            "cost_usd_estimate": round(self.cost_usd(), 6),
            "cost_note": (
                "Set LLM_INPUT_PRICE_PER_1M and LLM_OUTPUT_PRICE_PER_1M "
                "in .env for a non-zero cost estimate."
            ),
        }


client = make_client()
usage = UsageTracker()


def _call_model(response_model: type[T], system: str, user: str) -> T:
    result, completion = client.chat.completions.create(
        model=MODEL,
        response_model=response_model,
        max_retries=3,
        temperature=0.0,
        with_completion=True,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    usage.add(completion)
    return result


def extract_reviews(source_text: str) -> tuple[list[Review], int]:
    reviews_by_id: dict[str, Review] = {}
    validation_errors = 0
    for chunk in split_review_blocks(source_text, size=5):
        expected = len(re.findall(r"^### REVIEW\s+\d+", chunk, flags=re.MULTILINE))
        try:
            chunk_reviews = _call_model(list[Review], IE_SYSTEM, chunk)
        except ValidationError:
            validation_errors += expected
            continue
        for review in chunk_reviews:
            reviews_by_id[review.review_id] = review
        validation_errors += max(0, expected - len(chunk_reviews))
    return list(reviews_by_id.values()), validation_errors


def extract_aspects(source_text: str) -> list[ReviewSentiment]:
    sentiments_by_id: dict[str, ReviewSentiment] = {}
    for chunk in split_review_blocks(source_text, size=5):
        chunk_sentiments = _call_model(list[ReviewSentiment], ASPECTS_SYSTEM, chunk)
        for sentiment in chunk_sentiments:
            sentiments_by_id[sentiment.review_id] = sentiment
    return list(sentiments_by_id.values())


def split_review_blocks(source_text: str, size: int = 8) -> list[str]:
    blocks = re.split(r"(?=^### REVIEW\s+\d+)", source_text, flags=re.MULTILINE)
    blocks = [b.strip() for b in blocks if b.strip().startswith("### REVIEW")]
    if not blocks:
        return [source_text]
    return ["\n\n".join(blocks[i : i + size]) for i in range(0, len(blocks), size)]


def summarize_chunk(chunk_id: int, chunk: str) -> ChunkSummary:
    return _call_model(
        ChunkSummary,
        CHUNK_SYSTEM,
        f"chunk_id: {chunk_id}\n\n{chunk}",
    )


def reduce_summaries(summaries: list[ChunkSummary], extra_instruction: str = "") -> ReviewSummary:
    payload = "\n\n".join(s.model_dump_json(indent=2) for s in summaries)
    system = REDUCE_SYSTEM if not extra_instruction else REDUCE_SYSTEM + "\n\n" + extra_instruction
    return _call_model(ReviewSummary, system, payload)


def summarize_reviews(source_text: str, workers: int = 5) -> tuple[list[ChunkSummary], ReviewSummary]:
    chunks = split_review_blocks(source_text)
    chunk_summaries: list[ChunkSummary | None] = [None] * len(chunks)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(summarize_chunk, i + 1, chunk): i
            for i, chunk in enumerate(chunks)
        }
        for future in as_completed(futures):
            idx = futures[future]
            chunk_summaries[idx] = future.result()
    ready = [s for s in chunk_summaries if s is not None]
    return ready, reduce_summaries(ready)


def build_evidence_packet(reviews: list[Review], summary: ReviewSummary) -> str:
    lines = ["## Action items"]
    for i, item in enumerate(summary.action_items, 1):
        lines.append(f"{i}. {item}")
    lines.append("\n## Review issues and exact quotes")
    for review in reviews:
        for issue in review.issues:
            lines.append(
                f"- [{review.review_id}/{issue.category}/sev={issue.severity}] "
                f"rating={review.rating}; quote: «{issue.quote}»"
            )
    return "\n".join(lines)


def judge(reviews: list[Review], summary: ReviewSummary) -> JudgeReport:
    return _call_model(JudgeReport, JUDGE_SYSTEM, build_evidence_packet(reviews, summary))


def _normalize_for_quote_match(value: str) -> str:
    value = value.lower().replace("ё", "е")
    value = re.sub(r"[^0-9a-zа-я]+", " ", value)
    return " ".join(value.split())


def _quote_exists(quote: str, source_text: str) -> bool:
    normalized = _normalize_for_quote_match(quote)
    haystack = _normalize_for_quote_match(source_text)
    if not normalized:
        return True
    if normalized in haystack:
        return True
    if len(normalized) <= 30:
        return normalized in haystack
    probe = normalized[:35]
    if probe in haystack:
        return True
    words = normalized.split()
    for i in range(max(1, len(words) - 4)):
        if " ".join(words[i : i + 5]) in haystack:
            return True
    return False


def check_review_quotes(reviews: list[Review], source_text: str) -> list[dict[str, str]]:
    ghosts: list[dict[str, str]] = []
    for review in reviews:
        for issue in review.issues:
            if not _quote_exists(issue.quote, source_text):
                ghosts.append(
                    {
                        "review_id": review.review_id,
                        "kind": "issue",
                        "quote": issue.quote,
                    }
                )
    return ghosts


def check_aspect_quotes(aspects: list[ReviewSentiment], source_text: str) -> list[dict[str, str]]:
    ghosts: list[dict[str, str]] = []
    for review in aspects:
        for aspect in review.aspects:
            if not _quote_exists(aspect.quote, source_text):
                ghosts.append(
                    {
                        "review_id": review.review_id,
                        "kind": f"aspect:{aspect.aspect}",
                        "quote": aspect.quote,
                    }
                )
    return ghosts


def build_heatmap(aspects: list[ReviewSentiment], out_path: Path) -> None:
    sent_to_num = {"positive": 1, "negative": -1, "neutral": 0}
    labels = [f"{r.review_id[:8]} ({r.rating})" for r in aspects]
    matrix = np.full((len(aspects), len(ASPECTS)), np.nan)
    for row_idx, review in enumerate(aspects):
        for item in review.aspects:
            matrix[row_idx, ASPECTS.index(item.aspect)] = sent_to_num[item.sentiment]

    height = max(6, len(aspects) * 0.28)
    plt.figure(figsize=(10, height))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".0f",
        xticklabels=ASPECTS,
        yticklabels=labels,
        center=0,
        cmap="RdYlGn",
        cbar_kws={"label": "sentiment"},
    )
    plt.title("Ozon review aspect sentiment")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_conclusions(
    out_dir: Path,
    input_count: int,
    structured_count: int,
    validation_errors: int,
    ghosts: list[dict[str, str]],
    report: JudgeReport,
    elapsed: float,
    metrics: dict[str, Any],
) -> None:
    weak = next(
        (
            v
            for v in report.verdicts
            if v.support in {"weakly_supported", "not_supported"}
        ),
        report.verdicts[0] if report.verdicts else None,
    )
    ghost_examples = ghosts[:3]
    ghost_text = "\n".join(
        f"- `{g['review_id']}`: модель сослалась на цитату «{g['quote'][:160]}», "
        "но sanity-check не нашел ее в исходных отзывах."
        for g in ghost_examples
    ) or "- Ghost-цитаты не найдены; явных примеров искажения по цитатам нет."
    judge_example = (
        f"- Judge отметил `{weak.action}` как `{weak.support}`: {weak.comment}"
        if weak
        else "- Judge не вернул отдельных вердиктов."
    )

    text = f"""# Выводы по ДЗ семинара 3

## 1. Что получилось

- Обработано входных отзывов: {input_count}.
- Валидных структурированных отзывов: {structured_count}; ValidationError/не извлечено: {validation_errors}.
- Ghost-цитат найдено sanity-check: {len(ghosts)}.
- Overall score от judge: {report.overall_score:.2f}.
- Время полного прогона: {elapsed:.1f} с.
- LLM-вызовы: {metrics['usage']['calls']}; total_tokens: {metrics['usage']['total_tokens']}; оценка стоимости: ${metrics['usage']['cost_usd_estimate']:.6f}.

## 2. Где модель ошибалась

{ghost_text}

{judge_example}

Основной риск пайплайна - модель иногда обобщает проблемы доставки, оплаты и
возвратов слишком широко. Поэтому action_items дополнительно проверяются judge
по пакету цитат из IE.

## 3. Что бы изменили в production

- Оставил бы связку IE -> aspect analysis -> judge: она дает понятный аудит
  того, из каких отзывов выросли рекомендации.
- Переделал бы сбор данных: добавил бы стабильный API/хранилище отзывов,
  дедупликацию авторов и стратификацию по rating/app_version.
- Добавил бы ручную проверку спорных цитат, A/B двух моделей для judge и
  реальные цены модели через `LLM_INPUT_PRICE_PER_1M` / `LLM_OUTPUT_PRICE_PER_1M`.
"""
    (out_dir.parent / "выводы.md").write_text(text, encoding="utf-8")


def analyze(input_path: str, out_dir: str = "output") -> None:
    started = time.time()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    source_text = Path(input_path).read_text(encoding="utf-8")
    input_count = len(re.findall(r"^### REVIEW\s+\d+", source_text, flags=re.MULTILINE))

    print("1/4 IE: extracting reviews and issues...")
    validation_errors = 0
    reviews, validation_errors = extract_reviews(source_text)
    save_json(out / "reviews.json", [r.model_dump(mode="json") for r in reviews])

    print("2/4 Aspect analysis...")
    aspects = extract_aspects(source_text)
    save_json(out / "aspects.json", [a.model_dump() for a in aspects])
    aspect_rows = [
        {
            "review_id": review.review_id,
            "author": review.author,
            "rating": review.rating,
            "aspect": aspect.aspect,
            "sentiment": aspect.sentiment,
            "confidence": aspect.confidence,
            "quote": aspect.quote,
        }
        for review in aspects
        for aspect in review.aspects
    ]
    pd.DataFrame(aspect_rows).to_csv(out / "aspects.csv", index=False, encoding="utf-8")
    build_heatmap(aspects, out / "heatmap.png")

    print("3/4 Map-Reduce summary...")
    chunk_summaries, summary = summarize_reviews(source_text)
    save_json(out / "chunk_summaries.json", [s.model_dump() for s in chunk_summaries])
    save_json(out / "summary.json", summary.model_dump())

    print("4/4 Judge...")
    report = judge(reviews, summary)
    if report.overall_score < 0.7:
        print("Judge score < 0.7, tightening reduce prompt and retrying...")
        summary = reduce_summaries(
            chunk_summaries,
            "Перепиши action_items: оставь только рекомендации, подтвержденные "
            "минимум двумя независимыми отзывами или одной критичной цитатой.",
        )
        save_json(out / "summary.json", summary.model_dump())
        report = judge(reviews, summary)
    save_json(out / "judge_report.json", report.model_dump())

    ghosts = check_review_quotes(reviews, source_text) + check_aspect_quotes(aspects, source_text)
    elapsed = time.time() - started
    total_quotes = sum(len(r.issues) for r in reviews) + sum(len(r.aspects) for r in aspects)
    metrics = {
        "input_reviews": input_count,
        "structured_reviews": len(reviews),
        "validation_errors": validation_errors,
        "issue_count": sum(len(r.issues) for r in reviews),
        "aspect_quote_count": sum(len(r.aspects) for r in aspects),
        "total_checked_quotes": total_quotes,
        "ghost_quote_count": len(ghosts),
        "ghost_quote_rate": round(len(ghosts) / total_quotes, 4) if total_quotes else 0,
        "ghost_quotes": ghosts,
        "judge_overall_score": report.overall_score,
        "elapsed_seconds": round(elapsed, 2),
        "usage": usage.as_dict(),
    }
    save_json(out / "metrics.json", metrics)
    write_conclusions(out, input_count, len(reviews), validation_errors, ghosts, report, elapsed, metrics)

    print("\nDone.")
    print(f"Reviews: {len(reviews)}")
    print(f"Ghost quotes: {len(ghosts)}/{total_quotes}")
    print(f"Judge score: {report.overall_score:.2f}")
    print(f"Artifacts: {out}")


def main() -> None:
    input_path = sys.argv[1] if len(sys.argv) > 1 else "input/ozon_reviews.txt"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "output"
    analyze(input_path, out_dir)


if __name__ == "__main__":
    main()
