from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

APP_ID = "ru.ozon.app.android"
TARGET_COUNT = 40


def _load_scraper():
    try:
        from google_play_scraper import Sort, reviews
    except ImportError as exc:
        raise SystemExit(
            "Не установлен google-play-scraper. Запусти: "
            "python -m pip install google-play-scraper"
        ) from exc
    return Sort, reviews


def _fetch_batch(lang: str, country: str, count: int) -> list[dict[str, Any]]:
    Sort, reviews = _load_scraper()
    rows, _ = reviews(
        APP_ID,
        lang=lang,
        country=country,
        sort=Sort.NEWEST,
        count=count,
        filter_score_with=None,
    )
    for row in rows:
        row["source_lang"] = lang
        row["source_country"] = country
    return rows


def _clean_text(value: object) -> str:
    return str(value or "").replace("\r\n", "\n").replace("\r", "\n").strip()


def _review_date(value: object) -> str:
    if isinstance(value, datetime):
        return value.date().isoformat()
    return str(value or "")[:10]


def _to_block(index: int, row: dict[str, Any]) -> str:
    text = _clean_text(row.get("content"))
    version = row.get("reviewCreatedVersion") or ""
    return "\n".join(
        [
            f"### REVIEW {index:03d}",
            f"review_id: {row.get('reviewId') or f'review-{index:03d}'}",
            f"author: {_clean_text(row.get('userName')) or 'Unknown'}",
            f"rating: {row.get('score')}",
            f"date: {_review_date(row.get('at'))}",
            f"app_version: {version}",
            f"language: {row.get('source_lang')}",
            "text:",
            text,
            "",
        ]
    )


def collect_reviews(target_count: int = TARGET_COUNT) -> list[dict[str, Any]]:
    rows = _fetch_batch("ru", "ru", target_count)
    if len(rows) < 30:
        seen = {r.get("reviewId") for r in rows}
        for row in _fetch_batch("en", "us", target_count - len(rows)):
            if row.get("reviewId") not in seen:
                rows.append(row)
    return rows[:target_count]


def main() -> None:
    out = Path("input")
    out.mkdir(parents=True, exist_ok=True)
    rows = collect_reviews()

    raw_path = out / "ozon_reviews_raw.json"
    raw_path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    blocks = [_to_block(i, row) for i, row in enumerate(rows, 1)]
    text_path = out / "ozon_reviews.txt"
    text_path.write_text(
        "# Google Play reviews for Ozon marketplace app\n"
        f"# app_id: {APP_ID}\n"
        "# source: https://play.google.com/store/apps/details?id=ru.ozon.app.android\n"
        f"# collected_reviews: {len(rows)}\n\n"
        + "\n".join(blocks),
        encoding="utf-8",
    )

    langs = sorted({str(r.get("source_lang")) for r in rows})
    print(f"Сохранено отзывов: {len(rows)}")
    print(f"Языки: {', '.join(langs)}")
    print(f"TXT: {text_path}")
    print(f"RAW: {raw_path}")


if __name__ == "__main__":
    main()
