from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from agent import optimize_tweet
from schema import TweetInput


PROJECT_ROOT = Path(__file__).resolve().parent
INPUT_PATH = PROJECT_ROOT / "input" / "eval_cases.json"
OUTPUT_DIR = PROJECT_ROOT / "output"


def _load_cases() -> list[TweetInput]:
    raw = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    return [TweetInput(**item) for item in raw]


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Evaluate tweet optimization agent")
    parser.add_argument("--no-llm", action="store_true", help="Use deterministic fallback instead of LLM")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of cases for smoke tests")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cases = _load_cases()
    if args.limit:
        cases = cases[: args.limit]

    rows = []
    results = []
    for idx, case in enumerate(cases, start=1):
        result = optimize_tweet(case, use_llm=not args.no_llm)
        delta = result.best_prediction.engagement_score - result.original_prediction.engagement_score
        passed = result.judge.passes and delta >= 0
        rows.append(
            {
                "case_id": idx,
                "original_score": result.original_prediction.engagement_score,
                "best_score": result.best_prediction.engagement_score,
                "delta": round(delta, 2),
                "passed": passed,
                "judge_score": result.judge.score,
                "tool_steps": len(result.tool_trace),
                "ghost_numbers": result.hallucination_report.ghost_numbers,
                "ghost_quotes": result.hallucination_report.ghost_quotes,
                "used_tools": ";".join(sorted({step["tool"] for step in result.tool_trace if "tool" in step})),
            }
        )
        results.append(result.model_dump())

    pass_rate = sum(1 for row in rows if row["passed"]) / len(rows)
    avg_delta = sum(float(row["delta"]) for row in rows) / len(rows)
    summary = {
        "cases": len(rows),
        "pass_rate": round(pass_rate, 3),
        "avg_delta": round(avg_delta, 3),
        "ghost_numbers_total": sum(int(row["ghost_numbers"]) for row in rows),
        "ghost_quotes_total": sum(int(row["ghost_quotes"]) for row in rows),
        "llm_mode": not args.no_llm,
    }

    (OUTPUT_DIR / "eval_results.json").write_text(
        json.dumps({"summary": summary, "rows": rows, "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (OUTPUT_DIR / "eval_table.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
