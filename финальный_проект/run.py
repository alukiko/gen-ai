from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from agent import optimize_tweet
from schema import TweetInput


PROJECT_ROOT = Path(__file__).resolve().parent


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Tweet engagement optimization agent")
    parser.add_argument("--text", required=True)
    parser.add_argument("--weekday", default="Wednesday")
    parser.add_argument("--hour", type=int, default=13)
    parser.add_argument("--no-llm", action="store_true", help="Use deterministic fallback instead of LLM")
    parser.add_argument("--output", default="output/single_result.json")
    args = parser.parse_args()

    tweet = TweetInput(text=args.text, weekday=args.weekday, hour=args.hour)
    result = optimize_tweet(tweet, use_llm=not args.no_llm)
    out_path = PROJECT_ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    print(json.dumps(result.model_dump(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
