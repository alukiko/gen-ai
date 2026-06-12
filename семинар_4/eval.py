import argparse
import json
from pathlib import Path

from pipeline import retrieve_many


BASE_DIR = Path(__file__).parent
GOLD_PATH = BASE_DIR / "data" / "gold.json"
RESULTS_PATH = BASE_DIR / "eval_results.json"


def load_gold() -> list[dict]:
    return json.loads(GOLD_PATH.read_text(encoding="utf-8"))


def score_hit_rate(retrieved_sources: list[str], gold_sources: list[str]) -> float:
    retrieved = set(retrieved_sources)
    return sum(1 for source in gold_sources if source in retrieved) / len(gold_sources)


def evaluate_strategy(strategy: str, k: int, backend: str) -> dict:
    gold = load_gold()
    questions = [item["question"] for item in gold]
    retrieved, chunks, used_backend = retrieve_many(questions, strategy=strategy, k=k, backend=backend)

    rows = []
    total = 0.0
    for item, result in zip(gold, retrieved):
        score = score_hit_rate(result["sources"], item["gold_sources"])
        total += score
        rows.append(
            {
                "id": item["id"],
                "type": item["type"],
                "question": item["question"],
                "gold_sources": item["gold_sources"],
                "retrieved_ids": result["ids"],
                "retrieved_sources": result["sources"],
                "score": score,
            }
        )

    return {
        "strategy": strategy,
        "backend": used_backend,
        "k": k,
        "chunks": len(chunks),
        "hit_rate": total / len(gold),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--backend", choices=["sentence-transformers", "tfidf"], default="sentence-transformers")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    all_results = [evaluate_strategy(strategy, args.k, args.backend) for strategy in ["fixed", "recursive"]]
    RESULTS_PATH.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.quiet:
        return

    for result in all_results:
        print(
            f"\n{result['strategy']} | backend={result['backend']} | chunks={result['chunks']} | "
            f"hit-rate@{result['k']}={result['hit_rate']:.3f}"
        )
        for row in result["rows"]:
            mark = "OK" if row["score"] == 1 else ("PART" if row["score"] > 0 else "MISS")
            print(
                f"[{row['id']:02d}] {mark:4s} {row['score']:.2f} "
                f"gold={row['gold_sources']} got={row['retrieved_sources']}"
            )

    print(f"\nSaved detailed results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
