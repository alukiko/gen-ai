"""Микробенчмарк параллельного исполнения уровней без LLM.

Нужен, чтобы быстро проверить саму инфраструктуру ThreadPoolExecutor:
одинаковые независимые worker-задачи со sleep должны выполняться быстрее
параллельно, чем последовательно.

Запуск:
    python parallel_benchmark.py
"""
from __future__ import annotations

import time
import argparse
from contextlib import contextmanager
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import orchestrator
from schemas_pwc import SubQuestion, WorkerAnswer


def fake_worker(sq: SubQuestion, prev_answers: dict[int, WorkerAnswer]) -> WorkerAnswer:
    time.sleep(0.7)
    return WorkerAnswer(
        subquestion_id=sq.id,
        question_snippet=sq.question[:60],
        answer=f"ok {sq.id}",
        used_tools=sq.expected_tools,
        raw_trace=[],
    )


@contextmanager
def patched_worker():
    old = orchestrator.worker
    orchestrator.worker = fake_worker
    try:
        yield
    finally:
        orchestrator.worker = old


def sequential(level: list[SubQuestion]) -> float:
    start = time.perf_counter()
    for sq in level:
        fake_worker(sq, {})
    return time.perf_counter() - start


def parallel(level: list[SubQuestion]) -> float:
    start = time.perf_counter()
    with patched_worker():
        orchestrator.execute_level(level, {})
    return time.perf_counter() - start


def sequential_live(level: list[SubQuestion]) -> float:
    start = time.perf_counter()
    answers = {}
    for sq in level:
        answers[sq.id] = orchestrator.worker(sq, answers)
    return time.perf_counter() - start


def parallel_live(level: list[SubQuestion]) -> float:
    start = time.perf_counter()
    orchestrator.execute_level(level, {})
    return time.perf_counter() - start


def bench(name: str, level: list[SubQuestion]) -> dict:
    seq = sequential(level)
    par = parallel(level)
    return {
        "name": name,
        "tasks": len(level),
        "sequential_sec": round(seq, 3),
        "parallel_sec": round(par, 3),
        "speedup": round(seq / par, 2) if par else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="Замерить реальные worker-вызовы через LLM")
    args = ap.parse_args()

    q1_level = [
        SubQuestion(id=1, question="Курс USD на 2022-01-01", expected_tools=["get_fx_rate"]),
        SubQuestion(id=2, question="Курс USD сегодня", expected_tools=["get_fx_rate"]),
    ]
    bonus_level = [
        SubQuestion(id=1, question="Курс USD сегодня", expected_tools=["get_fx_rate"]),
        SubQuestion(id=2, question="Курс EUR сегодня", expected_tools=["get_fx_rate"]),
        SubQuestion(id=3, question="Курс CNY сегодня", expected_tools=["get_fx_rate"]),
        SubQuestion(id=4, question="Ключевая ставка сегодня", expected_tools=["get_key_rate"]),
    ]

    rows = [
        bench("Q1: два независимых курса", q1_level),
        bench("Q5: четыре независимых показателя", bonus_level),
    ]

    print("| Кейс | задач | последовательно, с | параллельно, с | ускорение |")
    print("|---|---:|---:|---:|---:|")
    for row in rows:
        print(
            f"| {row['name']} | {row['tasks']} | {row['sequential_sec']} "
            f"| {row['parallel_sec']} | {row['speedup']}x |"
        )

    if args.live:
        seq = sequential_live(q1_level)
        par = parallel_live(q1_level)
        speedup = seq / par if par else 0
        print("\nLive Q1 worker benchmark:")
        print(f"sequential={seq:.3f}s parallel={par:.3f}s speedup={speedup:.2f}x")


if __name__ == "__main__":
    main()
