"""Эксперимент: критик при temperature=0.0 против 0.7.

Скрипт прогоняет пять заведомо сломанных наборов "план + ответы" и считает,
сколько раз критик ошибочно поставил ok=True.

Запуск:
    python critic_temperature_experiment.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from critic import critic
from schemas_pwc import Plan, SubQuestion, WorkerAnswer


def wa(sq_id: int, answer: str, used_tools: list[str]) -> WorkerAnswer:
    return WorkerAnswer(
        subquestion_id=sq_id,
        question_snippet=f"подвопрос {sq_id}",
        answer=answer,
        used_tools=used_tools,
        raw_trace=[],
    )


FAKE_BROKEN = [
    {
        "name": "арифметика без calculate",
        "question": "Во сколько раз USD дороже EUR?",
        "plan": Plan(
            reasoning="Получить два курса и сравнить.",
            subquestions=[
                SubQuestion(id=1, question="Курс USD?", expected_tools=["get_fx_rate"]),
                SubQuestion(id=2, question="Курс EUR?", expected_tools=["get_fx_rate"]),
            ],
        ),
        "answers": {
            1: wa(1, "USD=82.5, EUR=89, отношение=0.927", ["get_fx_rate"]),
        },
    },
    {
        "name": "выдуманное число",
        "question": "Какая реальная ставка сейчас?",
        "plan": Plan(
            reasoning="Нужны ставка, инфляция и разность.",
            subquestions=[
                SubQuestion(id=1, question="Ключевая ставка?", expected_tools=["get_key_rate"]),
                SubQuestion(id=2, question="Инфляция?", expected_tools=["get_inflation"]),
                SubQuestion(id=3, question="Посчитать реальную ставку", expected_tools=["calculate"], depends_on=[1, 2]),
            ],
        ),
        "answers": {
            1: wa(1, "Ключевая ставка 16%.", ["get_key_rate"]),
            2: wa(2, "Инфляция 8%.", ["get_inflation"]),
            3: wa(3, "Реальная ставка 11.5%.", ["calculate"]),
        },
    },
    {
        "name": "несогласованные данные",
        "question": "На сколько пунктов изменилась инфляция?",
        "plan": Plan(
            reasoning="Получить две инфляции и разность.",
            subquestions=[
                SubQuestion(id=1, question="Инфляция в 2022-01?", expected_tools=["get_inflation"]),
                SubQuestion(id=2, question="Инфляция в 2026-03?", expected_tools=["get_inflation"]),
                SubQuestion(id=3, question="Посчитать разность", expected_tools=["calculate"], depends_on=[1, 2]),
            ],
        ),
        "answers": {
            1: wa(1, "В январе 2022 инфляция 8.73%.", ["get_inflation"]),
            2: wa(2, "В марте 2026 инфляция 7.12%.", ["get_inflation"]),
            3: wa(3, "Разность 4.9 п.п.", ["calculate"]),
        },
    },
    {
        "name": "ошибка инструмента принята как ответ",
        "question": "Какая инфляция за последний доступный месяц?",
        "plan": Plan(
            reasoning="Получить инфляцию.",
            subquestions=[
                SubQuestion(id=1, question="Инфляция за июнь 2026?", expected_tools=["get_inflation"]),
            ],
        ),
        "answers": {
            1: wa(1, "(ошибка: нет данных ИПЦ на 2026-06)", ["get_inflation"]),
        },
    },
    {
        "name": "план не покрывает вопрос",
        "question": "Сравни USD и ключевую ставку, затем посчитай разницу.",
        "plan": Plan(
            reasoning="Получить только курс USD.",
            subquestions=[
                SubQuestion(id=1, question="Курс USD сегодня?", expected_tools=["get_fx_rate"]),
            ],
        ),
        "answers": {
            1: wa(1, "Курс USD 71.91 руб.", ["get_fx_rate"]),
        },
    },
]


def run_experiment(n: int = 10) -> list[dict]:
    rows = []
    for case in FAKE_BROKEN:
        row = {
            "case": case["name"],
            "false_accept_t0": 0,
            "false_accept_t07": 0,
            "errors_t0": 0,
            "errors_t07": 0,
        }
        for _ in range(n):
            try:
                v0 = critic(case["question"], case["plan"], case["answers"], temperature=0.0)
                row["false_accept_t0"] += int(v0.ok)
            except Exception:
                row["errors_t0"] += 1

            try:
                v7 = critic(case["question"], case["plan"], case["answers"], temperature=0.7)
                row["false_accept_t07"] += int(v7.ok)
            except Exception:
                row["errors_t07"] += 1
        row["n"] = n
        rows.append(row)
    return rows


def main() -> None:
    rows = run_experiment(n=10)
    print("| Битый кейс | T=0.0, ложных принятий | T=0.7, ложных принятий | ошибки JSON T=0.0/T=0.7 |")
    print("|---|---:|---:|---:|")
    for row in rows:
        print(
            f"| {row['case']} | {row['false_accept_t0']}/{row['n']} "
            f"| {row['false_accept_t07']}/{row['n']} "
            f"| {row['errors_t0']}/{row['errors_t07']} |"
        )

    out = Path(__file__).parent / "critic_temperature_results.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nРезультаты: {out}")


if __name__ == "__main__":
    main()
