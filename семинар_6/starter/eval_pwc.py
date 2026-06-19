"""
Eval мульти-агента: 6 вопросов, на которых одиночный агент С5 ломается.

Каждый вопрос прогоняется в трех конфигурациях:
  1) через одиночного агента С5 (agent_s5.run_agent)
  2) через PWC-цикл без валидатора
  3) через PWC-цикл с валидатором

и сравниваются:
  - вызван ли calculate там, где нужно (для арифметических вопросов)
  - нет ли галлюцинаций инструментов
  - есть ли в ответе обязательная подстрока (must_have)

Прогон N=5 раз, считаем долю успешных прогонов. Результат пишется в eval_pwc_results.json.

Запуск:
    python eval_pwc.py           # полный прогон
    python eval_pwc.py --single  # только один прогон каждого, быстрая проверка
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent_s5 import run_agent
from orchestrator import run_pwc


CASES = [
    {
        "id": "Q1",
        "query": "Во сколько раз USD подорожал с 1 января 2022 по сегодня?",
        "comment": (
            "Класс ошибки C: одиночный часто считает в уме, не зовёт calculate. "
            "PWC должен починить — Планировщик обязан добавить calculate-подвопрос."
        ),
        "expected_tools_pwc": {"get_fx_rate", "calculate"},
        "must_have_keywords": ["раз", "USD"],
        "forbid_hallucinated_tools": True,
    },
    {
        "id": "Q2",
        "query": (
            "Какая сейчас реальная ключевая ставка, если инфляцию брать "
            "по последнему доступному месяцу, а не по году?"
        ),
        "comment": (
            "Класс ошибки B: одиночный не умеет искать «последний доступный» "
            "месяц, зацикливается. PWC должен разбить на шаги."
        ),
        "expected_tools_pwc": {"get_inflation", "get_key_rate", "calculate"},
        "must_have_keywords": ["%"],
        "forbid_hallucinated_tools": True,
    },
    {
        "id": "Q3",
        "query": (
            "Какова накопленная инфляция с января 2022 по март 2026? "
            "Рассчитай как произведение всех (1 + ипц_м/100) по месяцам."
        ),
        "comment": (
            "Класс ошибки D (граница паттерна): требует get_inflation за много "
            "месяцев + большое calculate-выражение. Одиночный галлюцинирует "
            "get_cumulative_inflation; PWC обычно тоже (Планировщик может добавить "
            "выдуманный инструмент в план). Это — повод для Schema-Validator в домашке."
        ),
        "expected_tools_pwc": {"get_inflation", "calculate"},
        "must_have_keywords": ["%"],
        "forbid_hallucinated_tools": True,
    },
    {
        "id": "Q4",
        "query": (
            "Какая накопленная инфляция с января 2022 по март 2026? "
            "Если для этого нужен один инструмент накопленной инфляции, объясни, "
            "почему он недоступен, и не выдумывай его."
        ),
        "comment": (
            "Проверка Schema-Validator: модель может захотеть "
            "get_cumulative_inflation, но валидатор должен вернуть планировщик "
            "к разрешённым инструментам или честному отказу."
        ),
        "expected_tools_pwc": {"get_inflation", "calculate"},
        "must_have_keywords": ["инструмент"],
        "forbid_hallucinated_tools": True,
    },
    {
        "id": "Q5",
        "query": (
            "Собери независимые показатели на сегодня: курс USD, курс EUR, "
            "курс CNY и ключевую ставку ЦБ. Ничего не сравнивай."
        ),
        "comment": (
            "Проверка параллельности: 3+ независимых подвопроса первого уровня "
            "должны исполняться через ThreadPoolExecutor."
        ),
        "expected_tools_pwc": {"get_fx_rate", "get_key_rate"},
        "must_have_keywords": ["USD", "EUR", "CNY"],
        "forbid_hallucinated_tools": True,
    },
    {
        "id": "Q6",
        "query": (
            "По уже известным числам: ключевая ставка 16%, последняя доступная "
            "инфляция 7.72%. Какой реальный процентный зазор между ними в пунктах?"
        ),
        "comment": (
            "Реалистичный короткий вопрос про макроэкономику: числа уже даны, "
            "поэтому нужен calculate для разности."
        ),
        "expected_tools_pwc": {"calculate"},
        "must_have_keywords": ["8.28"],
        "forbid_hallucinated_tools": True,
    },
]


VALID_TOOL_NAMES = {"get_fx_rate", "get_key_rate", "get_inflation", "calculate"}


def _check_single(case: dict, result: dict) -> dict:
    """Проверить результат одиночного прогона."""
    used = {e["call"] for e in result.get("trace", []) if "call" in e}
    ans = (result.get("answer") or "").lower()
    hallucinated = used - VALID_TOOL_NAMES
    must = all(kw.lower() in ans for kw in case["must_have_keywords"])
    sanity_ok = True
    if case["id"] == "Q3":
        # На этом кейсе одиночный агент стабильно делает методическую ошибку:
        # перемножает годовую инфляцию как месячную и получает тысячи процентов.
        sanity_ok = "8587" not in ans and "тысяч" not in ans
    arith_without_calc = (
        case["id"] in {"Q1", "Q2", "Q3"}
        and "calculate" not in used
        and bool(ans)
    )
    ok = bool(ans) and not hallucinated and must and sanity_ok and not arith_without_calc
    return {
        "ok": ok,
        "used_tools": sorted(used),
        "hallucinated": sorted(hallucinated),
        "must_have_ok": must,
        "sanity_ok": sanity_ok,
        "arith_without_calc": arith_without_calc,
        "answer_preview": (result.get("answer") or "")[:180],
    }


def _check_pwc(case: dict, result: dict) -> dict:
    """Проверить результат PWC-прогона."""
    used = set()
    for t in result.get("trace", []):
        if t.get("kind") == "worker":
            used.update(t.get("used_tools") or [])
    ans = (result.get("answer") or "").lower()
    hallucinated = used - VALID_TOOL_NAMES
    # Также проверим галлюцинации на этапе Планировщика (в плане expected_tools)
    plan_tools = set()
    plan = result.get("plan")
    if plan is not None:
        for sq in plan.subquestions:
            plan_tools.update(sq.expected_tools)
    plan_hallucinated = plan_tools - VALID_TOOL_NAMES

    must = all(kw.lower() in ans for kw in case["must_have_keywords"])
    sanity_ok = True
    if case["id"] == "Q3":
        sanity_ok = "8587" not in ans and "тысяч" not in ans
    ok = (
        bool(result.get("answer"))
        and not hallucinated
        and not plan_hallucinated
        and must
        and sanity_ok
    )
    if case["id"] == "Q5":
        levels = [
            t.get("level")
            for t in result.get("trace", [])
            if t.get("kind") == "worker"
        ]
        parallel_level_ok = len([lvl for lvl in levels if lvl == 1]) >= 3
        ok = ok and parallel_level_ok
    else:
        parallel_level_ok = None
    return {
        "ok": ok,
        "used_tools": sorted(used),
        "plan_tools": sorted(plan_tools),
        "hallucinated_in_workers": sorted(hallucinated),
        "hallucinated_in_plan": sorted(plan_hallucinated),
        "must_have_ok": must,
        "sanity_ok": sanity_ok,
        "parallel_level_ok": parallel_level_ok,
        "iterations": result.get("iterations", -1),
        "answer_preview": (result.get("answer") or "")[:180],
    }


def run_case(case: dict, *, n: int = 5) -> dict:
    single = {"runs": [], "pass": 0}
    pwc_no_validator = {"runs": [], "pass": 0}
    pwc_validator = {"runs": [], "pass": 0}

    for i in range(n):
        # --- Одиночный агент ---
        try:
            r1 = run_agent(case["query"], max_iter=8, verbose=False)
        except Exception as e:
            r1 = {"answer": None, "error": f"{type(e).__name__}: {e}", "trace": []}
        check1 = _check_single(case, r1)
        single["runs"].append(check1)
        single["pass"] += int(check1["ok"])

        # --- PWC без валидатора ---
        try:
            r2 = run_pwc(
                case["query"],
                max_iter=3,
                verbose=False,
                enable_validator=False,
            )
        except Exception as e:
            r2 = {"answer": None, "error": f"{type(e).__name__}: {e}",
                  "trace": [], "plan": None}
        check2 = _check_pwc(case, r2)
        pwc_no_validator["runs"].append(check2)
        pwc_no_validator["pass"] += int(check2["ok"])

        # --- PWC с валидатором ---
        try:
            r3 = run_pwc(
                case["query"],
                max_iter=3,
                verbose=False,
                enable_validator=True,
            )
        except Exception as e:
            r3 = {"answer": None, "error": f"{type(e).__name__}: {e}",
                  "trace": [], "plan": None}
        check3 = _check_pwc(case, r3)
        pwc_validator["runs"].append(check3)
        pwc_validator["pass"] += int(check3["ok"])

    return {
        "id": case["id"],
        "query": case["query"],
        "comment": case["comment"],
        "n": n,
        "single": single,
        "pwc_no_validator": pwc_no_validator,
        "pwc_validator": pwc_validator,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--single", action="store_true",
                    help="Только один прогон каждого кейса (быстро)")
    ap.add_argument("-n", type=int, default=5,
                    help="Сколько прогонов на кейс (default=5)")
    args = ap.parse_args()
    n = 1 if args.single else args.n

    print(f"Eval С6: {len(CASES)} кейсов × {n} прогонов\n")
    results = []
    for case in CASES:
        print(f"=== {case['id']}: {case['query'][:70]}...")
        r = run_case(case, n=n)
        results.append(r)
        s = r["single"]
        p0 = r["pwc_no_validator"]
        p1 = r["pwc_validator"]
        print(
            f"   single: {s['pass']}/{n}    "
            f"pwc без валидатора: {p0['pass']}/{n}    "
            f"pwc+validator: {p1['pass']}/{n}"
        )
        for run in p1["runs"][:1]:
            if run["hallucinated_in_plan"]:
                print(f"   ⚠ План содержит выдуманные инструменты: {run['hallucinated_in_plan']}")
        print()

    # Итог
    print("=" * 60)
    print("ИТОГО:")
    for r in results:
        print(
            f"  {r['id']}: single {r['single']['pass']}/{n}  "
            f"pwc-no-validator {r['pwc_no_validator']['pass']}/{n}  "
            f"pwc+validator {r['pwc_validator']['pass']}/{n}  — {r['query'][:60]}"
        )

    out = Path(__file__).parent / "eval_pwc_results.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2,
                              default=str), encoding="utf-8")
    print(f"\nРезультаты: {out}")


if __name__ == "__main__":
    main()
