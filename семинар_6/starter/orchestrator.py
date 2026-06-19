"""
Оркестратор: главный цикл Планировщик-Исполнитель-Критик.

На семинаре нужно:
- реализовать topological_sort (TODO 1),
- реализовать replan/rework-ветки цикла (TODO 2),
- написать synthesize для финального ответа (TODO 3).

Важно: max_iter защищает от бесконечного цикла, если Критик
постоянно говорит «переделай».
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from critic import critic
from llm_client import get_model, make_raw_client
from planner import planner
from schemas_pwc import Plan, SubQuestion, WorkerAnswer
from worker import worker

VALID_TOOLS = {"get_fx_rate", "get_key_rate", "get_inflation", "calculate"}


def validate_plan(plan: Plan) -> list[str]:
    """Вернуть список ошибок плана. Пустой список означает валидный план."""
    errors: list[str] = []
    ids = [sq.id for sq in plan.subquestions]
    id_set = set(ids)

    if len(ids) != len(id_set):
        seen: set[int] = set()
        duplicates = sorted({i for i in ids if i in seen or seen.add(i)})
        errors.append(f"дублирующиеся id подвопросов: {duplicates}")

    for sq in plan.subquestions:
        if not sq.expected_tools:
            errors.append(f"подвопрос {sq.id}: expected_tools пустой")

        bad_tools = sorted(set(sq.expected_tools) - VALID_TOOLS)
        if bad_tools:
            errors.append(f"подвопрос {sq.id}: неизвестные инструменты {bad_tools}")

        for dep_id in sq.depends_on:
            if dep_id not in id_set:
                errors.append(f"подвопрос {sq.id}: depends_on ссылается на {dep_id}, которого нет")
            elif dep_id == sq.id:
                errors.append(f"подвопрос {sq.id}: зависит сам от себя")

    try:
        _topological_levels(plan.subquestions)
    except ValueError as e:
        errors.append(str(e))

    return errors


def _topological_sort(subqs: list[SubQuestion]) -> list[SubQuestion]:
    """Отсортировать подвопросы так, чтобы depends_on шли раньше."""
    return [sq for level in _topological_levels(subqs) for sq in level]


def _topological_levels(subqs: list[SubQuestion]) -> list[list[SubQuestion]]:
    """Разбить подвопросы на уровни: внутри уровня зависимостей нет."""
    by_id = {s.id: s for s in subqs}
    state: dict[int, str] = {}
    depth_cache: dict[int, int] = {}

    def depth(node_id: int, path: list[int]) -> int:
        if node_id not in by_id:
            return -1
        if state.get(node_id) == "visiting":
            cycle = path[path.index(node_id):] + [node_id] if node_id in path else path + [node_id]
            raise ValueError(f"цикл в depends_on: {cycle}")
        if state.get(node_id) == "done":
            return depth_cache[node_id]

        state[node_id] = "visiting"
        deps = [dep for dep in by_id[node_id].depends_on if dep in by_id]
        node_depth = 0
        if deps:
            node_depth = 1 + max(depth(dep, path + [node_id]) for dep in deps)
        state[node_id] = "done"
        depth_cache[node_id] = node_depth
        return node_depth

    for sq in subqs:
        depth(sq.id, [])

    levels: dict[int, list[SubQuestion]] = {}
    for sq in subqs:
        levels.setdefault(depth_cache[sq.id], []).append(sq)
    return [levels[i] for i in sorted(levels)]


def execute_level(level: list[SubQuestion], prev_answers: dict[int, WorkerAnswer]) -> dict[int, WorkerAnswer]:
    """Прогнать все независимые подвопросы уровня параллельно."""
    if not level:
        return {}
    if len(level) == 1:
        sq = level[0]
        return {sq.id: worker(sq, prev_answers=prev_answers)}

    out: dict[int, WorkerAnswer] = {}
    with ThreadPoolExecutor(max_workers=len(level)) as pool:
        future_to_sq = {pool.submit(worker, sq, prev_answers): sq for sq in level}
        for fut in as_completed(future_to_sq):
            sq = future_to_sq[fut]
            out[sq.id] = fut.result()
    return out


def _synthesize(
    question: str,
    plan: Plan,
    answers: dict[int, WorkerAnswer],
) -> str:
    """Собрать финальный ответ одним LLM-вызовом без tools."""
    parts = []
    for sq in plan.subquestions:
        ans = answers.get(sq.id)
        if ans is not None:
            parts.append(f"{sq.id}. {sq.question}\nОтвет: {ans.answer}")
    evidence = "\n\n".join(parts)

    try:
        client = make_raw_client()
        resp = client.chat.completions.create(
            model=get_model(),
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Собери короткий финальный ответ пользователю в 1-2 фразы. "
                        "Используй только данные из ответов исполнителей, не вызывай tools "
                        "и не придумывай новые числа."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Исходный вопрос: {question}\n\nДанные:\n{evidence}",
                },
            ],
            temperature=0.0,
        )
        text = resp.choices[0].message.content
        if text:
            return text.strip()
    except Exception:
        pass

    return " ".join(a.answer for _, a in sorted(answers.items()))


def run_pwc(
    question: str,
    *,
    max_iter: int = 3,
    verbose: bool = True,
    enable_validator: bool = True,
) -> dict[str, Any]:
    """Запустить цикл Планировщик-Исполнитель-Критик."""
    trace: list[dict[str, Any]] = []

    plan = planner(question)
    errors = validate_plan(plan) if enable_validator else []
    if errors:
        plan = planner(
            question,
            feedback=f"Инструменты не существуют или схема плана невалидна: {errors}",
        )
    trace.append(
        {
            "iter": 0,
            "kind": "plan",
            "reasoning": plan.reasoning,
            "subquestions": [sq.model_dump() for sq in plan.subquestions],
        }
    )

    if verbose:
        print(f"\n[plan] {plan.reasoning}")
        for sq in plan.subquestions:
            print(f"  {sq.id}. [{','.join(sq.expected_tools)}] {sq.question}")

    if not plan.subquestions:
        trace.append(
            {
                "iter": 0,
                "kind": "empty_plan",
                "answer": plan.reasoning,
            }
        )
        return {
            "answer": plan.reasoning,
            "plan": plan,
            "answers": {},
            "trace": trace,
            "iterations": 0,
        }

    for iter_num in range(1, max_iter + 1):
        answers: dict[int, WorkerAnswer] = {}
        plan_errors = validate_plan(plan) if enable_validator else []
        if plan_errors:
            trace.append(
                {
                    "iter": iter_num,
                    "kind": "validator",
                    "errors": plan_errors,
                }
            )
            plan = planner(
                question,
                feedback=f"Инструменты не существуют или схема плана невалидна: {plan_errors}",
            )
            trace.append(
                {
                    "iter": iter_num,
                    "kind": "replan",
                    "reasoning": plan.reasoning,
                    "subquestions": [sq.model_dump() for sq in plan.subquestions],
                }
            )
            continue

        levels = _topological_levels(plan.subquestions)
        for level_num, level in enumerate(levels, start=1):
            level_answers = execute_level(level, answers)
            for sq in level:
                ans = level_answers[sq.id]
                answers[sq.id] = ans
                trace.append(
                    {
                        "iter": iter_num,
                        "kind": "worker",
                        "level": level_num,
                        "sq_id": sq.id,
                        "used_tools": ans.used_tools,
                        "answer": ans.answer,
                    }
                )
                if verbose:
                    print(f"  [L{level_num}:{sq.id}] → {ans.answer}   tools={ans.used_tools}")

        verdict = critic(question, plan, answers)
        trace.append(
            {
                "iter": iter_num,
                "kind": "verdict",
                "ok": verdict.ok,
                "action": verdict.action,
                "reason": verdict.reason,
                "rework_ids": verdict.rework_ids,
            }
        )

        if verbose:
            mark = "✅" if verdict.ok else "❌"
            print(f"  [critic {mark}] {verdict.action}: {verdict.reason}")

        if verdict.ok:
            final = _synthesize(question, plan, answers)
            return {
                "answer": final,
                "plan": plan,
                "answers": answers,
                "trace": trace,
                "iterations": iter_num,
            }

        if verdict.action == "replan":
            feedback = verdict.reason
        else:
            feedback = f"{verdict.reason}; переделать подвопросы: {verdict.rework_ids}"
        plan = planner(question, feedback=feedback)
        trace.append(
            {
                "iter": iter_num,
                "kind": "replan",
                "reasoning": plan.reasoning,
                "subquestions": [sq.model_dump() for sq in plan.subquestions],
            }
        )

    return {
        "answer": None,
        "error": f"не удалось получить вердикт 'accept' за {max_iter} итераций",
        "plan": plan,
        "answers": answers,
        "trace": trace,
        "iterations": max_iter,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="+", help="Вопрос к агенту")
    ap.add_argument("--max-iter", type=int, default=3)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument(
        "--trace", type=Path, default=None, help="Куда сохранить JSON-лог (если задан)"
    )
    args = ap.parse_args()

    q = " ".join(args.query)
    res = run_pwc(q, max_iter=args.max_iter, verbose=not args.quiet)

    print("\n=== ВОПРОС ===")
    print(q)
    print("\n=== ОТВЕТ ===")
    print(res.get("answer") or res.get("error"))
    print(f"\n(итераций: {res.get('iterations', '?')})")

    if args.trace:
        args.trace.write_text(
            json.dumps(
                {"query": q, **_serialize(res)},
                ensure_ascii=False,
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        print(f"Трейс сохранён: {args.trace}")


def _serialize(res: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in res.items():
        if k == "plan" and v is not None:
            out[k] = v.model_dump()
        elif k == "answers":
            out[k] = {i: a.model_dump() for i, a in v.items()}
        else:
            out[k] = v
    return out


if __name__ == "__main__":
    main()
