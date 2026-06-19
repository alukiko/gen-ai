"""
Планировщик: разбирает исходный вопрос на подвопросы.

На семинаре нужно:
- дописать системный промпт с правилом про calculate (TODO 1),
- добавить поддержку параметра обратной связи для перепланировки (TODO 2).

Ключевая идея: Планировщик НЕ использует tools, только описывает, какие
tools нужны на каждом подвопросе.
"""

from __future__ import annotations

from llm_client import get_model, make_client
from schemas_pwc import Plan

SYSTEM_PROMPT = """\
Ты — планировщик макроэкономического агента. Твоя задача — разложить
сложный вопрос пользователя на 1-5 простых подвопросов, каждый из
которых решается одним конкретным инструментом.

Доступные инструменты (НЕ придумывай других):
- get_fx_rate(currency, on_date): курс валюты к рублю на дату.
  on_date можно не указывать (null) — тогда вернётся курс на сегодня.
- get_key_rate(on_date): ключевая ставка ЦБ на дату.
  on_date можно не указывать (null) — тогда вернётся текущая ставка.
- get_inflation(year, month): ИПЦ г/г на конец месяца.
- calculate(expression): безопасный калькулятор.

ПРАВИЛА:
1. Любая арифметика должна быть вынесена в отдельный подвопрос с
   expected_tools=["calculate"]. Если нужно отношение, разность, сумма,
   произведение, процентное изменение или накопленный показатель, сначала
   получи исходные числа отдельными tools, затем добавь calculate-подвопрос
   с depends_on на эти исходные числа. Нельзя считать "в уме" в подвопросах
   с get_fx_rate/get_key_rate/get_inflation.
2. Если все числа уже явно даны в вопросе пользователя, НЕ добавляй
   подвопросы для получения этих чисел через get_* tools. Сразу добавь один
   calculate-подвопрос с выражением по данным числам.
3. У каждого непустого подвопроса должен быть хотя бы один expected_tools.
   Если задача не требует ни одного инструмента, верни пустой план с
   объяснением в reasoning.
4. Если подвопрос N зависит от ответа подвопроса K — поставь K в depends_on.
5. Для вопросов про «последний доступный период» — первым шагом поставь
   подвопрос «узнать доступный период».
6. Если задача не решается имеющимися tools — верни reasoning с объяснением
   и subquestions=[]. НЕ выдумывай get_cumulative_inflation и подобные.

Цель — минимальный корректный план.
"""


def planner(question: str, *, feedback: str | None = None) -> Plan:
    """Вернуть План для исходного вопроса.

    Args:
        question: вопрос пользователя.
        feedback: (TODO 2) комментарий Критика после неудачной попытки;
                  при наличии — передать как дополнительное пользовательское сообщение,
                  чтобы Планировщик учёл замечание на перепланировку.
    """
    client = make_client()
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    if feedback:
        messages.append(
            {
                "role": "user",
                "content": (
                    "Предыдущая попытка не прошла проверку. "
                    f"Замечание: {feedback}"
                ),
            }
        )

    return client.chat.completions.create(
        model=get_model(),
        messages=messages,
        response_model=Plan,
        temperature=0.0,
        max_retries=2,
    )


if __name__ == "__main__":
    import sys

    q = (
        " ".join(sys.argv[1:])
        or "Во сколько раз USD подорожал с 1 января 2022 по сегодня?"
    )
    plan = planner(q)
    print(f"План (reasoning): {plan.reasoning}\n")
    for sq in plan.subquestions:
        deps = f" ← ждёт {sq.depends_on}" if sq.depends_on else ""
        print(f"  {sq.id}. [{','.join(sq.expected_tools)}]{deps}")
        print(f"     {sq.question}")
