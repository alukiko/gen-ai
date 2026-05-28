"""
Генератор синтетических заявок на курсы ДПО.

Стратификация: по 5 заявок на каждый из 10 городов (50 всего).
В промпт передаётся seed_city и seed_speciality против mode collapse.
"""

from __future__ import annotations

import random
import sys
import time
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from llm_client import get_model, make_client
from schema import (
    CITIES,
    CURRENT_YEAR,
    DESIRED_COURSES,
    SPECIALITIES,
    Application,
    min_age_for_graduation,
)

_HOME = Path(__file__).resolve().parent

N_APPLICATIONS = 50
PER_CITY = N_APPLICATIONS // len(CITIES)
MAX_RETRIES = 3
MAX_SLOT_ATTEMPTS = 3  # повтор всего запроса к API, если слот не прошёл после retry внутри llm_client
OUT_DIR = Path(__file__).resolve().parent

SYSTEM_PROMPT = f"""Ты генерируешь одну синтетическую заявку на курс повышения квалификации (ДПО) в России.
Текущий год: {CURRENT_YEAR}.

Правило согласованности (обязательно):
  age >= {CURRENT_YEAR} - graduation_year + 22
Пример: graduation_year=2000 → age не меньше {min_age_for_graduation(2000)}.

Ответ — один JSON-объект по схеме Application, без текста вокруг.
Поля age, graduation_year, address.city, speciality — возьми ТОЧНО из запроса пользователя.
Остальное придумай правдоподобно: ФИО (кириллица), район города, курс из списка, стаж 0–40 (не больше age-22)."""

USER_PROMPT = """Создай заявку. Жёстко зафиксированные поля (скопируй в JSON как есть):
- address.city = "{seed_city}" (district — правдоподобный район этого города)
- speciality = "{seed_speciality}"
- graduation_year = {graduation_year}
- age = {age}

Проверка: при graduation_year={graduation_year} минимальный age = {min_age} (текущий год {current_year}).

Желаемый курс — только из: {courses}.
Стаж years_of_experience: от 0 до 40, не больше {max_experience}.
Разнообразное ФИО, не «Иванов Иван Иванович»."""

EXAMPLE_JSON = """{
  "full_name": "Козлова Мария Петровна",
  "age": 48,
  "address": {"city": "Казань", "district": "Вахитовский"},
  "speciality": "учитель",
  "desired_course": "цифровые компетенции педагога",
  "years_of_experience": 20,
  "graduation_year": 1998
}"""


def build_city_quota() -> list[str]:
    """Стратификация: ровно PER_CITY заявок на каждый город."""
    quota = []
    for city in CITIES:
        quota.extend([city] * PER_CITY)
    random.shuffle(quota)
    return quota


def build_speciality_quota() -> list[str]:
    """Равномернее специальностей: цикл по 9 специальностям + перемешивание."""
    quota: list[str] = []
    while len(quota) < N_APPLICATIONS:
        quota.extend(SPECIALITIES)
    quota = quota[:N_APPLICATIONS]
    random.shuffle(quota)
    return quota


def pick_age_and_graduation(rng: random.Random) -> tuple[int, int]:
    """Согласованная пара год окончания / возраст — проходит @model_validator."""
    graduation_year = rng.randint(1985, min(2018, CURRENT_YEAR - 22))
    min_age = min_age_for_graduation(graduation_year)
    max_age = min(65, min_age + rng.randint(0, 15))
    age = rng.randint(min_age, max_age)
    return graduation_year, age


def generate_one(
    client,
    model: str,
    seed_city: str,
    seed_speciality: str,
    graduation_year: int,
    age: int,
) -> Application:
    min_age = min_age_for_graduation(graduation_year)
    max_experience = max(0, min(40, age - 22))
    return client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Пример валидного JSON:\n{EXAMPLE_JSON}"},
            {
                "role": "user",
                "content": USER_PROMPT.format(
                    seed_city=seed_city,
                    seed_speciality=seed_speciality,
                    graduation_year=graduation_year,
                    age=age,
                    min_age=min_age,
                    current_year=CURRENT_YEAR,
                    courses=", ".join(DESIRED_COURSES),
                    max_experience=max_experience,
                ),
            },
        ],
        response_model=Application,
        max_retries=MAX_RETRIES,
        temperature=0.7,
    )


def generate_one_with_slot_retries(
    client,
    model: str,
    seed_city: str,
    seed_speciality: str,
    rng: random.Random,
) -> Application:
    """До MAX_SLOT_ATTEMPTS вызовов API с новой парой age/graduation при полном провале."""
    last_err: Exception | None = None
    for _ in range(MAX_SLOT_ATTEMPTS):
        graduation_year, age = pick_age_and_graduation(rng)
        try:
            app = generate_one(
                client, model, seed_city, seed_speciality, graduation_year, age
            )
            if app.address.city != seed_city:
                app = app.model_copy(
                    update={"address": app.address.model_copy(update={"city": seed_city})}
                )
            if app.speciality != seed_speciality:
                app = app.model_copy(update={"speciality": seed_speciality})
            if app.graduation_year != graduation_year or app.age != age:
                app = app.model_copy(
                    update={"graduation_year": graduation_year, "age": age}
                )
            return app
        except Exception as e:
            last_err = e
    assert last_err is not None
    raise last_err


def applications_to_dataframe(apps: list[Application]) -> pd.DataFrame:
    rows = []
    for a in apps:
        row = a.model_dump()
        addr = row.pop("address")
        row["city"] = addr["city"]
        row["district"] = addr["district"]
        rows.append(row)
    return pd.DataFrame(rows)


def save_plots(df: pd.DataFrame) -> None:
    for col, title, filename, color in (
        ("city", "Распределение заявок по городам", "cities.png", "#7AB66E"),
        ("speciality", "Распределение по специальностям", "specialities.png", "#D97A4A"),
    ):
        counts = df[col].value_counts()
        plt.figure(figsize=(9, 4))
        counts.plot.bar(color=color, edgecolor="white")
        plt.title(title)
        plt.ylabel("Число заявок")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(OUT_DIR / filename, dpi=120)
        plt.close()


# Районы по городам — для офлайн-режима без API
DISTRICTS: dict[str, list[str]] = {
    "Москва": ["СВАО", "ЮЗАО", "ЦАО", "САО", "ВАО"],
    "Санкт-Петербург": ["Невский", "Василеостровский", "Приморский", "Московский", "Калининский"],
    "Новосибирск": ["Центральный", "Советский", "Октябрьский", "Кировский", "Дзержинский"],
    "Екатеринбург": ["ВИЗ", "Ленинский", "Октябрьский", "Чкаловский", "Железнодорожный"],
    "Казань": ["Вахитовский", "Советский", "Приволжский", "Ново-Савиновский", "Авиастроительный"],
    "Нижний Новгород": ["Нижегородский", "Советский", "Приокский", "Автозаводский", "Канавинский"],
    "Самара": ["Советский", "Промышленный", "Кировский", "Октябрьский", "Самарский"],
    "Краснодар": ["Центральный", "Западный", "Прикубанский", "Карасунский", "Юбилейный"],
    "Ростов-на-Дону": ["Центральный", "Советский", "Ворошиловский", "Первомайский", "Ленинский"],
    "Воронеж": ["Центральный", "Советский", "Коминтерновский", "Левобережный", "Железнодорожный"],
}

FIRST_NAMES = (
    "Алексей", "Мария", "Дмитрий", "Елена", "Сергей", "Ольга", "Андрей", "Наталья",
    "Игорь", "Татьяна", "Павел", "Анна", "Виктор", "Светлана", "Николай", "Юлия",
)
LAST_NAMES = (
    "Смирнов", "Кузнецова", "Попов", "Васильева", "Новиков", "Морозова", "Фёдоров",
    "Волкова", "Соколов", "Лебедева", "Козлов", "Семёнова", "Егоров", "Павлова",
)
PATRONYMICS_M = ("Иванович", "Петрович", "Сергеевич", "Андреевич", "Дмитриевич")
PATRONYMICS_F = ("Ивановна", "Петровна", "Сергеевна", "Андреевна", "Дмитриевична")

COURSE_BY_SPECIALITY: dict[str, list[str]] = {
    "учитель": ["цифровые компетенции педагога", "soft skills для руководителей"],
    "врач": ["медицинская реабилитация", "управление проектами"],
    "инженер": ["управление проектами", "Python для анализа данных"],
    "бухгалтер": ["налоговый учёт и отчётность", "управление проектами"],
    "юрист": ["корпоративное право", "управление проектами"],
    "экономист": ["налоговый учёт и отчётность", "управление проектами"],
    "IT-специалист": ["Python для анализа данных", "управление проектами"],
    "менеджер": ["soft skills для руководителей", "управление проектами"],
    "психолог": ["soft skills для руководителей", "цифровые компетенции педагога"],
}


def _random_fio(rng: random.Random) -> str:
    female = rng.random() < 0.5
    male_names = ("Алексей", "Дмитрий", "Сергей", "Андрей", "Игорь", "Павел", "Виктор", "Николай")
    female_names = ("Мария", "Елена", "Ольга", "Наталья", "Татьяна", "Анна", "Светлана", "Юлия")
    first = rng.choice(female_names if female else male_names)
    last = rng.choice(LAST_NAMES)
    if female:
        if last.endswith("ов"):
            last = last[:-2] + "ова"
        elif last.endswith("ев"):
            last = last[:-2] + "ева"
        elif last.endswith("ин"):
            last = last + "а"
    pat = rng.choice(PATRONYMICS_F if female else PATRONYMICS_M)
    return f"{last} {first} {pat}"


def generate_offline(city_quota: list[str]) -> list[Application]:
    """Стратифицированная генерация без LLM — для проверки пайплайна и артефактов."""
    rng = random.Random(42)
    apps: list[Application] = []
    specialities_cycle = list(SPECIALITIES) * 6
    rng.shuffle(specialities_cycle)

    for i, city in enumerate(city_quota):
        spec = specialities_cycle[i % len(specialities_cycle)]
        course = rng.choice(COURSE_BY_SPECIALITY[spec])
        graduation_year = rng.randint(1985, 2018)
        min_age = date.today().year - graduation_year + 22
        age = rng.randint(min_age, min(65, min_age + 25))
        years_exp = min(40, max(0, age - 22 - rng.randint(0, 5)))

        app = Application(
            full_name=_random_fio(rng),
            age=age,
            address={
                "city": city,
                "district": rng.choice(DISTRICTS[city]),
            },
            speciality=spec,
            desired_course=course,
            years_of_experience=years_exp,
            graduation_year=graduation_year,
        )
        apps.append(app)
    return apps


def print_distribution_report(df: pd.DataFrame) -> None:
    n = len(df)
    print("\n-- Распределения --")
    for col, threshold in (("city", 0.40), ("speciality", 0.35)):
        vc = df[col].value_counts()
        top = vc.iloc[0]
        pct = top / n * 100
        status = "OK" if pct <= threshold * 100 else "WARN: порог превышен"
        print(f"  {col}: топ '{vc.index[0]}' -- {top}/{n} ({pct:.0f}%) [{status}]")


def main() -> None:
    random.seed(42)
    offline = "--offline" in sys.argv
    city_quota = build_city_quota()
    applications: list[Application] = []

    if offline:
        print(f"Офлайн-режим: {N_APPLICATIONS} заявок, стратификация {PER_CITY}/город\n")
        applications = generate_offline(city_quota)
    else:
        try:
            client = make_client()
            model = get_model()
        except RuntimeError as e:
            print(f"⚠ {e}")
            print("Запустите с ключом в .env или: python generator.py --offline\n")
            sys.exit(1)

        slot_failures = 0
        speciality_quota = build_speciality_quota()
        rng = random.Random(42)
        print(
            f"Модель: {model}, заявок: {N_APPLICATIONS}, "
            f"стратификация: {PER_CITY}/город, seed age+graduation_year\n"
        )

        for i, seed_city in enumerate(city_quota, 1):
            seed_speciality = speciality_quota[i - 1]
            print(
                f"[{i:02d}/{N_APPLICATIONS}] {seed_city}, {seed_speciality}...",
                end=" ",
            )
            try:
                app = generate_one_with_slot_retries(
                    client, model, seed_city, seed_speciality, rng
                )
                applications.append(app)
                print("OK")
            except Exception as e:
                print(f"FAIL {type(e).__name__}")
                slot_failures += 1
            time.sleep(0.2)

        print(f"\nСгенерировано валидных: {len(applications)}/{N_APPLICATIONS}")
        if slot_failures:
            print(f"Слотов без заявки (все {MAX_SLOT_ATTEMPTS} попытки): {slot_failures}")

        if not applications:
            sys.exit("Нет заявок — проверьте LLM_BASE_URL / LLM_AUTH_TOKEN в .env")

    if len(applications) < N_APPLICATIONS:
        sys.exit(f"Недостаточно заявок: {len(applications)}/{N_APPLICATIONS}")

    df = applications_to_dataframe(applications)
    csv_path = OUT_DIR / "applications.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"Сохранено: {csv_path}")

    save_plots(df)
    print("Сохранено: cities.png, specialities.png")
    print_distribution_report(df)


if __name__ == "__main__":
    main()
