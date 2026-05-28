"""Проверка applications.csv против схемы Application."""

import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from pydantic import ValidationError

from schema import Application

_HOME = Path(__file__).resolve().parent

CSV = Path(__file__).resolve().parent / "applications.csv"


def row_to_dict(row: pd.Series) -> dict:
    return {
        "full_name": row["full_name"],
        "age": int(row["age"]),
        "address": {"city": row["city"], "district": row["district"]},
        "speciality": row["speciality"],
        "desired_course": row["desired_course"],
        "years_of_experience": int(row["years_of_experience"]),
        "graduation_year": int(row["graduation_year"]),
    }


def main(path: Path = CSV) -> None:
    df = pd.read_csv(path, encoding="utf-8-sig")
    valid, invalid = 0, 0
    cities, specs = Counter(), Counter()

    for i, row in df.iterrows():
        try:
            app = Application(**row_to_dict(row))
            valid += 1
            cities[app.city] += 1
            specs[app.speciality] += 1
        except ValidationError as e:
            invalid += 1
            print(f"  #{i + 1} ✗ {e}")

    print(f"\nВсего: {len(df)}, валидных: {valid}, ошибок: {invalid}")
    if valid:
        print(f"Города: {dict(cities)}")
        print(f"Специальности: {dict(specs)}")
        for name, cnt, lim in (
            ("Город", max(cities.values()), 0.40),
            ("Спец.", max(specs.values()), 0.35),
        ):
            pct = cnt / valid * 100
            ok = "OK" if pct <= lim * 100 else "WARN"
            print(f"  {name} max: {cnt}/{valid} ({pct:.0f}%) {ok}")


if __name__ == "__main__":
    p = Path(sys.argv[1]) if len(sys.argv) > 1 else CSV
    main(p)
