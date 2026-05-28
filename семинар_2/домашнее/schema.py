"""
Pydantic-схема заявки на курс повышения квалификации (ДПО).
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

CURRENT_YEAR = date.today().year


def min_age_for_graduation(graduation_year: int) -> int:
    """Минимальный возраст при окончании вуза в ~22 года (для промпта и проверок)."""
    return CURRENT_YEAR - graduation_year + 22


CITIES = (
    "Москва",
    "Санкт-Петербург",
    "Новосибирск",
    "Екатеринбург",
    "Казань",
    "Нижний Новгород",
    "Самара",
    "Краснодар",
    "Ростов-на-Дону",
    "Воронеж",
)

SPECIALITIES = (
    "учитель",
    "врач",
    "инженер",
    "бухгалтер",
    "юрист",
    "экономист",
    "IT-специалист",
    "менеджер",
    "психолог",
)

DESIRED_COURSES = (
    "цифровые компетенции педагога",
    "управление проектами",
    "налоговый учёт и отчётность",
    "медицинская реабилитация",
    "Python для анализа данных",
    "корпоративное право",
    "soft skills для руководителей",
)


class Address(BaseModel):
    city: str
    district: str = Field(min_length=2, max_length=60)

    @field_validator("city")
    @classmethod
    def city_must_be_in_list(cls, v: str) -> str:
        if v not in CITIES:
            raise ValueError(f"Город «{v}» не из утверждённого списка: {', '.join(CITIES)}")
        return v


class Application(BaseModel):
    full_name: str = Field(min_length=5, max_length=120)
    age: int = Field(ge=22, le=65)
    address: Address
    speciality: Literal[
        "учитель",
        "врач",
        "инженер",
        "бухгалтер",
        "юрист",
        "экономист",
        "IT-специалист",
        "менеджер",
        "психолог",
    ]
    desired_course: Literal[
        "цифровые компетенции педагога",
        "управление проектами",
        "налоговый учёт и отчётность",
        "медицинская реабилитация",
        "Python для анализа данных",
        "корпоративное право",
        "soft skills для руководителей",
    ]
    years_of_experience: int = Field(ge=0, le=40)
    graduation_year: int = Field(ge=1980, le=2024)

    @property
    def city(self) -> str:
        return self.address.city

    @model_validator(mode="after")
    def graduation_year_matches_age(self) -> Application:
        """Год окончания вуза и возраст не противоречат друг другу (окончание ~22 года)."""
        if self.graduation_year > CURRENT_YEAR:
            raise ValueError(
                f"Год окончания {self.graduation_year} не может быть позже {CURRENT_YEAR}"
            )
        min_age_after_grad = min_age_for_graduation(self.graduation_year)
        if self.age < min_age_after_grad:
            raise ValueError(
                f"При окончании в {self.graduation_year} возраст {self.age} слишком мал "
                f"(ожидается не меньше {min_age_after_grad})"
            )
        return self
