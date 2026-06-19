from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TypeVar

import httpx
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, TypeAdapter


T = TypeVar("T")


PROJECT_ROOT = Path(__file__).resolve().parent
SEMINAR2_ENV = PROJECT_ROOT.parent / "семинар_2" / "домашнее" / ".env"


def load_project_env() -> None:
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    load_dotenv(SEMINAR2_ENV, override=False)


def get_model() -> str:
    return os.getenv("LLM_MODEL", "openai/gpt-4o-mini")


def make_openai_client() -> OpenAI:
    load_project_env()
    token = os.getenv("LLM_AUTH_TOKEN") or os.getenv("OPENAI_API_KEY")
    if not token:
        raise RuntimeError("Set LLM_AUTH_TOKEN in .env or семинар_2/домашнее/.env")
    base_url = os.getenv("LLM_BASE_URL")
    http = httpx.Client(timeout=float(os.getenv("LLM_TIMEOUT", "120")))
    if base_url:
        return OpenAI(api_key=token, base_url=base_url.rstrip("/"), http_client=http)
    return OpenAI(api_key=token, http_client=http)


def _extract_json(text: str) -> object:
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch in "{[":
            try:
                obj, _ = decoder.raw_decode(text[i:])
                return obj
            except json.JSONDecodeError:
                continue
    raise ValueError(f"No JSON object found in LLM response: {text[:200]}")


def structured_completion(
    messages: list[dict],
    response_model: type[T],
    *,
    max_retries: int = 2,
    temperature: float = 0.2,
) -> T:
    adapter = TypeAdapter(response_model)
    schema = adapter.json_schema()
    schema_text = json.dumps(schema, ensure_ascii=False)
    client = make_openai_client()
    model = get_model()
    working_messages = list(messages)
    working_messages.insert(
        0,
        {
            "role": "system",
            "content": (
                "Return exactly one valid JSON object matching this schema. "
                "No markdown, no prose outside JSON.\n"
                f"{schema_text}"
            ),
        },
    )

    last_error: Exception | None = None
    for _ in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=working_messages,
                response_format={"type": "json_object"},
                temperature=temperature,
            )
            raw = response.choices[0].message.content or ""
            return adapter.validate_python(_extract_json(raw))
        except Exception as exc:
            last_error = exc
            working_messages.append(
                {
                    "role": "user",
                    "content": f"Previous answer failed validation: {exc}. Return valid JSON only.",
                }
            )
    raise RuntimeError(f"LLM structured output failed: {last_error}")


class _RewritePlan(BaseModel):
    variants: list[dict]

