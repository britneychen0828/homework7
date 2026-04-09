from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Protocol

from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from openai import OpenAI
from PIL import Image

from .io_utils import encode_image_base64


LOGGER = logging.getLogger(__name__)


class JsonClient(Protocol):
    def generate_json(
        self,
        *,
        system_prompt: str,
        user_payload: dict[str, Any],
        image_path: Path | None = None,
    ) -> dict[str, Any]:
        ...


class OpenAIJsonClient:
    def __init__(self, api_key: str, model: str) -> None:
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate_json(
        self,
        *,
        system_prompt: str,
        user_payload: dict[str, Any],
        image_path: Path | None = None,
    ) -> dict[str, Any]:
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": json.dumps(user_payload, indent=2, ensure_ascii=True),
            }
        ]
        if image_path is not None:
            data_url = f"data:image/png;base64,{encode_image_base64(image_path)}"
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": data_url},
                }
            )

        LOGGER.debug("Calling OpenAI model %s", self.model)
        response = self.client.chat.completions.create(
            model=self.model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
        )
        text = response.choices[0].message.content
        if not text:
            raise RuntimeError("Model returned an empty JSON response.")
        return parse_json_response(text)


class GeminiJsonClient:
    def __init__(self, api_key: str, model: str) -> None:
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def generate_json(
        self,
        *,
        system_prompt: str,
        user_payload: dict[str, Any],
        image_path: Path | None = None,
    ) -> dict[str, Any]:
        parts: list[Any] = [json.dumps(user_payload, indent=2, ensure_ascii=True)]
        if image_path is not None:
            with Image.open(image_path) as image:
                parts.append(image.copy())
        response = call_with_retries(
            lambda: self.client.models.generate_content(
                model=self.model,
                contents=parts,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    response_mime_type="application/json",
                ),
            ),
            model_name=self.model,
        )
        text = response.text
        if not text:
            raise RuntimeError("Gemini returned an empty JSON response.")
        return parse_json_response(text)


def build_json_client(*, provider: str, model: str, openai_api_key: str | None, gemini_api_key: str | None) -> JsonClient:
    provider = provider.lower()
    if provider == "openai":
        if not openai_api_key:
            raise RuntimeError("Text provider 'openai' requires OPENAI_API_KEY.")
        return OpenAIJsonClient(api_key=openai_api_key, model=model)
    if provider == "gemini":
        if not gemini_api_key:
            raise RuntimeError("Text provider 'gemini' requires GEMINI_API_KEY or GOOGLE_API_KEY.")
        return GeminiJsonClient(api_key=gemini_api_key, model=model)
    raise RuntimeError(f"Unsupported text provider: {provider}")


def parse_json_response(text: str) -> dict[str, Any]:
    candidate = extract_json_object(text)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        repaired = escape_control_characters_in_strings(candidate)
        return json.loads(repaired)


def extract_json_object(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end >= start:
        return stripped[start : end + 1]
    return stripped


def escape_control_characters_in_strings(text: str) -> str:
    result: list[str] = []
    in_string = False
    escape = False

    for char in text:
        if in_string:
            if escape:
                result.append(char)
                escape = False
                continue
            if char == "\\":
                result.append(char)
                escape = True
                continue
            if char == '"':
                result.append(char)
                in_string = False
                continue
            if ord(char) < 32:
                result.append(
                    {
                        "\n": "\\n",
                        "\r": "\\r",
                        "\t": "\\t",
                        "\b": "\\b",
                        "\f": "\\f",
                    }.get(char, f"\\u{ord(char):04x}")
                )
                continue
            result.append(char)
            continue

        result.append(char)
        if char == '"':
            in_string = True

    return "".join(result)


def call_with_retries(func, *, model_name: str, max_attempts: int = 5, base_delay_seconds: float = 2.0):
    for attempt in range(1, max_attempts + 1):
        try:
            return func()
        except genai_errors.ServerError as exc:
            if attempt == max_attempts:
                raise
            delay = base_delay_seconds * attempt
            LOGGER.warning(
                "Transient Gemini error from %s on attempt %d/%d: %s. Retrying in %.1fs.",
                model_name,
                attempt,
                max_attempts,
                exc,
                delay,
            )
            time.sleep(delay)
