from __future__ import annotations

import logging

from .io_utils import write_json
from .llm import JsonClient


LOGGER = logging.getLogger(__name__)


PREMISE_SYSTEM_PROMPT = """You are synthesizing the overall premise of a lecture from detailed slide descriptions.
Return strict JSON only with concise, high-signal fields that later narration stages can reuse.
The output must make the lecture thesis, scope, learning objectives, and audience explicit."""


def generate_premise(*, slide_descriptions: dict, client: JsonClient, output_path) -> dict:
    LOGGER.info("Generating premise.json")
    payload = {
        "task": "Summarize the lecture's premise from the complete slide descriptions.",
        "slide_descriptions": slide_descriptions,
        "required_output_schema": {
            "lecture_title": "string",
            "thesis": "string",
            "scope": "string",
            "core_premise": "string",
            "learning_objectives": ["string"],
            "audience": "string",
            "key_concepts": ["string"],
            "why_it_matters": "string",
        },
    }
    response = client.generate_json(system_prompt=PREMISE_SYSTEM_PROMPT, user_payload=payload)
    normalized = normalize_premise(response)
    write_json(output_path, normalized)
    return normalized


def normalize_premise(response: dict) -> dict:
    thesis = response.get("thesis") or response.get("core_premise", "")
    scope = response.get("scope") or "Defines the lecture boundaries, examples, and applications covered by the deck."
    core_premise = response.get("core_premise") or thesis
    audience = response.get("audience") or response.get("target_audience") or ""
    return {
        "lecture_title": response.get("lecture_title", ""),
        "thesis": thesis,
        "scope": scope,
        "core_premise": core_premise,
        "learning_objectives": response.get("learning_objectives", []),
        "audience": audience,
        "key_concepts": response.get("key_concepts", []),
        "why_it_matters": response.get(
            "why_it_matters",
            "Shows how a structured agent pipeline can improve long-form generation quality and coherence.",
        ),
    }
