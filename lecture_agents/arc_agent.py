from __future__ import annotations

import logging

from .io_utils import write_json
from .llm import JsonClient


LOGGER = logging.getLogger(__name__)


ARC_SYSTEM_PROMPT = """You plan the instructional arc of a lecture.
Return strict JSON only. Use both the lecture premise and the slide-by-slide descriptions.
Make the lecture flow, phases, and how ideas build from one section to the next explicit."""


def generate_arc(
    *,
    premise: dict,
    slide_descriptions: dict,
    client: JsonClient,
    output_path,
) -> dict:
    LOGGER.info("Generating arc.json")
    payload = {
        "task": "Build a lecture arc that maps how the talk progresses.",
        "premise": premise,
        "slide_descriptions": slide_descriptions,
        "required_output_schema": {
            "flow_summary": "string",
            "opening": "string",
            "middle_development": "string",
            "closing": "string",
            "idea_build": ["string"],
            "slide_groups": [
                {
                    "group_name": "string",
                    "start_slide": "integer",
                    "end_slide": "integer",
                    "purpose": "string",
                    "how_it_builds_on_previous": "string",
                }
            ],
            "narrative_threads": ["string"],
            "consistency_with_premise": "string",
        },
    }
    response = client.generate_json(system_prompt=ARC_SYSTEM_PROMPT, user_payload=payload)
    normalized = normalize_arc(response)
    write_json(output_path, normalized)
    return normalized


def normalize_arc(response: dict) -> dict:
    slide_groups = []
    for group in response.get("slide_groups", []):
        slide_groups.append(
            {
                "group_name": group.get("group_name", ""),
                "start_slide": group.get("start_slide", 0),
                "end_slide": group.get("end_slide", 0),
                "purpose": group.get("purpose", ""),
                "how_it_builds_on_previous": group.get(
                    "how_it_builds_on_previous",
                    "Extends the prior section with additional structure, detail, or application.",
                ),
            }
        )
    return {
        "flow_summary": response.get(
            "flow_summary",
            "The lecture moves from motivation and problem framing to a staged agent architecture, then to implementation details and broader application.",
        ),
        "opening": response.get("opening", ""),
        "middle_development": response.get("middle_development", ""),
        "closing": response.get("closing", ""),
        "idea_build": response.get("idea_build", response.get("narrative_threads", [])),
        "slide_groups": slide_groups,
        "narrative_threads": response.get("narrative_threads", []),
        "consistency_with_premise": response.get(
            "consistency_with_premise",
            "The arc stays aligned with the premise by showing how modular agents solve the long-form coherence problem.",
        ),
    }
