from __future__ import annotations

import logging
from pathlib import Path

from .io_utils import write_json
from .llm import JsonClient


LOGGER = logging.getLogger(__name__)


DESCRIPTION_SYSTEM_PROMPT = """You describe lecture slides for a downstream narration pipeline.
Return strict JSON only.
For every slide, inspect the image and explain the visible content with enough detail that later stages can build coherent narration.
Use the previous slide descriptions to maintain continuity and avoid contradictions.
Treat prior slide descriptions as substantive context, not optional background.
Use them to identify recurring themes, references, terminology, and transitions across the lecture.
JSON keys must be snake_case."""


def generate_slide_descriptions(
    *,
    slide_paths: list[Path],
    client: JsonClient,
    output_path: Path,
) -> dict:
    slides: list[dict] = []
    write_json(output_path, {"slides": slides})

    for index, slide_path in enumerate(slide_paths, start=1):
        prior_descriptions = [dict(item) for item in slides]
        payload = {
            "task": "Describe the current lecture slide in context.",
            "current_slide_number": index,
            "current_slide_image_filename": slide_path.name,
            "instructions": [
                "Read the current slide image carefully.",
                "Use all prior slide descriptions as real context to maintain continuity.",
                "Infer the lecture role of this slide and how it connects to the previous material.",
                "Carry forward terminology, examples, and instructional progression established earlier.",
                "Return one object with slide-specific fields only.",
            ],
            "prior_slide_descriptions": prior_descriptions,
            "required_output_schema": {
                "slide_number": "integer",
                "image_file": "string",
                "title": "string",
                "summary": "string",
                "visual_elements": ["string"],
                "key_points": ["string"],
                "continuity_notes": "string",
            },
        }
        LOGGER.info(
            "Generating slide description %03d with %d prior descriptions",
            index,
            len(prior_descriptions),
        )
        response = client.generate_json(
            system_prompt=DESCRIPTION_SYSTEM_PROMPT,
            user_payload=payload,
            image_path=slide_path,
        )
        slide_record = {
            "slide_number": int(response["slide_number"]),
            "image_file": slide_path.name,
            "title": response["title"],
            "summary": response["summary"],
            "visual_elements": response["visual_elements"],
            "key_points": response["key_points"],
            "continuity_notes": response["continuity_notes"],
        }
        slides.append(slide_record)
        write_json(output_path, {"slides": slides})

    document = {"slides": slides}
    return document
