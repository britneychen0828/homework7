from __future__ import annotations

import logging
import re
from pathlib import Path

from .io_utils import write_json
from .llm import JsonClient


LOGGER = logging.getLogger(__name__)


NARRATION_SYSTEM_PROMPT = """You write narrated lecture voiceover for one slide at a time.
Return strict JSON only.
The narration must sound like the speaker described in style.json.
You must use the slide image plus the style profile, lecture premise, lecture arc, full slide descriptions, and all prior narrations.
The title slide narration must explicitly introduce the speaker in first person and summarize the lecture topic.
For slide 1, start the narration with a direct self-introduction such as "Hi everyone, I'm your instructor for today's lecture..." before summarizing the topic.
Make the opening feel like a real classroom introduction: mention that the speaker is teaching the lecture, briefly establish authority or teaching role, and then preview what the lecture will cover."""


def generate_slide_narrations(
    *,
    slide_paths: list[Path],
    style_profile: dict,
    premise: dict,
    arc: dict,
    slide_descriptions: dict,
    client: JsonClient,
    output_path: Path,
) -> dict:
    description_lookup = {
        item["slide_number"]: item for item in slide_descriptions["slides"]
    }
    combined_rows: list[dict] = []
    write_json(output_path, {"slides": combined_rows})

    for index, slide_path in enumerate(slide_paths, start=1):
        prior_narrations = [
            {
                "slide_number": row["slide_number"],
                "narration": row["narration"],
                "transition_to_next": row["transition_to_next"],
            }
            for row in combined_rows
        ]
        payload = {
            "task": "Write narration for the current slide in context.",
            "current_slide_number": index,
            "current_slide_image_filename": slide_path.name,
            "current_slide_description": description_lookup[index],
            "style_profile": style_profile,
            "premise": premise,
            "arc": arc,
            "all_slide_descriptions": slide_descriptions,
            "prior_slide_narrations": prior_narrations,
            "instructions": [
                "Narration should sound spoken, not written.",
                "Maintain continuity with previous narrations.",
                "For slide 1, explicitly introduce the speaker in first person and summarize the lecture topic.",
                "For slide 1, begin with a direct introduction of the speaker before describing the lecture.",
                "For slide 1, make the introduction fuller than a one-line greeting by briefly stating the speaker's teaching role or expertise and previewing the lecture.",
                "Do not mention being an AI or refer to hidden metadata files.",
            ],
            "required_output_schema": {
                "slide_number": "integer",
                "title": "string",
                "summary": "string",
                "visual_elements": ["string"],
                "key_points": ["string"],
                "continuity_notes": "string",
                "narration": "string",
                "transition_to_next": "string",
            },
        }
        LOGGER.info(
            "Generating narration %03d with %d prior narrations",
            index,
            len(prior_narrations),
        )
        current_description = description_lookup[index]
        previous_narration = combined_rows[-1]["narration"] if combined_rows else ""
        response = generate_aligned_narration(
            client=client,
            slide_path=slide_path,
            payload=payload,
            current_description=current_description,
            previous_narration=previous_narration,
        )
        narration_text = enforce_title_slide_requirements(index=index, narration=response["narration"])
        combined_rows.append(
            {
                "slide_number": index,
                "image_file": slide_path.name,
                "title": current_description["title"],
                "summary": current_description["summary"],
                "visual_elements": current_description["visual_elements"],
                "key_points": current_description["key_points"],
                "continuity_notes": current_description["continuity_notes"],
                "narration": narration_text,
                "transition_to_next": response["transition_to_next"],
            }
        )
        write_json(output_path, {"slides": combined_rows})

    document = {"slides": combined_rows}
    write_json(output_path, document)
    return document


def generate_aligned_narration(
    *,
    client: JsonClient,
    slide_path: Path,
    payload: dict,
    current_description: dict,
    previous_narration: str,
) -> dict:
    response = client.generate_json(
        system_prompt=NARRATION_SYSTEM_PROMPT,
        user_payload=payload,
        image_path=slide_path,
    )
    if narration_matches_current_slide(
        narration=response["narration"],
        current_description=current_description,
        previous_narration=previous_narration,
    ):
        return response

    LOGGER.warning(
        "Narration for slide %03d looked misaligned with the current slide. Regenerating once with a stricter prompt.",
        current_description["slide_number"],
    )
    retry_payload = dict(payload)
    retry_payload["instructions"] = list(payload["instructions"]) + [
        "Your previous attempt appeared to repeat or drift from the prior slide.",
        "Do not reuse the prior slide's narration or examples.",
        "Anchor the narration in the current slide title, current slide summary, and current slide key points.",
        "Make sure the narration clearly reflects this exact slide, not the previous one.",
    ]
    return client.generate_json(
        system_prompt=NARRATION_SYSTEM_PROMPT,
        user_payload=retry_payload,
        image_path=slide_path,
    )


def narration_matches_current_slide(
    *,
    narration: str,
    current_description: dict,
    previous_narration: str,
) -> bool:
    narration_lower = narration.lower()
    title_signals = extract_signal_terms(current_description.get("title", ""))
    key_point_signals = []
    for item in current_description.get("key_points", []):
        key_point_signals.extend(extract_signal_terms(item))
    summary_signals = extract_signal_terms(current_description.get("summary", ""))
    signal_terms = distinct_terms(title_signals + key_point_signals + summary_signals)

    has_current_signal = any(term in narration_lower for term in signal_terms[:8]) if signal_terms else True
    too_similar_to_previous = False
    if previous_narration:
        too_similar_to_previous = normalized_token_overlap(narration_lower, previous_narration.lower()) > 0.72

    return has_current_signal and not too_similar_to_previous


def extract_signal_terms(text: str) -> list[str]:
    lowered = text.lower()
    json_refs = re.findall(r"[a-z_]+\.json", lowered)
    word_terms = [
        token
        for token in re.findall(r"[a-z][a-z0-9_-]{3,}", lowered)
        if token not in {"slide", "agent", "story", "movie", "screenplay", "current", "stage", "builds", "using"}
    ]
    phrase_terms = []
    if "arc agent" in lowered:
        phrase_terms.append("arc agent")
    if "premise agent" in lowered:
        phrase_terms.append("premise agent")
    if "sequence agent" in lowered:
        phrase_terms.append("sequence agent")
    if "scene agent" in lowered:
        phrase_terms.append("scene agent")
    return phrase_terms + json_refs + word_terms


def distinct_terms(terms: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for term in terms:
        if term in seen:
            continue
        seen.add(term)
        ordered.append(term)
    return ordered


def normalized_token_overlap(left: str, right: str) -> float:
    left_tokens = set(re.findall(r"[a-z0-9_.-]+", left))
    right_tokens = set(re.findall(r"[a-z0-9_.-]+", right))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / max(1, min(len(left_tokens), len(right_tokens)))


def enforce_title_slide_requirements(*, index: int, narration: str) -> str:
    if index != 1:
        return narration
    lower = narration.lower()
    strong_intro = (
        "Hi everyone, I'm your instructor for today's lecture from Yale University, and I spend a lot of time working with AI systems and creative workflows. "
        "In this session, I'll walk you through how AI can generate full screenplays by using a structured agentic pipeline instead of trying to write everything in one shot. "
        "We'll cover the core problem, the staged workflow, and how each agent helps keep a long-form document coherent from beginning to end. "
    )
    has_intro = any(
        phrase in lower
        for phrase in [
            "hi everyone, i'm your instructor",
            "hi everyone, i am your instructor",
            "i'm your instructor",
            "i am your instructor",
            "i'm your lecturer",
            "i am your lecturer",
        ]
    )
    has_topic = any(
        phrase in lower
        for phrase in [
            "agentic pipeline",
            "screenplays",
            "one shot",
            "one-shot",
            "long-form document",
            "lecture topic",
            "today, we're going to",
            "today we're going to",
        ]
    )
    if has_intro and has_topic:
        return narration
    return strong_intro + narration
