from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from .io_utils import read_text, write_json
from .llm import JsonClient


FILLERS = [
    "okay",
    "right",
    "like",
    "so",
    "yeah",
    "cool",
    "um",
    "uh",
]


STYLE_SYSTEM_PROMPT = """You analyze a lecture transcript and produce a structured speaking-style profile for a downstream narration agent.
Return strict JSON only.
Ground the profile in the actual transcript rather than generic teaching advice.
Make the instructor's tone, pacing, filler habits, framing patterns, and audience relationship explicit.
JSON keys must be snake_case."""


def build_style_profile(captions_path: Path, output_path: Path, client: JsonClient | None = None) -> dict:
    transcript = read_text(captions_path).strip()
    normalized = re.sub(r"\s+", " ", transcript)
    tokens = re.findall(r"[A-Za-z']+", normalized.lower())
    token_counts = Counter(tokens)
    sentence_samples = extract_sentence_samples(transcript)
    filler_counts = [{"token": token, "count": token_counts[token]} for token in FILLERS]

    if client is not None:
        payload = {
            "task": "Create a style profile for lecture narration grounded in this transcript.",
            "transcript_file": captions_path.name,
            "transcript": transcript,
            "detected_filler_counts": filler_counts,
            "required_output_schema": {
                "source": {
                    "captions_file": "string",
                    "method": "string",
                },
                "speaker_profile": {
                    "persona": "string",
                    "audience_relationship": "string",
                    "delivery": ["string"],
                },
                "tone_profile": {
                    "overall_tone": "string",
                    "humor_style": "string",
                    "energy_level": "string",
                },
                "pacing_profile": {
                    "tempo": "string",
                    "rhythm_notes": ["string"],
                    "length_guidance": "string",
                },
                "framing_patterns": {
                    "idea_intro_patterns": ["string"],
                    "audience_engagement_patterns": ["string"],
                    "explanation_strategy": "string",
                },
                "style_rules": {
                    "narration_goals": ["string"],
                    "dos": ["string"],
                    "donts": ["string"],
                },
                "language_signals": {
                    "common_fillers": [{"token": "string", "count": "integer"}],
                    "signature_patterns": ["string"],
                    "sample_sentences": ["string"],
                },
                "narration_constraints": {
                    "tone": "string",
                    "target_length_guidance": "string",
                    "title_slide_requirement": "string",
                },
            },
        }
        response = client.generate_json(system_prompt=STYLE_SYSTEM_PROMPT, user_payload=payload)
        profile = normalize_style_profile(
            response=response,
            captions_file=captions_path.name,
            filler_counts=filler_counts,
            sentence_samples=sentence_samples,
        )
        write_json(output_path, profile)
        return profile

    profile = {
        "source": {
            "captions_file": captions_path.name,
            "method": "deterministic_local_analysis_fallback",
        },
        "speaker_profile": {
            "persona": "Energetic lecturer who explains technical ideas with casual, conversational language.",
            "audience_relationship": "Speaks directly to students, uses check-ins and rhetorical questions, and keeps the tone lively.",
            "delivery": [
                "high-energy",
                "informal",
                "humorous",
                "student-facing",
                "example-driven",
            ],
        },
        "tone_profile": {
            "overall_tone": "confident, warm, playful, and classroom-conversational",
            "humor_style": "Uses jokes, asides, and informal reactions to keep technical material lively.",
            "energy_level": "brisk and high-energy",
        },
        "pacing_profile": {
            "tempo": "brisk",
            "rhythm_notes": [
                "Moves quickly between points while pausing to check audience understanding.",
                "Uses short rhetorical questions and quick transitions to keep momentum.",
                "Alternates between examples and takeaways instead of staying abstract for long.",
            ],
            "length_guidance": "Usually 2-6 spoken sentences per slide depending on slide density.",
        },
        "framing_patterns": {
            "idea_intro_patterns": [
                "Starts with a practical question or scenario.",
                "Frames concepts through concrete examples before abstraction.",
                "Uses casual transition phrases to pivot between ideas.",
            ],
            "audience_engagement_patterns": [
                "Addresses students directly.",
                "Uses rhetorical questions to keep listeners involved.",
                "Adds side comments and quick reactions to maintain energy.",
            ],
            "explanation_strategy": "Introduce a real-world example, explain the concept in plain language, then connect it back to the larger lecture goal.",
        },
        "style_rules": {
            "narration_goals": [
                "Sound like a live lecture, not a formal essay.",
                "Explain slide content clearly and keep momentum between slides.",
                "Use plain language and occasional rhetorical questions to maintain engagement.",
                "Ground explanations in concrete examples before abstract takeaways.",
            ],
            "dos": [
                "Address the audience directly with second-person language when helpful.",
                "Use short transitions that make the lecture feel continuous.",
                "Mix concise summaries with a few conversational flourishes.",
                "Keep the pace brisk and classroom-friendly.",
            ],
            "donts": [
                "Do not sound robotic, overly academic, or corporate.",
                "Do not repeat slide text verbatim unless it helps orient the viewer.",
                "Do not insert long pauses, stage directions, or filler-heavy rambling.",
                "Do not ignore the broader lecture arc when narrating later slides.",
            ],
        },
        "language_signals": {
            "common_fillers": filler_counts,
            "signature_patterns": [
                "frequent rhetorical questions",
                "quick asides to students",
                "casual emphasis such as 'cool', 'awesome', and 'alright'",
                "concepts introduced with practical examples",
            ],
            "sample_sentences": sentence_samples,
        },
        "narration_constraints": {
            "tone": "confident, warm, and slightly playful",
            "target_length_guidance": "Usually 2-6 sentences per slide depending on content density.",
            "title_slide_requirement": "Introduce the speaker and summarize the lecture topic explicitly on the title slide.",
        },
    }
    write_json(output_path, profile)
    return profile


def normalize_style_profile(
    *,
    response: dict,
    captions_file: str,
    filler_counts: list[dict],
    sentence_samples: list[str],
) -> dict:
    return {
        "source": {
            "captions_file": captions_file,
            "method": response.get("source", {}).get("method", "llm_transcript_analysis"),
        },
        "speaker_profile": {
            "persona": response.get("speaker_profile", {}).get(
                "persona",
                "Energetic lecturer who explains technical ideas with casual, conversational language.",
            ),
            "audience_relationship": response.get("speaker_profile", {}).get(
                "audience_relationship",
                "Speaks directly to students and uses check-ins to keep the lecture interactive.",
            ),
            "delivery": response.get("speaker_profile", {}).get(
                "delivery",
                ["high-energy", "informal", "student-facing", "example-driven"],
            ),
        },
        "tone_profile": {
            "overall_tone": response.get("tone_profile", {}).get(
                "overall_tone",
                "confident, warm, and conversational",
            ),
            "humor_style": response.get("tone_profile", {}).get(
                "humor_style",
                "Uses light humor and casual asides to keep the lecture lively.",
            ),
            "energy_level": response.get("tone_profile", {}).get(
                "energy_level",
                "high-energy",
            ),
        },
        "pacing_profile": {
            "tempo": response.get("pacing_profile", {}).get("tempo", "brisk"),
            "rhythm_notes": response.get("pacing_profile", {}).get(
                "rhythm_notes",
                [
                    "Moves quickly between ideas while checking audience understanding.",
                    "Uses short transitions and rhetorical questions to keep momentum.",
                ],
            ),
            "length_guidance": response.get("pacing_profile", {}).get(
                "length_guidance",
                "Usually 2-6 spoken sentences per slide depending on content density.",
            ),
        },
        "framing_patterns": {
            "idea_intro_patterns": response.get("framing_patterns", {}).get(
                "idea_intro_patterns",
                [
                    "Starts with a practical question or scenario.",
                    "Uses examples before abstract explanation.",
                ],
            ),
            "audience_engagement_patterns": response.get("framing_patterns", {}).get(
                "audience_engagement_patterns",
                [
                    "Addresses students directly.",
                    "Uses rhetorical questions and quick asides.",
                ],
            ),
            "explanation_strategy": response.get("framing_patterns", {}).get(
                "explanation_strategy",
                "Introduce a concrete example, explain it in plain language, and tie it back to the lecture goal.",
            ),
        },
        "style_rules": {
            "narration_goals": response.get("style_rules", {}).get(
                "narration_goals",
                [
                    "Sound like a live lecture, not a formal essay.",
                    "Explain slide content clearly and keep momentum between slides.",
                ],
            ),
            "dos": response.get("style_rules", {}).get(
                "dos",
                [
                    "Address the audience directly when helpful.",
                    "Keep transitions clear and conversational.",
                ],
            ),
            "donts": response.get("style_rules", {}).get(
                "donts",
                [
                    "Do not sound robotic or overly academic.",
                    "Do not ignore the broader lecture arc.",
                ],
            ),
        },
        "language_signals": {
            "common_fillers": response.get("language_signals", {}).get("common_fillers", filler_counts),
            "signature_patterns": response.get("language_signals", {}).get(
                "signature_patterns",
                [
                    "frequent rhetorical questions",
                    "quick asides to students",
                    "concepts introduced with practical examples",
                ],
            ),
            "sample_sentences": response.get("language_signals", {}).get("sample_sentences", sentence_samples),
        },
        "narration_constraints": {
            "tone": response.get("narration_constraints", {}).get(
                "tone",
                "confident, warm, and slightly playful",
            ),
            "target_length_guidance": response.get("narration_constraints", {}).get(
                "target_length_guidance",
                response.get("pacing_profile", {}).get(
                    "length_guidance",
                    "Usually 2-6 sentences per slide depending on content density.",
                ),
            ),
            "title_slide_requirement": response.get("narration_constraints", {}).get(
                "title_slide_requirement",
                "Introduce the speaker and summarize the lecture topic explicitly on the title slide.",
            ),
        },
    }


def extract_sentence_samples(transcript: str, limit: int = 5) -> list[str]:
    cleaned = transcript.replace("\n", " ")
    cleaned = re.sub(r"^\[.*?\]\s*", "", cleaned)
    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    samples: list[str] = []
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence.split()) >= 8:
            samples.append(sentence)
        if len(samples) >= limit:
            break
    return samples
