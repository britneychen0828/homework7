from __future__ import annotations

import logging
import re
import subprocess
from pathlib import Path

from google import genai
from google.genai import types
from openai import OpenAI

from .io_utils import zero_padded_name
from .llm import call_with_retries


LOGGER = logging.getLogger(__name__)


def synthesize_all_slides(
    *,
    narration_document: dict,
    audio_dir: Path,
    temp_dir: Path,
    provider: str,
    api_key: str | None,
    gemini_api_key: str | None,
    model: str,
    voice: str,
) -> list[Path]:
    client = OpenAI(api_key=api_key) if provider == "openai" else None
    gemini_client = genai.Client(api_key=gemini_api_key) if provider == "gemini" and gemini_api_key else None
    audio_paths: list[Path] = []
    for slide in narration_document["slides"]:
        slide_number = int(slide["slide_number"])
        output_path = audio_dir / zero_padded_name("slide", slide_number, "mp3")
        if provider == "openai":
            synthesize_with_openai(
                client=client,
                model=model,
                voice=voice,
                text=slide["narration"],
                output_path=output_path,
                temp_dir=temp_dir / f"slide_{slide_number:03d}",
            )
        elif provider == "gemini":
            synthesize_with_gemini(
                client=gemini_client,
                model=model,
                voice=voice,
                text=slide["narration"],
                output_path=output_path,
                temp_dir=temp_dir / f"slide_{slide_number:03d}",
            )
        elif provider == "say":
            synthesize_with_say(
                text=slide["narration"],
                output_path=output_path,
                temp_dir=temp_dir / f"slide_{slide_number:03d}",
            )
        else:
            raise RuntimeError(f"Unsupported TTS provider: {provider}")
        audio_paths.append(output_path)
    return audio_paths


def synthesize_with_openai(
    *,
    client: OpenAI | None,
    model: str,
    voice: str,
    text: str,
    output_path: Path,
    temp_dir: Path,
) -> None:
    if client is None:
        raise RuntimeError("OpenAI TTS selected without an OpenAI client.")
    temp_dir.mkdir(parents=True, exist_ok=True)
    chunks = chunk_text(text, max_chars=1500)
    chunk_files: list[Path] = []
    for index, chunk in enumerate(chunks, start=1):
        chunk_path = temp_dir / f"chunk_{index:03d}.mp3"
        LOGGER.info("Synthesizing %s chunk %03d/%03d", output_path.name, index, len(chunks))
        with client.audio.speech.with_streaming_response.create(
            model=model,
            voice=voice,
            input=chunk,
            format="mp3",
        ) as response:
            response.stream_to_file(chunk_path)
        chunk_files.append(chunk_path)

    if len(chunk_files) == 1:
        output_path.write_bytes(chunk_files[0].read_bytes())
        return

    concat_list = temp_dir / "concat.txt"
    concat_list.write_text(
        "".join(f"file '{path.resolve()}'\n" for path in chunk_files),
        encoding="utf-8",
    )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_list),
            "-c",
            "copy",
            str(output_path),
        ],
        check=True,
        capture_output=True,
        text=True,
        )


def synthesize_with_say(*, text: str, output_path: Path, temp_dir: Path) -> None:
    temp_dir.mkdir(parents=True, exist_ok=True)
    aiff_path = temp_dir / "speech.aiff"
    subprocess.run(
        [
            "say",
            "-o",
            str(aiff_path),
            text,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(aiff_path),
            "-codec:a",
            "libmp3lame",
            "-q:a",
            "2",
            str(output_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def synthesize_with_gemini(
    *,
    client: genai.Client | None,
    model: str,
    voice: str,
    text: str,
    output_path: Path,
    temp_dir: Path,
) -> None:
    if client is None:
        raise RuntimeError("Gemini TTS selected without GEMINI_API_KEY.")
    temp_dir.mkdir(parents=True, exist_ok=True)
    pcm_path = temp_dir / "speech.pcm"
    response = call_with_retries(
        lambda: client.models.generate_content(
            model=model,
            contents=text,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice)
                    )
                )
            ),
        ),
        model_name=model,
    )
    audio_bytes = extract_gemini_audio_bytes(response)
    if not audio_bytes:
        raise RuntimeError("Gemini TTS response did not include audio bytes.")
    pcm_path.write_bytes(audio_bytes)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "s16le",
            "-ar",
            "24000",
            "-ac",
            "1",
            "-i",
            str(pcm_path),
            "-codec:a",
            "libmp3lame",
            "-q:a",
            "2",
            str(output_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def extract_gemini_audio_bytes(response: object) -> bytes:
    candidates = getattr(response, "candidates", None) or []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", None) or []
        for part in parts:
            inline_data = getattr(part, "inline_data", None)
            data = getattr(inline_data, "data", None) if inline_data is not None else None
            if isinstance(data, bytes) and data:
                return data
    return b""


def chunk_text(text: str, max_chars: int) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        if not sentence:
            continue
        candidate = sentence if not current else f"{current} {sentence}"
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            chunks.append(current)
        current = sentence
    if current:
        chunks.append(current)
    return chunks or [text]
