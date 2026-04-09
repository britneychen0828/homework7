from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Iterable


class PipelineValidationError(RuntimeError):
    """Raised when required local inputs or tools are missing."""


def require_file(path: Path, description: str) -> None:
    if not path.exists():
        raise PipelineValidationError(f"Missing {description}: {path}")


def require_openai_key(api_key: str | None, stage_name: str) -> None:
    if not api_key:
        raise PipelineValidationError(
            f"{stage_name} requires OPENAI_API_KEY. Export it before running the pipeline."
        )


def require_binary(binary_name: str) -> None:
    if shutil.which(binary_name) is None:
        raise PipelineValidationError(
            f"Required executable '{binary_name}' was not found on PATH."
        )


def require_text_provider_credentials(provider: str, *, openai_api_key: str | None, gemini_api_key: str | None) -> None:
    if provider == "openai" and not openai_api_key:
        raise PipelineValidationError("Text provider 'openai' requires OPENAI_API_KEY.")
    if provider == "gemini" and not gemini_api_key:
        raise PipelineValidationError("Text provider 'gemini' requires GEMINI_API_KEY or GOOGLE_API_KEY.")


def require_tts_provider_support(provider: str, *, openai_api_key: str | None) -> None:
    if provider == "openai" and not openai_api_key:
        raise PipelineValidationError("TTS provider 'openai' requires OPENAI_API_KEY.")
    if provider == "gemini":
        return
    if provider == "say":
        if sys.platform != "darwin":
            raise PipelineValidationError("TTS provider 'say' is only available on macOS.")
        require_binary("say")


def require_matching_counts(slides: Iterable[object], audio_files: Iterable[object]) -> None:
    slide_count = len(list(slides))
    audio_count = len(list(audio_files))
    if slide_count != audio_count:
        raise PipelineValidationError(
            f"Slide/audio mismatch: {slide_count} slides but {audio_count} audio files."
        )
