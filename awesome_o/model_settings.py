"""Resolve LLM id for Awesome-O (Gemini via GEMINI_API_KEY in .env)."""

from __future__ import annotations

import os

from dotenv import load_dotenv

DEFAULT_GEMINI_MODEL = "google-gla:gemini-2.5-flash"


def resolve_default_model() -> str:
    """
    Pick a model string for pydantic-ai.

    Loads variables from a local ``.env`` if present (``GEMINI_API_KEY``, etc.).

    Precedence:
    1. ``AWESOME_O_MODEL`` if set (full provider id).
    2. Otherwise ``DEFAULT_GEMINI_MODEL`` when ``GEMINI_API_KEY`` or ``GOOGLE_API_KEY`` is set
       after loading dotenv.
    """
    load_dotenv()

    override = os.environ.get("AWESOME_O_MODEL", "").strip()
    if override:
        return override

    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        return DEFAULT_GEMINI_MODEL

    raise RuntimeError(
        "No Gemini API key found. Put GEMINI_API_KEY in a .env file in the current working "
        "directory (or export it), or set AWESOME_O_MODEL to a full provider id."
    )
