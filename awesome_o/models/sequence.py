"""sequence.json — ~10–15 min blocks (Sequence agent appends one StorySequence at a time)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


def _normalize_locations(v: Any) -> list[str]:
    if v is None:
        return []
    if isinstance(v, str):
        return [v]
    if isinstance(v, list):
        return [str(x) for x in v]
    return [str(v)]


class StorySequence(BaseModel):
    """
    One row in sequence.json `sequences[]`.
    Named StorySequence to avoid clashing with typing.Sequence.
    """

    id: str = Field(description="Stable id, e.g. SEQ1, SEQ2.")
    title: str
    act: int = Field(ge=1)
    goal: str
    emotional_tone: str
    key_locations: list[str] = Field(default_factory=list)
    exit_beat: str = Field(description="Handoff into the next sequence.")

    @field_validator("key_locations", mode="before")
    @classmethod
    def coerce_locations(cls, v: Any) -> list[str]:
        return _normalize_locations(v)


class SequenceDocument(BaseModel):
    """Full sequence.json."""

    project_id: str
    sequence_count: int | None = Field(
        default=None,
        description="Optional denormalized count; may omit when growing file.",
    )
    note: str | None = None
    sequences: list[StorySequence] = Field(default_factory=list)
