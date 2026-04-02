"""arc.json — act spine (Arc agent appends one Act at a time)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Act(BaseModel):
    """Single act object inside arc.json `acts[]`."""

    act: int = Field(ge=1, description="1-based act index.")
    label: str = Field(description="Short act name, e.g. Setup / confrontation.")
    approx_pages_note: str | None = Field(
        default=None,
        description="Optional pacing note; not required for agents.",
    )
    dramatic_question: str | None = Field(
        default=None,
        description="What the audience is asking during this act.",
    )
    summary: str
    turning_point_end: str | None = Field(
        default=None,
        description="How the act resolves / launches the next.",
    )
    must_hit_beats: list[str] = Field(
        default_factory=list,
        description="Non-negotiable story beats in this act.",
    )
    midpoint_note: str | None = Field(
        default=None,
        description="Act-two midpoint pivot, if applicable.",
    )


class ArcDocument(BaseModel):
    """Full arc.json on disk after all acts are written."""

    project_id: str
    structure_model: str = Field(
        default="three_act_feature",
        description="Label for shape: three_act, five_act_tv, etc.",
    )
    acts: list[Act] = Field(default_factory=list)
    series_thread: str | None = Field(
        default=None,
        description="Sequel / franchise continuity note.",
    )
