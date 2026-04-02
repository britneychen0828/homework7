"""scene_plan.json — per-sequence scene counts (adaptive scene batch)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class SequenceSceneAllocation(BaseModel):
    """How many scenes to generate for one sequence row."""

    sequence_id: str = Field(description="Must match StorySequence.id, e.g. SEQ2.")
    scene_count: int = Field(ge=1, le=32, description="Planned scenes for this sequence.")
    rationale: str | None = Field(
        default=None,
        description="Short reason (pacing, set-pieces, breathers).",
    )


class ScenePlanDocument(BaseModel):
    """LLM planner output; saved to disk for reproducible resumes."""

    project_id: str
    allocations: list[SequenceSceneAllocation]
    plan_note: str | None = Field(
        default=None,
        description="Optional overall note (e.g. adaptive vs fixed).",
    )
