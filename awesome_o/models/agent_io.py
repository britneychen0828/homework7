"""
Explicit *agent call* payloads — what you assemble before invoking an LLM.

On-disk documents (PremiseDocument, ArcDocument, …) stay the source of truth;
these types document how slices of them combine for each step.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from awesome_o.models.arc import Act, ArcDocument
from awesome_o.models.premise import PremiseDocument
from awesome_o.models.scene import Scene
from awesome_o.models.sequence import SequenceDocument, StorySequence


class PremiseAgentInputs(BaseModel):
    """Premise agent: first in the pipeline."""

    seed_brief: str = Field(description="Unstructured pitch / bullets / comps from the human.")
    program_constraints: str | None = Field(
        default=None,
        description="Optional fixed rails: rating, genre rules, assignment rubric.",
    )
    project_id: str = Field(description="Stable id for filenames and cross-refs.")


class ArcAgentInputs(BaseModel):
    """Arc agent: emit the next Act given premise + acts already committed."""

    premise: PremiseDocument
    prior_acts: list[Act] = Field(
        default_factory=list,
        description="Acts 1 … n−1 already appended to arc.json.",
    )
    next_act_number: int = Field(ge=1, description="The act index to generate now.")
    structure_model: str = Field(
        default="three_act_feature",
        description="Agreed shape so the model knows how many acts to plan toward.",
    )


class SequenceAgentInputs(BaseModel):
    """Sequence agent: emit the next StorySequence row."""

    premise: PremiseDocument
    arc: ArcDocument
    prior_sequences: list[StorySequence] = Field(
        default_factory=list,
        description="SEQ1 … SEQn−1 already in sequence.json.",
    )
    next_sequence_id: str = Field(description="Stable id to assign, e.g. SEQ3.")


class SceneAgentInputs(BaseModel):
    """Scene agent: emit the next Scene (or revise) for the current sequence."""

    premise: PremiseDocument
    arc: ArcDocument
    sequences: SequenceDocument
    current_sequence_id: str = Field(description="SEQ being drafted.")
    scenes_from_previous_sequence: list[Scene] = Field(
        default_factory=list,
        description="Continuity tail: all rows whose sequence_id is the prior SEQ.",
    )
    target_slug_id: str | None = Field(
        default=None,
        description="If known from a beat sheet; else agent proposes slugline + id.",
    )
