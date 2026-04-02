"""scenes.json rows — Scene agent fills blocks[] (action + dialogue)."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field


class ActionBlock(BaseModel):
    kind: Literal["action"] = "action"
    text: str


class DialogueBlock(BaseModel):
    kind: Literal["dialogue"] = "dialogue"
    character: str
    lines: list[str] = Field(default_factory=list)
    parenthetical: str | None = None


SceneBlock = Annotated[ActionBlock | DialogueBlock, Field(discriminator="kind")]


class ScenesDocument(BaseModel):
    """Full scenes.json envelope (optional metadata + all rows)."""

    project_id: str
    source_script: str | None = None
    note: str | None = None
    slug_count: int | None = None
    scenes: list[Scene] = Field(default_factory=list)


class Scene(BaseModel):
    """
    One scene / slug row in scenes.json.
    `line_start` is optional (script extraction); generation can omit until merge.
    """

    id: str = Field(description="Stable id, e.g. S0001.")
    slug_id: str = Field(description="Shooting-script style slug id, e.g. 8A.")
    slugline: str
    line_start: int | None = None
    sequence_id: str = Field(description="e.g. SEQ1")
    blocks: list[SceneBlock] = Field(default_factory=list)
