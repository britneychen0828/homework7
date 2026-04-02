"""premise.json — story contract (Premise agent output, downstream global input)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class StorySetting(BaseModel):
    """When / where the story lives."""

    primary_time: str = Field(description="Era, story present vs frame device, etc.")
    primary_places: list[str] = Field(
        default_factory=list,
        description="Key recurring locations (not slug-level).",
    )


class StoryCharacter(BaseModel):
    """Cast entry used in premise (protagonist, ally, or antagonist)."""

    name: str
    role: str
    want: str | None = Field(
        default=None,
        description="What they pursue; optional for flat antagonist blurbs.",
    )


class ProjectFolderNaming(BaseModel):
    """LLM output for /generate: a poster-style title used to build the projects/ folder name."""

    working_title: str = Field(
        description=(
            "One distinctive movie-style title (typically 2–8 words). "
            "Not a file path; no slashes. Like a title you'd see on a poster."
        ),
    )


class PremiseDocument(BaseModel):
    """
    Full premise.json shape. Agents below the stack treat this as read-only global context
    once authored (entire file passed each call).
    """

    project_id: str
    title: str
    writers: list[str] = Field(default_factory=list)
    source_script: str | None = Field(
        default=None,
        description="Optional tie to an imported screenplay file.",
    )
    format_notes: str | None = None
    logline: str
    genre: list[str] = Field(default_factory=list)
    themes: list[str] = Field(default_factory=list)
    central_conflict: str
    setting: StorySetting
    protagonists: list[StoryCharacter] = Field(default_factory=list)
    key_antagonist: StoryCharacter
    key_ally: StoryCharacter
    stakes: str
