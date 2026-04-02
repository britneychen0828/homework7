"""
Awesome-O: JSON-first screenplay pipeline (premise → arc → sequences → scenes).

CLI lives in ``awesome_o.cli``; data contracts in ``awesome_o.models``.
"""

from awesome_o.models import (
    Act,
    ActionBlock,
    ArcAgentInputs,
    ArcDocument,
    DialogueBlock,
    PremiseAgentInputs,
    PremiseDocument,
    Scene,
    SceneAgentInputs,
    SceneBlock,
    ScenePlanDocument,
    ScenesDocument,
    SequenceAgentInputs,
    SequenceDocument,
    SequenceSceneAllocation,
    StoryCharacter,
    StorySequence,
    StorySetting,
)

__all__ = [
    "Act",
    "ActionBlock",
    "ArcAgentInputs",
    "ArcDocument",
    "DialogueBlock",
    "PremiseAgentInputs",
    "PremiseDocument",
    "Scene",
    "SceneAgentInputs",
    "SceneBlock",
    "ScenePlanDocument",
    "ScenesDocument",
    "SequenceAgentInputs",
    "SequenceDocument",
    "SequenceSceneAllocation",
    "StoryCharacter",
    "StorySequence",
    "StorySetting",
]
