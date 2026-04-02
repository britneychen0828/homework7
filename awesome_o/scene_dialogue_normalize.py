"""Deterministic fixes: character cue + quoted line wrongly stored as two action blocks."""

from __future__ import annotations

from awesome_o.models.scene import ActionBlock, DialogueBlock, Scene, SceneBlock

MAX_CHARACTER_CUE_CHARS = 48


def _looks_like_outer_quoted_speech(text: str) -> bool:
    t = text.strip()
    return bool(t) and t[0] in ('"', "\u201c", "'")


def _strip_outer_quotes(text: str) -> str:
    t = text.strip()
    pairs = (('"', '"'), ("\u201c", "\u201d"), ("'", "'"))
    for open_q, close_q in pairs:
        if len(t) >= 2 and t.startswith(open_q) and t.endswith(close_q):
            return t[1:-1].strip()
    return t


def _split_dialogue_lines(body: str) -> list[str]:
    parts = [p.strip() for p in body.split("\n") if p.strip()]
    return parts if parts else ([body.strip()] if body.strip() else [""])


def _character_from_cue(cue: str) -> str | None:
    cue = cue.strip().rstrip(".")
    if not cue:
        return None
    if "\n" in cue:
        return None
    if len(cue) > MAX_CHARACTER_CUE_CHARS:
        return None
    if _looks_like_outer_quoted_speech(cue):
        return None
    # Skip prose masquerading as cue (many lowercase words)
    words = cue.replace(",", " ").split()
    lowerish = sum(1 for w in words if w and w[0].islower())
    if len(words) >= 4 and lowerish >= len(words) // 2:
        return None
    return cue.upper()


def try_merge_action_cue_and_line(a0: ActionBlock, a1: ActionBlock) -> DialogueBlock | None:
    """If a0 is a short character cue and a1 is quoted speech, merge into DialogueBlock."""
    t0 = a0.text.strip()
    t1 = a1.text.strip()
    if "\n" in t0:
        first = t0.split("\n", 1)[0].strip()
        if len(t0) > MAX_CHARACTER_CUE_CHARS:
            return None
        t0 = first
    char = _character_from_cue(t0)
    if char is None:
        return None
    if not _looks_like_outer_quoted_speech(t1):
        return None
    body = _strip_outer_quotes(t1)
    lines = _split_dialogue_lines(body)
    return DialogueBlock(character=char, lines=lines)


def normalize_scene_dialogue_blocks(scene: Scene) -> Scene:
    """Merge cue+quoted-action pairs until stable."""
    blocks: list[SceneBlock] = list(scene.blocks)
    changed = True
    while changed:
        changed = False
        new_blocks: list[SceneBlock] = []
        i = 0
        while i < len(blocks):
            if i + 1 < len(blocks):
                b0, b1 = blocks[i], blocks[i + 1]
                if isinstance(b0, ActionBlock) and isinstance(b1, ActionBlock):
                    merged = try_merge_action_cue_and_line(b0, b1)
                    if merged is not None:
                        new_blocks.append(merged)
                        i += 2
                        changed = True
                        continue
            new_blocks.append(blocks[i])
            i += 1
        blocks = new_blocks
    return scene.model_copy(update={"blocks": blocks})


def scene_has_quoted_action_block(scene: Scene) -> bool:
    """True if any action block still looks like spoken line in action clothing."""
    for b in scene.blocks:
        if isinstance(b, ActionBlock) and _looks_like_outer_quoted_speech(b.text):
            return True
    return False
