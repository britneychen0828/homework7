"""Fix dialogue typed as action: deterministic merge, optional LLM polish (tqdm)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pydantic_ai import Agent
from tqdm import tqdm

from awesome_o.model_settings import resolve_default_model
from awesome_o.models.premise import PremiseDocument
from awesome_o.models.scene import Scene, ScenesDocument
from awesome_o.persona import SCENE_DIALOGUE_FIX_SYSTEM_PROMPT
from awesome_o.scene_dialogue_normalize import (
    normalize_scene_dialogue_blocks,
    scene_has_quoted_action_block,
)

DEFAULT_TEST_PROJECT = Path("projects/premise_20260401_221032")


def _load_premise_for_project_id(project_dir: Path) -> PremiseDocument:
    path = project_dir / "premise.json"
    if not path.is_file():
        raise FileNotFoundError(f"No premise.json at {path}")
    return PremiseDocument.model_validate_json(path.read_text(encoding="utf-8"))


def _load_scenes(project_dir: Path, premise: PremiseDocument) -> ScenesDocument:
    path = project_dir / "scenes.json"
    if not path.is_file():
        raise FileNotFoundError(f"No scenes.json at {path}")
    doc = ScenesDocument.model_validate_json(path.read_text(encoding="utf-8"))
    if doc.project_id != premise.project_id:
        doc = doc.model_copy(update={"project_id": premise.project_id})
    return doc


def _save_scenes(project_dir: Path, doc: ScenesDocument) -> None:
    path = project_dir / "scenes.json"
    out = doc.model_dump()
    out["slug_count"] = len(doc.scenes)
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run_fix(
    project_dir: Path,
    *,
    llm_polish: bool,
    no_progress: bool,
) -> None:
    project_dir = project_dir.resolve()
    premise = _load_premise_for_project_id(project_dir)
    doc = _load_scenes(project_dir, premise)

    new_scenes = [normalize_scene_dialogue_blocks(s) for s in doc.scenes]
    merged_any = any(
        a.model_dump() != b.model_dump() for a, b in zip(doc.scenes, new_scenes, strict=True)
    )
    doc = doc.model_copy(update={"scenes": new_scenes})
    if merged_any:
        _save_scenes(project_dir, doc)
        print("Applied deterministic cue+dialogue merges; saved scenes.json.")

    need_llm = [s for s in doc.scenes if scene_has_quoted_action_block(s)]
    if not need_llm:
        print("No quoted-in-action blocks left; done.")
        return

    print(
        f"{len(need_llm)} scene(s) still have dialogue-looking action blocks "
        + ("— run with --llm to polish, or edit manually." if not llm_polish else "")
    )

    if not llm_polish:
        return

    model = resolve_default_model()
    agent = Agent(
        model,
        system_prompt=SCENE_DIALOGUE_FIX_SYSTEM_PROMPT,
        output_type=Scene,
        output_retries=3,
    )

    id_to_idx = {s.id: i for i, s in enumerate(doc.scenes)}
    for sc in tqdm(need_llm, desc="LLM dialogue fix", unit="scene", disable=no_progress):
        idx = id_to_idx[sc.id]
        blob = (
            "Rewrite blocks[] only into valid action/dialogue typing. "
            "Preserve story and order of beats.\n\n"
            "SCENE_JSON:\n"
            f"{json.dumps(sc.model_dump(), indent=2, ensure_ascii=False)}"
        )
        result = agent.run_sync(blob)
        out = result.output
        out = out.model_copy(
            update={
                "id": sc.id,
                "slug_id": sc.slug_id,
                "slugline": sc.slugline,
                "line_start": sc.line_start,
                "sequence_id": sc.sequence_id,
            }
        )
        out = normalize_scene_dialogue_blocks(out)
        scenes = list(doc.scenes)
        scenes[idx] = out
        doc = doc.model_copy(update={"scenes": scenes})
        _save_scenes(project_dir, doc)

    print(f"LLM-polished {len(need_llm)} scene(s); saved scenes.json.")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Fix dialogue stored as action (deterministic + optional LLM).",
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=DEFAULT_TEST_PROJECT,
        help=f"Project folder (default: {DEFAULT_TEST_PROJECT})",
    )
    parser.add_argument(
        "--llm",
        action="store_true",
        help="After deterministic merge, call the model on scenes that still have quoted action.",
    )
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm.")
    args = parser.parse_args(argv)

    try:
        run_fix(args.project, llm_polish=args.llm, no_progress=args.no_progress)
    except (FileNotFoundError, RuntimeError) as e:
        print(f"Error: {e}")
        raise SystemExit(1) from e

