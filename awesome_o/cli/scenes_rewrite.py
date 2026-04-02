"""Rewrite scenes.json → scenes_rewrite.json using premise, arc, sequence, and local context."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pydantic_ai import Agent
from tqdm import tqdm

from awesome_o.model_settings import resolve_default_model
from awesome_o.models.arc import ArcDocument
from awesome_o.models.premise import PremiseDocument
from awesome_o.models.scene import DialogueBlock, Scene, ScenesDocument
from awesome_o.models.sequence import SequenceDocument, StorySequence
from awesome_o.persona import REWRITE_SCENE_SYSTEM_PROMPT
from awesome_o.scene_dialogue_normalize import normalize_scene_dialogue_blocks

DEFAULT_TEST_PROJECT = Path("projects/premise_20260401_221032")
DEFAULT_PRIOR_SEQ_TAIL = 8
SCENES_SOURCE = "scenes.json"
SCENES_REWRITE_OUT = "scenes_rewrite.json"


def _load_premise(project_dir: Path) -> PremiseDocument:
    path = project_dir / "premise.json"
    if not path.is_file():
        raise FileNotFoundError(f"No premise.json at {path}")
    return PremiseDocument.model_validate_json(path.read_text(encoding="utf-8"))


def _load_arc(project_dir: Path) -> ArcDocument:
    path = project_dir / "arc.json"
    if not path.is_file():
        raise FileNotFoundError(f"No arc.json at {path}")
    return ArcDocument.model_validate_json(path.read_text(encoding="utf-8"))


def _load_sequence(project_dir: Path, premise: PremiseDocument) -> SequenceDocument:
    path = project_dir / "sequence.json"
    if not path.is_file():
        raise FileNotFoundError(f"No sequence.json at {path}")
    doc = SequenceDocument.model_validate_json(path.read_text(encoding="utf-8"))
    if doc.project_id != premise.project_id:
        doc = doc.model_copy(update={"project_id": premise.project_id})
    return doc


def _load_source_scenes(project_dir: Path, premise: PremiseDocument) -> ScenesDocument:
    path = project_dir / SCENES_SOURCE
    if not path.is_file():
        raise FileNotFoundError(f"No {SCENES_SOURCE} at {path}")
    doc = ScenesDocument.model_validate_json(path.read_text(encoding="utf-8"))
    if doc.project_id != premise.project_id:
        doc = doc.model_copy(update={"project_id": premise.project_id})
    if not doc.scenes:
        raise ValueError(f"{SCENES_SOURCE} has no scenes.")
    return doc


def _load_or_init_rewrite_doc(project_dir: Path, premise: PremiseDocument) -> ScenesDocument:
    path = project_dir / SCENES_REWRITE_OUT
    if not path.is_file():
        return ScenesDocument(
            project_id=premise.project_id,
            source_script=None,
            note="Rewritten pass from scenes.json (Awesome-O scenes_rewrite).",
            scenes=[],
        )
    doc = ScenesDocument.model_validate_json(path.read_text(encoding="utf-8"))
    if doc.project_id != premise.project_id:
        doc = doc.model_copy(update={"project_id": premise.project_id})
    return doc


def _save_rewrite_doc(project_dir: Path, doc: ScenesDocument) -> None:
    path = project_dir / SCENES_REWRITE_OUT
    out = doc.model_dump()
    out["slug_count"] = len(doc.scenes)
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _prior_sequence_id(sequences: list[StorySequence], current_id: str) -> str | None:
    ids = [s.id for s in sequences]
    try:
        i = ids.index(current_id)
    except ValueError:
        return None
    return ids[i - 1] if i > 0 else None


def _sequence_row(seq_doc: SequenceDocument, sequence_id: str) -> StorySequence:
    for s in seq_doc.sequences:
        if s.id == sequence_id:
            return s
    raise ValueError(f"No sequence row {sequence_id!r} in sequence.json")


def _same_seq_prior_rewritten(
    rewritten_so_far: list[Scene],
    current: Scene,
) -> list[Scene]:
    return [s for s in rewritten_so_far if s.sequence_id == current.sequence_id]


def _prior_seq_tail_rewritten(
    rewritten_so_far: list[Scene],
    prior_seq_id: str | None,
    limit: int,
) -> list[Scene]:
    if not prior_seq_id or limit <= 0:
        return []
    tail = [s for s in rewritten_so_far if s.sequence_id == prior_seq_id]
    return tail[-limit:]


def run_rewrite_batch(
    project_dir: Path,
    *,
    prior_seq_tail: int,
    note: str | None,
    no_progress: bool,
) -> None:
    project_dir = project_dir.resolve()
    premise = _load_premise(project_dir)
    arc = _load_arc(project_dir)
    seq_doc = _load_sequence(project_dir, premise)
    source = _load_source_scenes(project_dir, premise)
    out_doc = _load_or_init_rewrite_doc(project_dir, premise)

    if note is not None:
        out_doc = out_doc.model_copy(update={"note": note})

    done_ids = {s.id for s in out_doc.scenes}
    pending = [s for s in source.scenes if s.id not in done_ids]

    if not pending:
        print(f"All {len(source.scenes)} scene(s) already in {SCENES_REWRITE_OUT}. Nothing to do.")
        return

    model = resolve_default_model()
    agent = Agent(
        model,
        system_prompt=REWRITE_SCENE_SYSTEM_PROMPT,
        output_type=Scene,
        output_retries=4,
    )

    premise_json = json.dumps(premise.model_dump(), indent=2, ensure_ascii=False)
    arc_json = json.dumps(arc.model_dump(), indent=2, ensure_ascii=False)
    seq_json = json.dumps(seq_doc.model_dump(), indent=2, ensure_ascii=False)

    rewritten = list(out_doc.scenes)

    for orig in tqdm(pending, desc="Rewrite", unit="scene", disable=no_progress):
        prior_sid = _prior_sequence_id(seq_doc.sequences, orig.sequence_id)
        tail_prev = _prior_seq_tail_rewritten(rewritten, prior_sid, prior_seq_tail)
        same_prior = _same_seq_prior_rewritten(rewritten, orig)
        row_sq = _sequence_row(seq_doc, orig.sequence_id)

        draft_json = json.dumps(orig.model_dump(), indent=2, ensure_ascii=False)
        n_blocks = len(orig.blocks)
        n_dialogue = sum(1 for b in orig.blocks if isinstance(b, DialogueBlock))

        blob = (
            f"draft_stats: total_blocks={n_blocks}, dialogue_blocks={n_dialogue}\n"
            "(If many blocks but almost no dialogue and a character drives the scene, "
            "add appropriate spoken beats per system prompt.)\n\n"
            "PREMISE:\n"
            f"{premise_json}\n\n"
            "ARC:\n"
            f"{arc_json}\n\n"
            "SEQUENCE_DOCUMENT (sequence.json):\n"
            f"{seq_json}\n\n"
            "CURRENT_SEQUENCE_ROW:\n"
            f"{json.dumps(row_sq.model_dump(), indent=2, ensure_ascii=False)}\n\n"
            "PRIOR_SCENES_SAME_SEQUENCE (already rewritten, earlier in this SEQ only):\n"
            f"{json.dumps([s.model_dump() for s in same_prior], indent=2, ensure_ascii=False)}\n\n"
            "PRIOR_SEQUENCE_TAIL (last scenes from previous SEQ, rewritten; may be empty):\n"
            f"{json.dumps([s.model_dump() for s in tail_prev], indent=2, ensure_ascii=False)}\n\n"
            "DRAFT_SCENE_TO_REWRITE:\n"
            f"{draft_json}\n\n"
            "Return the improved Scene with identical id, slug_id, slugline, line_start, "
            "sequence_id."
        )

        result = agent.run_sync(blob)
        new_scene = result.output
        new_scene = new_scene.model_copy(
            update={
                "id": orig.id,
                "slug_id": orig.slug_id,
                "slugline": orig.slugline,
                "line_start": orig.line_start,
                "sequence_id": orig.sequence_id,
            }
        )
        new_scene = normalize_scene_dialogue_blocks(new_scene)
        rewritten.append(new_scene)
        out_doc = out_doc.model_copy(update={"scenes": rewritten, "slug_count": len(rewritten)})
        _save_rewrite_doc(project_dir, out_doc)

    print(f"Wrote {len(pending)} scene(s) → {project_dir / SCENES_REWRITE_OUT}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            f"Rewrite {SCENES_SOURCE} using premise, arc, sequence.json → {SCENES_REWRITE_OUT}."
        ),
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=DEFAULT_TEST_PROJECT,
        help=f"Project folder (default: {DEFAULT_TEST_PROJECT})",
    )
    parser.add_argument(
        "--prior-seq-tail",
        type=int,
        default=DEFAULT_PRIOR_SEQ_TAIL,
        help=(
            "Max rewritten scenes to include from the previous sequence (default: "
            f"{DEFAULT_PRIOR_SEQ_TAIL})."
        ),
    )
    parser.add_argument(
        "--note",
        type=str,
        default=None,
        help=f"Override note field in {SCENES_REWRITE_OUT}.",
    )
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm.")
    args = parser.parse_args(argv)

    if args.prior_seq_tail < 0:
        parser.error("--prior-seq-tail cannot be negative")

    try:
        run_rewrite_batch(
            args.project,
            prior_seq_tail=args.prior_seq_tail,
            note=args.note,
            no_progress=args.no_progress,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"Error: {e}")
        raise SystemExit(1) from e
