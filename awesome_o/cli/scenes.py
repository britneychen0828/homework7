"""Automated scene agent: appends Scene rows to scenes.json (tqdm progress)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pydantic_ai import Agent
from tqdm import tqdm

from awesome_o.model_settings import resolve_default_model
from awesome_o.models.arc import ArcDocument
from awesome_o.models.premise import PremiseDocument
from awesome_o.models.scene import Scene, ScenesDocument
from awesome_o.models.scene_plan import ScenePlanDocument, SequenceSceneAllocation
from awesome_o.models.sequence import SequenceDocument, StorySequence
from awesome_o.persona import SCENE_PLAN_SYSTEM_PROMPT, SCENE_STRUCT_SYSTEM_PROMPT

DEFAULT_TEST_PROJECT = Path("projects/premise_20260401_221032")
DEFAULT_SCENES_PER_SEQUENCE = 4
DEFAULT_PRIOR_TAIL = 12
DEFAULT_SCENE_MIN = 2
DEFAULT_SCENE_MAX = 8
SCENE_PLAN_FILENAME = "scene_plan.json"


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


def _load_or_init_scenes(project_dir: Path, premise: PremiseDocument) -> ScenesDocument:
    path = project_dir / "scenes.json"
    if not path.is_file():
        return ScenesDocument(project_id=premise.project_id, scenes=[])
    doc = ScenesDocument.model_validate_json(path.read_text(encoding="utf-8"))
    if doc.project_id != premise.project_id:
        doc = doc.model_copy(update={"project_id": premise.project_id})
    return doc


def _save_scenes(project_dir: Path, doc: ScenesDocument) -> None:
    path = project_dir / "scenes.json"
    out = doc.model_dump()
    out["slug_count"] = len(doc.scenes)
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _counts_by_sequence(scenes: list[Scene]) -> dict[str, int]:
    d: dict[str, int] = {}
    for s in scenes:
        d[s.sequence_id] = d.get(s.sequence_id, 0) + 1
    return d


def _prior_sequence_id(sequences: list[StorySequence], current_id: str) -> str | None:
    ids = [s.id for s in sequences]
    try:
        i = ids.index(current_id)
    except ValueError:
        return None
    return ids[i - 1] if i > 0 else None


def _prior_tail(
    scenes: list[Scene],
    prior_seq_id: str | None,
    limit: int,
) -> list[Scene]:
    if not prior_seq_id:
        return []
    tail = [s for s in scenes if s.sequence_id == prior_seq_id]
    return tail[-limit:] if limit else tail


def _apply_plan_clamps(
    plan: ScenePlanDocument,
    sequences: list[StorySequence],
    min_c: int,
    max_c: int,
) -> dict[str, int]:
    ids = [s.id for s in sequences]
    by_id = {a.sequence_id: a for a in plan.allocations}
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise ValueError(f"scene_plan missing sequence_ids: {missing}")
    extra = set(by_id) - set(ids)
    if extra:
        raise ValueError(f"scene_plan has unknown sequence_ids: {sorted(extra)}")
    return {i: max(min_c, min(max_c, by_id[i].scene_count)) for i in ids}


def _save_scene_plan(
    project_dir: Path,
    premise: PremiseDocument,
    sequences: list[StorySequence],
    targets: dict[str, int],
    rationale: dict[str, str | None],
    plan_note: str | None,
) -> None:
    path = project_dir / SCENE_PLAN_FILENAME
    alloc = [
        SequenceSceneAllocation(
            sequence_id=sid,
            scene_count=targets[sid],
            rationale=rationale.get(sid),
        )
        for sid in [s.id for s in sequences]
    ]
    doc = ScenePlanDocument(
        project_id=premise.project_id,
        allocations=alloc,
        plan_note=plan_note,
    )
    path.write_text(
        json.dumps(doc.model_dump(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _run_scene_planner(
    premise: PremiseDocument,
    arc: ArcDocument,
    seq_doc: SequenceDocument,
    min_c: int,
    max_c: int,
) -> ScenePlanDocument:
    model = resolve_default_model()
    planner = Agent(
        model,
        system_prompt=SCENE_PLAN_SYSTEM_PROMPT,
        output_type=ScenePlanDocument,
        output_retries=3,
    )
    ordered = [s.id for s in seq_doc.sequences]
    premise_json = json.dumps(premise.model_dump(), indent=2, ensure_ascii=False)
    arc_json = json.dumps(arc.model_dump(), indent=2, ensure_ascii=False)
    seq_json = json.dumps(seq_doc.model_dump(), indent=2, ensure_ascii=False)
    blob = (
        f"project_id: {premise.project_id}\n"
        f"min_scene_count: {min_c}\n"
        f"max_scene_count: {max_c}\n"
        f"required_sequence_ids_in_order: {ordered}\n\n"
        "You must return exactly one allocation per id above, in the same order.\n\n"
        "SEQUENCE_DOCUMENT:\n"
        f"{seq_json}\n\n"
        "PREMISE:\n"
        f"{premise_json}\n\n"
        "ARC:\n"
        f"{arc_json}\n\n"
        "Produce ScenePlanDocument."
    )
    result = planner.run_sync(blob)
    plan = result.output
    if plan.project_id != premise.project_id:
        plan = plan.model_copy(update={"project_id": premise.project_id})
    return plan


def _build_tasks(
    sequences: list[StorySequence],
    targets: dict[str, int],
    counts: dict[str, int],
) -> list[tuple[StorySequence, int, int]]:
    """
    Each item: (story_sequence, scene_index_1based_within_seq, total_scenes_for_seq).
    """
    tasks: list[tuple[StorySequence, int, int]] = []
    for sq in sequences:
        target = targets[sq.id]
        have = counts.get(sq.id, 0)
        for k in range(have + 1, target + 1):
            tasks.append((sq, k, target))
    return tasks


def _warn_extra_scenes(counts: dict[str, int], targets: dict[str, int]) -> None:
    for sid, t in targets.items():
        h = counts.get(sid, 0)
        if h > t:
            print(
                f"Note: {sid} has {h} scene(s) on disk but target is {t}; "
                "extras are kept (not deleted)."
            )


def run_scenes_batch(
    project_dir: Path,
    targets_by_sequence: dict[str, int],
    *,
    prior_tail: int,
    note: str | None,
    no_progress: bool,
) -> None:
    project_dir = project_dir.resolve()
    premise = _load_premise(project_dir)
    arc = _load_arc(project_dir)
    seq_doc = _load_sequence(project_dir, premise)
    if not seq_doc.sequences:
        raise ValueError("sequence.json has no sequences; run the sequence agent first.")

    scenes_doc = _load_or_init_scenes(project_dir, premise)
    counts = _counts_by_sequence(scenes_doc.scenes)
    _warn_extra_scenes(counts, targets_by_sequence)

    tasks = _build_tasks(seq_doc.sequences, targets_by_sequence, counts)

    if not tasks:
        total_slots = sum(targets_by_sequence[s.id] for s in seq_doc.sequences)
        print(
            f"Already have enough scenes for all targets ({total_slots} slots across "
            f"{len(seq_doc.sequences)} sequences). Nothing to do."
        )
        return

    if note is not None:
        scenes_doc = scenes_doc.model_copy(update={"note": note})

    model = resolve_default_model()
    agent = Agent(
        model,
        system_prompt=SCENE_STRUCT_SYSTEM_PROMPT,
        output_type=Scene,
        output_retries=4,
    )

    premise_json = json.dumps(premise.model_dump(), indent=2, ensure_ascii=False)
    arc_json = json.dumps(arc.model_dump(), indent=2, ensure_ascii=False)
    seq_full_json = json.dumps(seq_doc.model_dump(), indent=2, ensure_ascii=False)

    for sq, k, total_for_seq in tqdm(
        tasks,
        desc="Scenes",
        unit="scene",
        disable=no_progress,
    ):
        prior_sid = _prior_sequence_id(seq_doc.sequences, sq.id)
        continuity = _prior_tail(scenes_doc.scenes, prior_sid, prior_tail)
        continuity_json = json.dumps(
            [s.model_dump() for s in continuity],
            indent=2,
            ensure_ascii=False,
        )
        same_seq_prior = [s for s in scenes_doc.scenes if s.sequence_id == sq.id]
        same_seq_json = json.dumps(
            [s.model_dump() for s in same_seq_prior],
            indent=2,
            ensure_ascii=False,
        )

        next_num = len(scenes_doc.scenes) + 1
        stable_id = f"S{next_num:04d}"
        slug_id = f"{sq.id}-{k:02d}"
        is_last_in_seq = k == total_for_seq

        last_hint = (
            "This is the LAST scene for this sequence—land a beat that hands off to the "
            "sequence's exit_beat and sets up the next sequence."
            if is_last_in_seq
            else "Leave room for later scenes in this sequence; do not exhaust the "
            "sequence goal yet."
        )

        current_seq_json = json.dumps(sq.model_dump(), indent=2, ensure_ascii=False)

        blob = (
            f"next_scene_stable_id: {stable_id}\n"
            f"next_slug_id: {slug_id}\n"
            f"sequence_id: {sq.id}\n"
            f"scene_index_within_sequence: {k}\n"
            f"scenes_planned_for_this_sequence: {total_for_seq}\n"
            f"scene_writing_hint: {last_hint}\n\n"
            "PREMISE:\n"
            f"{premise_json}\n\n"
            "ARC:\n"
            f"{arc_json}\n\n"
            "ALL_SEQUENCES (context):\n"
            f"{seq_full_json}\n\n"
            "CURRENT_SEQUENCE_ROW:\n"
            f"{current_seq_json}\n\n"
            "PRIOR_SEQUENCE_SCENES (continuity from previous SEQ only; may be empty):\n"
            f"{continuity_json}\n\n"
            "SCENES_ALREADY_IN_THIS_SEQUENCE (do not repeat beats; continue forward):\n"
            f"{same_seq_json}\n\n"
            f"Produce the single Scene for {stable_id} ({slug_id})."
        )

        try:
            result = agent.run_sync(blob)
            row = result.output
        except Exception as e:  # noqa: BLE001
            print(f"\nScene generation failed at {stable_id}: {e}")
            raise

        if row.id != stable_id:
            row = row.model_copy(update={"id": stable_id})
        if row.slug_id != slug_id:
            row = row.model_copy(update={"slug_id": slug_id})
        if row.sequence_id != sq.id:
            row = row.model_copy(update={"sequence_id": sq.id})

        scenes_doc.scenes.append(row)
        scenes_doc = scenes_doc.model_copy(update={"slug_count": len(scenes_doc.scenes)})
        _save_scenes(project_dir, scenes_doc)

    print(f"Wrote {len(tasks)} scene(s) → {project_dir / 'scenes.json'}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate Scene rows into scenes.json (premise + arc + sequence required).",
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=DEFAULT_TEST_PROJECT,
        help=f"Project folder (default: {DEFAULT_TEST_PROJECT})",
    )
    parser.add_argument(
        "--per-sequence",
        type=int,
        default=DEFAULT_SCENES_PER_SEQUENCE,
        help=(
            f"Scenes per sequence when not using --adaptive (default: "
            f"{DEFAULT_SCENES_PER_SEQUENCE})."
        ),
    )
    parser.add_argument(
        "--adaptive",
        action="store_true",
        help=(
            "Ask the model once for scene counts per sequence; write scene_plan.json. "
            "Re-use saved plan on later runs unless --replan-scene-plan."
        ),
    )
    parser.add_argument(
        "--scene-min",
        type=int,
        default=DEFAULT_SCENE_MIN,
        help=f"Minimum scenes per sequence in adaptive mode (default: {DEFAULT_SCENE_MIN}).",
    )
    parser.add_argument(
        "--scene-max",
        type=int,
        default=DEFAULT_SCENE_MAX,
        help=f"Maximum scenes per sequence in adaptive mode (default: {DEFAULT_SCENE_MAX}).",
    )
    parser.add_argument(
        "--replan-scene-plan",
        action="store_true",
        help="With --adaptive, ignore existing scene_plan.json and plan again.",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="With --adaptive, only write scene_plan.json and exit (no scene generation).",
    )
    parser.add_argument(
        "--prior-tail",
        type=int,
        default=DEFAULT_PRIOR_TAIL,
        help=(
            "Max scenes from the previous sequence to pass for continuity "
            f"(default: {DEFAULT_PRIOR_TAIL})."
        ),
    )
    parser.add_argument("--note", type=str, default=None, help="Optional scenes.json note.")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm bar.")
    args = parser.parse_args(argv)

    if args.per_sequence < 1:
        parser.error("--per-sequence must be at least 1")
    if args.prior_tail < 0:
        parser.error("--prior-tail cannot be negative")
    if args.scene_min < 1:
        parser.error("--scene-min must be at least 1")
    if args.scene_max < args.scene_min:
        parser.error("--scene-max must be >= --scene-min")

    project_dir = args.project.resolve()

    try:
        premise = _load_premise(project_dir)
        seq_doc = _load_sequence(project_dir, premise)
        arc = _load_arc(project_dir)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        raise SystemExit(1) from e

    if args.adaptive:
        plan_path = project_dir / SCENE_PLAN_FILENAME
        if plan_path.is_file() and not args.replan_scene_plan:
            raw = ScenePlanDocument.model_validate_json(plan_path.read_text(encoding="utf-8"))
            if raw.project_id != premise.project_id:
                raw = raw.model_copy(update={"project_id": premise.project_id})
            targets = _apply_plan_clamps(raw, seq_doc.sequences, args.scene_min, args.scene_max)
            rationale = {a.sequence_id: a.rationale for a in raw.allocations}
            note_txt = raw.plan_note
        else:
            raw = _run_scene_planner(
                premise, arc, seq_doc, args.scene_min, args.scene_max
            )
            targets = _apply_plan_clamps(raw, seq_doc.sequences, args.scene_min, args.scene_max)
            rationale = {a.sequence_id: a.rationale for a in raw.allocations}
            note_txt = raw.plan_note
            _save_scene_plan(project_dir, premise, seq_doc.sequences, targets, rationale, note_txt)
            print(f"Wrote scene plan → {plan_path}")

        if args.plan_only:
            print("Targets per sequence:", json.dumps(targets, indent=2))
            return

        try:
            run_scenes_batch(
                args.project,
                targets,
                prior_tail=args.prior_tail,
                note=args.note,
                no_progress=args.no_progress,
            )
        except (FileNotFoundError, ValueError, RuntimeError) as e:
            print(f"Error: {e}")
            raise SystemExit(1) from e
        return

    if args.plan_only:
        parser.error("--plan-only requires --adaptive")

    targets = {s.id: args.per_sequence for s in seq_doc.sequences}
    try:
        run_scenes_batch(
            args.project,
            targets,
            prior_tail=args.prior_tail,
            note=args.note,
            no_progress=args.no_progress,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"Error: {e}")
        raise SystemExit(1) from e

