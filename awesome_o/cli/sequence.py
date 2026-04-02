"""Automated sequence agent: fills sequence.json one StorySequence at a time (tqdm progress)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pydantic_ai import Agent
from tqdm import tqdm

from awesome_o.model_settings import resolve_default_model
from awesome_o.models.arc import ArcDocument
from awesome_o.models.premise import PremiseDocument
from awesome_o.models.sequence import SequenceDocument, StorySequence
from awesome_o.persona import SEQUENCE_STRUCT_SYSTEM_PROMPT

DEFAULT_TEST_PROJECT = Path("projects/premise_20260401_221032")
DEFAULT_SEQUENCE_COUNT = 8


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


def _load_or_init_sequence(project_dir: Path, premise: PremiseDocument) -> SequenceDocument:
    path = project_dir / "sequence.json"
    if not path.is_file():
        return SequenceDocument(project_id=premise.project_id, sequences=[])
    doc = SequenceDocument.model_validate_json(path.read_text(encoding="utf-8"))
    if doc.project_id != premise.project_id:
        doc = doc.model_copy(update={"project_id": premise.project_id})
    return doc


def _save_sequence(project_dir: Path, doc: SequenceDocument) -> None:
    path = project_dir / "sequence.json"
    out = doc.model_dump()
    out["sequence_count"] = len(doc.sequences)
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run_sequence_batch(
    project_dir: Path,
    target_count: int,
    *,
    note: str | None,
    no_progress: bool,
) -> None:
    project_dir = project_dir.resolve()
    premise = _load_premise(project_dir)
    arc = _load_arc(project_dir)
    doc = _load_or_init_sequence(project_dir, premise)

    act_nums = sorted({a.act for a in arc.acts})
    if not act_nums:
        raise ValueError("arc.json has no acts; run the arc agent first.")

    existing = len(doc.sequences)
    need = target_count - existing
    if need <= 0:
        print(f"Already have {existing} sequences (target {target_count}). Nothing to do.")
        return

    model = resolve_default_model()
    agent = Agent(
        model,
        system_prompt=SEQUENCE_STRUCT_SYSTEM_PROMPT,
        output_type=StorySequence,
    )

    premise_json = json.dumps(premise.model_dump(), indent=2, ensure_ascii=False)
    arc_json = json.dumps(arc.model_dump(), indent=2, ensure_ascii=False)
    acts_hint = ", ".join(str(n) for n in act_nums)

    if note is not None:
        doc = doc.model_copy(update={"note": note})

    for _ in tqdm(range(need), desc="Sequences", unit="seq", disable=no_progress):
        n = len(doc.sequences) + 1
        next_id = f"SEQ{n}"
        prior = [s.model_dump() for s in doc.sequences]
        prior_json = json.dumps(prior, indent=2, ensure_ascii=False)
        is_last = n == target_count
        last_hint = (
            "This is the FINAL sequence in the planned batch; exit_beat should land "
            "the story's resolution or thematic closure."
            if is_last
            else "Leave a strong hook for the next sequence."
        )

        blob = (
            f"next_sequence_id: {next_id}\n"
            f"sequence_index_1based: {n}\n"
            f"planned_total_sequences: {target_count}\n"
            f"valid_act_numbers_from_arc: [{acts_hint}]\n"
            f"closing_hint: {last_hint}\n\n"
            "PREMISE:\n"
            f"{premise_json}\n\n"
            "ARC:\n"
            f"{arc_json}\n\n"
            "PRIOR_SEQUENCES (do not duplicate; continue after these):\n"
            f"{prior_json}\n\n"
            f"Produce the single StorySequence for {next_id}."
        )

        result = agent.run_sync(blob)
        row = result.output
        if row.id != next_id:
            row = row.model_copy(update={"id": next_id})

        doc.sequences.append(row)
        doc = doc.model_copy(update={"sequence_count": len(doc.sequences)})
        _save_sequence(project_dir, doc)

    print(f"Wrote {need} row(s) → {project_dir / 'sequence.json'}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate StorySequence rows into sequence.json (premise + arc required).",
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=DEFAULT_TEST_PROJECT,
        help=f"Project folder (default: {DEFAULT_TEST_PROJECT})",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=DEFAULT_SEQUENCE_COUNT,
        help=(
            f"Target number of sequences total (default: {DEFAULT_SEQUENCE_COUNT}). "
            "Resumes if sequence.json already exists."
        ),
    )
    parser.add_argument(
        "--note",
        type=str,
        default=None,
        help="Optional note field stored on sequence.json.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm bar (e.g. for logs).",
    )
    args = parser.parse_args(argv)

    if args.count < 1:
        parser.error("--count must be at least 1")

    try:
        run_sequence_batch(
            args.project,
            args.count,
            note=args.note,
            no_progress=args.no_progress,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"Error: {e}")
        raise SystemExit(1) from e
