"""Arc agent CLI: chat with Awesome-O, /draft acts into arc.json, /edit to revise an act."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage

from awesome_o.model_settings import resolve_default_model
from awesome_o.models.arc import Act, ArcDocument
from awesome_o.models.premise import PremiseDocument
from awesome_o.persona import (
    ARC_ACT_STRUCT_SYSTEM_PROMPT,
    ARC_CHAT_SYSTEM_PROMPT,
    ARC_EDIT_ACT_SYSTEM_PROMPT,
)

DEFAULT_TEST_PROJECT = Path("projects/premise_20260401_221032")

_POST_DRAFT_LINES = [
    "I shoved that into arc.json. If it's lame, tell me what's wrong, then type /edit.",
    "Sweet, that's Act {n}. Not feeling it? Complain, then /edit. Next: /draft when ready.",
    "Done—act {n} is on disk. Whatever, read it. Fixes = chat + /edit.",
]


def _parse_command(line: str) -> tuple[str | None, str]:
    s = line.strip()
    if not s.startswith("/"):
        return None, s
    rest = s[1:].strip()
    if not rest:
        return "", ""
    parts = rest.split(maxsplit=1)
    cmd = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""
    return cmd, arg


def _format_transcript(lines: list[tuple[str, str]]) -> str:
    parts: list[str] = []
    for role, text in lines:
        label = "User" if role == "user" else "Awesome-O"
        parts.append(f"{label}: {text}")
    return "\n\n".join(parts)


def _load_premise(project_dir: Path) -> PremiseDocument:
    path = project_dir / "premise.json"
    if not path.is_file():
        raise FileNotFoundError(f"No premise.json at {path}")
    return PremiseDocument.model_validate_json(path.read_text(encoding="utf-8"))


def _save_arc(project_dir: Path, arc: ArcDocument) -> None:
    path = project_dir / "arc.json"
    path.write_text(
        json.dumps(arc.model_dump(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _load_or_init_arc(project_dir: Path, premise: PremiseDocument) -> ArcDocument:
    path = project_dir / "arc.json"
    if path.is_file():
        doc = ArcDocument.model_validate_json(path.read_text(encoding="utf-8"))
        if doc.project_id != premise.project_id:
            doc = doc.model_copy(update={"project_id": premise.project_id})
            _save_arc(project_dir, doc)
        return doc
    return ArcDocument(project_id=premise.project_id, acts=[])


def _chat_instructions(
    arc: ArcDocument,
    runtime_minutes: int | None,
    target_acts: int | None,
) -> str:
    bits = [
        "Current arc.json state:",
        json.dumps(arc.model_dump(), indent=2, ensure_ascii=False),
    ]
    if runtime_minutes is not None:
        bits.append(f"User set /runtime {runtime_minutes} (minutes).")
    if target_acts is not None:
        bits.append(f"User set /target {target_acts} (total acts to plan toward).")
    return "\n".join(bits)


def run_arc_chat(project_dir: Path) -> None:
    project_dir = project_dir.resolve()
    premise = _load_premise(project_dir)
    arc = _load_or_init_arc(project_dir, premise)

    model = resolve_default_model()
    premise_block = json.dumps(premise.model_dump(), indent=2, ensure_ascii=False)
    chat_agent = Agent(
        model,
        system_prompt=ARC_CHAT_SYSTEM_PROMPT + "\n\nLOCKED PREMISE:\n" + premise_block,
    )
    act_agent = Agent(
        model,
        system_prompt=ARC_ACT_STRUCT_SYSTEM_PROMPT,
        output_type=Act,
    )
    edit_agent = Agent(
        model,
        system_prompt=ARC_EDIT_ACT_SYSTEM_PROMPT,
        output_type=Act,
    )

    message_history: list[ModelMessage] = []
    transcript: list[tuple[str, str]] = []
    runtime_minutes: int | None = None
    target_acts: int | None = None
    last_drafted_act: int | None = arc.acts[-1].act if arc.acts else None

    print(
        "Awesome-O> Greetings. I'm Awesome-O.\n"
        "Awesome-O> Ok like, we're building the **arc** from your premise—one act at a time.\n"
        f"\nProject: {project_dir}\n"
        f"Premise: «{premise.title}»\n"
        "\nChat me up: runtime, how many acts, beats. "
        "**/draft** writes the next act to arc.json. "
        "Hate an act? Complain, then **/edit** (optional act number). "
        "**/show** [n] **/target N** **/runtime MIN** **/help** **/quit**\n"
    )

    while True:
        try:
            raw = input("you> ").rstrip("\n")
        except (EOFError, KeyboardInterrupt):
            print("\nAwesome-O> Ok like bye.")
            break

        cmd, arg = _parse_command(raw)
        if cmd in ("quit", "exit", "q"):
            print("Awesome-O> Whatever, later.")
            break

        if cmd == "help":
            print(
                "Awesome-O> /draft = LLM writes next act into arc.json and shows it. "
                "/edit [n] = revise act n (default: last drafted). "
                "/show [n] = print arc or one act. "
                "/target N = plan toward N acts. /runtime MIN = minutes for pacing notes. "
                "/quit = exit."
            )
            continue

        if cmd == "target":
            if not arg.isdigit() or int(arg) < 1:
                print("Awesome-O> Lame—use /target 3 or whatever, a positive integer.")
                continue
            target_acts = int(arg)
            print(f"Awesome-O> Ok like we're aiming for {target_acts} acts total. Sweet.")
            continue

        if cmd == "runtime":
            if not arg.isdigit() or int(arg) < 1:
                print("Awesome-O> /runtime needs minutes, like /runtime 118. Duh.")
                continue
            runtime_minutes = int(arg)
            print(f"Awesome-O> Noted ~{runtime_minutes} minutes. Default feature is like 110–120.")
            continue

        if cmd == "show":
            if not arc.acts:
                print("Awesome-O> No acts yet—chat, then /draft.")
                continue
            if not arg:
                print(json.dumps(arc.model_dump(), indent=2, ensure_ascii=False))
                continue
            if not arg.isdigit():
                print("Awesome-O> /show or /show 2—need a number for one act.")
                continue
            n = int(arg)
            found = next((a for a in arc.acts if a.act == n), None)
            if found is None:
                print(f"Awesome-O> No act {n} in arc.json yet, genius.")
                continue
            print(json.dumps(found.model_dump(), indent=2, ensure_ascii=False))
            continue

        if cmd == "draft":
            next_n = len(arc.acts) + 1
            if target_acts is not None and next_n > target_acts:
                print(
                    f"Awesome-O> Heads up: you said /target {target_acts} but this would be "
                    f"act {next_n}. /draft anyway if you want more—I'm not your mom."
                )

            prior = json.dumps([a.model_dump() for a in arc.acts], indent=2, ensure_ascii=False)
            blob = (
                f"next_act_number: {next_n}\n"
                f"structure_model: {arc.structure_model}\n"
                f"runtime_minutes: {runtime_minutes}\n"
                f"target_total_acts: {target_acts}\n\n"
                "Premise JSON:\n"
                f"{json.dumps(premise.model_dump(), indent=2, ensure_ascii=False)}\n\n"
                "Acts already committed (do not duplicate; continue the spine):\n"
                f"{prior}\n\n"
                "Design conversation:\n"
                f"{_format_transcript(transcript)}\n\n"
                "Produce the single Act for next_act_number."
            )
            try:
                result = act_agent.run_sync(blob)
                new_act = result.output
            except Exception as e:  # noqa: BLE001
                print(f"Awesome-O> Draft failed: {e}")
                continue

            if new_act.act != next_n:
                new_act = new_act.model_copy(update={"act": next_n})

            arc.acts.append(new_act)
            arc.acts.sort(key=lambda a: a.act)
            _save_arc(project_dir, arc)
            last_drafted_act = new_act.act

            print(json.dumps(new_act.model_dump(), indent=2, ensure_ascii=False))
            print()
            print(
                "Awesome-O> "
                + random.choice(_POST_DRAFT_LINES).format(n=new_act.act)
            )
            continue

        if cmd == "edit":
            if not arc.acts:
                print("Awesome-O> Nothing to edit—/draft an act first.")
                continue

            if arg.isdigit():
                act_n = int(arg)
            elif arg == "":
                if last_drafted_act is None:
                    print("Awesome-O> Say which act: /edit 2. No last draft tracked.")
                    continue
                act_n = last_drafted_act
            else:
                print("Awesome-O> Use /edit or /edit 2—numbers only.")
                continue

            idx = next((i for i, a in enumerate(arc.acts) if a.act == act_n), None)
            if idx is None:
                print(f"Awesome-O> No act {act_n} in arc.json.")
                continue

            current = arc.acts[idx]
            blob = (
                f"act_number (must stay this): {act_n}\n\n"
                "Premise JSON:\n"
                f"{json.dumps(premise.model_dump(), indent=2, ensure_ascii=False)}\n\n"
                "Full arc (context):\n"
                f"{json.dumps(arc.model_dump(), indent=2, ensure_ascii=False)}\n\n"
                "Act to revise:\n"
                f"{json.dumps(current.model_dump(), indent=2, ensure_ascii=False)}\n\n"
                "Conversation (apply user feedback):\n"
                f"{_format_transcript(transcript)}\n\n"
                "Output the revised Act only."
            )
            try:
                er = edit_agent.run_sync(blob)
                revised = er.output
            except Exception as e:  # noqa: BLE001
                print(f"Awesome-O> Edit failed: {e}")
                continue

            if revised.act != act_n:
                revised = revised.model_copy(update={"act": act_n})

            arc.acts[idx] = revised
            _save_arc(project_dir, arc)
            last_drafted_act = act_n

            print(json.dumps(revised.model_dump(), indent=2, ensure_ascii=False))
            print(f"Awesome-O> There, I patched act {act_n} in arc.json. Sweet?")
            continue

        if cmd is not None:
            print("Awesome-O> Unknown command. /help for the list.")
            continue

        if not raw.strip():
            continue

        try:
            cr = chat_agent.run_sync(
                raw.strip(),
                message_history=message_history,
                instructions=_chat_instructions(arc, runtime_minutes, target_acts),
            )
        except Exception as e:  # noqa: BLE001
            print(f"Awesome-O> That totally failed: {e}")
            continue

        message_history += cr.new_messages()
        reply = cr.output.strip()
        transcript.append(("user", raw.strip()))
        transcript.append(("assistant", reply))
        print(f"Awesome-O> {reply}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Awesome-O arc agent (reads premise.json).")
    parser.add_argument(
        "--project",
        type=Path,
        default=DEFAULT_TEST_PROJECT,
        help=f"Project folder containing premise.json (default: {DEFAULT_TEST_PROJECT})",
    )
    args = parser.parse_args(argv)
    try:
        run_arc_chat(args.project)
    except FileNotFoundError as e:
        print(f"Awesome-O> {e}")
        raise SystemExit(1) from e
    except RuntimeError as e:
        print(f"Awesome-O> {e}")
        raise SystemExit(1) from e
