"""Interactive premise agent: chat until /generate writes premise.json into a new project folder."""

from __future__ import annotations

import argparse
import json
import random
import re
import unicodedata
from datetime import UTC, datetime
from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage

from awesome_o.model_settings import resolve_default_model
from awesome_o.models.premise import PremiseDocument, ProjectFolderNaming
from awesome_o.persona import (
    CHAT_SYSTEM_PROMPT,
    PREMISE_STRUCT_SYSTEM_PROMPT,
    PROJECT_FOLDER_NAMING_PROMPT,
)

_COMMAND_RE = re.compile(r"^\s*/([a-zA-Z-]+)\s*$")

_POST_GENERATE_LINES = [
    "Sweet, premise.json is totally in {path}. Ok like you're welcome.",
    "Done. Wrote premise.json to {path} — that rules, honestly.",
    "Whatever, I crushed it. Check {path} for premise.json.",
    "Lame if you don't read it — premise.json dropped at {path}.",
]


def _utc_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%d_%H%M%S")


def _folder_slug_from_title(title: str, max_len: int = 48) -> str:
    """ASCII-ish slug safe for Windows/macOS/Linux folder names."""
    t = title.strip()
    if not t:
        return "untitled"
    nfkd = unicodedata.normalize("NFKD", t)
    ascii_only = nfkd.encode("ascii", "ignore").decode("ascii").lower()
    s = re.sub(r"[^a-z0-9]+", "_", ascii_only)
    s = re.sub(r"_+", "_", s).strip("_")
    if not s:
        s = "untitled"
    if len(s) > max_len:
        s = s[:max_len].rstrip("_") or "untitled"
    return s


def _projects_root() -> Path:
    return Path.cwd() / "projects"


def _allocate_unique_project_dir(base_project_id: str) -> tuple[Path, str]:
    """Create projects/<id>/; on collision append _2, _3, …"""
    root = _projects_root()
    root.mkdir(parents=True, exist_ok=True)
    for n in range(50):
        project_id = base_project_id if n == 0 else f"{base_project_id}_{n + 1}"
        candidate = root / project_id
        try:
            candidate.mkdir(parents=False)
            return candidate, project_id
        except FileExistsError:
            continue
    raise RuntimeError("Could not create a unique project folder (too many collisions).")


def _format_transcript(lines: list[tuple[str, str]]) -> str:
    parts: list[str] = []
    for role, text in lines:
        label = "User" if role == "user" else "Awesome-O"
        parts.append(f"{label}: {text}")
    return "\n\n".join(parts)


def _strip_command(line: str) -> str | None:
    m = _COMMAND_RE.match(line.strip())
    return m.group(1).lower() if m else None


def run_premise_chat() -> None:
    model = resolve_default_model()
    chat_agent = Agent(model, system_prompt=CHAT_SYSTEM_PROMPT)
    naming_agent = Agent(
        model,
        system_prompt=PROJECT_FOLDER_NAMING_PROMPT,
        output_type=ProjectFolderNaming,
    )
    premise_agent = Agent(
        model,
        system_prompt=PREMISE_STRUCT_SYSTEM_PROMPT,
        output_type=PremiseDocument,
    )

    message_history: list[ModelMessage] = []
    transcript: list[tuple[str, str]] = []

    print(
        "Awesome-O> Greetings. I'm Awesome-O.\n"
        "Awesome-O> Ok like, I'm here to help you write your movie premise.\n"
        "\n"
        "Chat until it feels right, then type /generate. I'll name the project like a movie "
        "title, then write premise.json under projects/<slug>_<datetime>/.\n"
        "Commands: /help  /generate  /quit\n"
    )

    while True:
        try:
            raw = input("you> ").rstrip("\n")
        except (EOFError, KeyboardInterrupt):
            print("\nAwesome-O> Ok like bye.")
            break

        cmd = _strip_command(raw)
        if cmd in ("quit", "exit", "q"):
            print("Awesome-O> Whatever, later.")
            break
        if cmd == "help":
            print(
                "Awesome-O> /generate = Gemini picks a movie-style title, new folder "
                "projects/<title-slug>_<datetime>/ + premise.json from this chat. "
                "/quit = exit. No other slash-commands yet, duh."
            )
            continue
        if cmd == "generate":
            if not any(role == "user" for role, _ in transcript):
                print(
                    "Awesome-O> Lame — you didn't actually pitch anything. "
                    "Tell me about your story first, then /generate."
                )
                continue

            naming_blob = (
                "Conversation transcript:\n"
                f"{_format_transcript(transcript)}\n\n"
                "Propose working_title for this story."
            )
            try:
                naming_result = naming_agent.run_sync(naming_blob)
                naming = naming_result.output
                working_title = naming.working_title.strip() or "Untitled"
            except Exception as e:  # noqa: BLE001 — CLI surfaces provider errors
                print(f"Awesome-O> Couldn't name the project (try again): {e}")
                continue

            stamp = _utc_stamp()
            slug = _folder_slug_from_title(working_title)
            base_id = f"{slug}_{stamp}"

            try:
                project_dir, project_id = _allocate_unique_project_dir(base_id)
            except OSError as e:
                print(f"Awesome-O> Couldn't create project folder: {e}")
                continue

            user_blob = (
                f"project_id (exact string for PremiseDocument.project_id): {project_id}\n"
                f"preferred_title (use as PremiseDocument.title): {working_title}\n\n"
                "Conversation transcript:\n"
                f"{_format_transcript(transcript)}\n\n"
                "Produce the PremiseDocument."
            )
            try:
                result = premise_agent.run_sync(user_blob)
                doc = result.output
                if doc.project_id != project_id:
                    doc = doc.model_copy(update={"project_id": project_id})
            except Exception as e:  # noqa: BLE001
                try:
                    project_dir.rmdir()
                except OSError:
                    pass
                print(f"Awesome-O> Ugh, generation failed (no files left behind): {e}")
                continue

            out_path = project_dir / "premise.json"
            out_path.write_text(
                json.dumps(doc.model_dump(), indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            print(
                f'Awesome-O> I\'m calling it "{working_title}" → folder `{project_id}`.\n'
                f"Awesome-O> {random.choice(_POST_GENERATE_LINES).format(path=out_path.resolve())}"
            )

            message_history.clear()
            transcript.clear()
            continue

        if raw.startswith("/"):
            print("Awesome-O> Unknown command. Try /help, genius.")
            continue

        if not raw.strip():
            continue

        try:
            cr = chat_agent.run_sync(raw.strip(), message_history=message_history)
        except Exception as e:  # noqa: BLE001
            print(f"Awesome-O> That request totally failed: {e}")
            continue

        message_history += cr.new_messages()
        reply = cr.output.strip()
        transcript.append(("user", raw.strip()))
        transcript.append(("assistant", reply))
        print(f"Awesome-O> {reply}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Awesome-O premise agent (interactive chat; /generate writes premise.json).",
    )
    parser.parse_args(argv)
    try:
        run_premise_chat()
    except RuntimeError as e:
        print(f"Awesome-O> {e}")
        raise SystemExit(1) from e
