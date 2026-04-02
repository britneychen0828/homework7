# Awesome-O

Classroom-scale **screenplay pipeline**: chat agents and batch scripts build **premise → arc → sequences → scenes**, with optional **dialogue repair** and a **rewrite** pass. Outputs are JSON files under a project folder (e.g. `projects/midnight_run_20260401_221032/` — slug from a Gemini-chosen title plus UTC timestamp).

Architecture slides for lecture: **`awesome_o_architecture_slides.html`** (open in a browser).

---

## Requirements

- **Python 3.11+**
- **Google Gemini** API access (default model: `gemini-2.5-flash` via pydantic-ai)

---

## Environment (`.env`)

Put a **`.env`** file in the **repo root** (next to `requirements.txt`). It is loaded automatically (`python-dotenv`).

| Variable | Notes |
|----------|--------|
| **`GEMINI_API_KEY`** | Primary; use your key from Google AI Studio / Gemini API. |
| **`GOOGLE_API_KEY`** | Alternative name; either works with the Google provider. |
| **`AWESOME_O_MODEL`** | Optional override, e.g. `google-gla:gemini-2.5-flash`. |

Copy **`.env.example`** → `.env` and paste your key. Do not commit `.env`.

---

## Install (repo root)

Open a terminal in the **repo root** (the folder that contains `requirements.txt` and `awesome_o/`). Install dependencies:

```bash
pip install -r requirements.txt
```

Keep running the `run_*.py` scripts from that same root so `import awesome_o` resolves and paths like `projects\…` work.

---

## Agentic flow

```text
premise.json  →  arc.json  →  sequence.json  →  scenes.json
     ↑              ↑              ↑                 ↑
  chat CLI     chat CLI      batch (+opt.      batch (+opt.
  /generate    /draft …       adaptive plan)    scene_plan)
```

Optional afterward (same project folder):

1. **`run_fix_scene_dialogue.py`** — Merge mis-typed dialogue in `scenes.json` (deterministic; add `--llm` if needed).
2. **`run_scenes_rewrite.py`** — New file **`scenes_rewrite.json`**: polish using premise + arc + `sequence.json` + prior scenes in the same sequence (and a tail from the previous sequence). Original `scenes.json` is left unchanged.

---

## How to run (class)

From the **repo root**, run **`python run_….py …`**. Each runner forwards **`sys.argv[1:]`** into **`argparse`** inside `awesome_o.cli.*` (use **`python run_….py --help`** for flags).

**Windows:** If double-clicking `.py` opens an editor, run from a terminal: **`python run_….py`**.

| Step | Runner | What it does |
|------|--------|----------------|
| 1. Premise | `python run_premise_agent.py` | Chat; `/generate` asks Gemini for a movie-style title, then writes `projects/<title_slug>_<UTC_datetime>/premise.json`. |
| 2. Arc | `python run_arc_agent.py --project projects\<id>` | Chat; `/draft`, `/edit`, `/show`, `/target`, `/runtime` → `arc.json`. |
| 3. Sequences | `python run_sequence_agent.py --project projects\<id>` | Writes `sequence.json` (default 8 rows; resumes if the file exists). |
| 4. Scenes | `python run_scenes_agent.py --project projects\<id>` | Writes `scenes.json` (`--per-sequence` or adaptive plan). |
| 5. Fix dialogue | `python run_fix_scene_dialogue.py --project projects\<id>` | Fixes `scenes.json` in place; optional `--llm`. |
| 6. Rewrite | `python run_scenes_rewrite.py --project projects\<id>` | Writes **`scenes_rewrite.json`**. |

---

## Package layout (`awesome_o/`)

| Path | Role |
|------|------|
| `models/` | Pydantic types for `premise.json`, `arc.json`, `sequence.json`, `scenes.json`, `scene_plan.json`, etc. |
| `cli/` | Entry points: premise, arc, sequence, scenes, fix_scene_dialogue, scenes_rewrite |
| `persona.py` | System prompts (Awesome-O voice + structured agents) |
| `model_settings.py` | `.env` + default Gemini model id |
| `scene_dialogue_normalize.py` | Rule-based cue+line → `dialogue` merge |

---

## Sample data (optional)

- **`projects/terminator_2_20260401_151738/`** — Example JSON from *Terminator 2* (large `scenes.json`). Not produced by the Awesome-O agents; useful as a shape reference.

---

## License / course use

Built for MGT575 / class demos; adjust as needed for your syllabus.
