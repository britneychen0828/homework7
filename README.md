# Homework 7: Agentic Video Lecture Pipeline

This repository implements a multi-stage Python pipeline that turns `Lecture_17_AI_screenplays.pdf` into a single narrated `.mp4` lecture video when run locally.

The main entrypoint is `run_lecture_pipeline.py`, and the reusable agent code lives under `lecture_agents/`.

## Deliverables in this repo

- `style.json` in the repo root, generated from `lecture_11_section_2_captions.txt` by a style-analysis agent
- `Lecture_17_AI_screenplays.pdf` in the repo root
- `run_lecture_pipeline.py` as the end-to-end entrypoint
- `lecture_agents/` with the stage implementations
- `projects/` with a sample `project_YYYYMMDD_HHMMSS/` JSON bundle

The repository does not commit generated slide images, audio, temp files, or final videos. Those are created locally when the pipeline runs and are ignored by Git.

## Pipeline stages

1. `style_agent.py`
Reads the instructor transcript, calls the text model, and writes `style.json` at the repo root.

2. `description_agent.py`
Creates a new `projects/project_YYYYMMDD_HHMMSS/` folder, rasterizes the PDF into `slide_images/slide_001.png`, etc., and generates `slide_description.json`.

3. `premise_agent.py`
Reads the full `slide_description.json` and writes `premise.json`.

4. `arc_agent.py`
Reads `premise.json` plus `slide_description.json` and writes `arc.json`.

5. `narration_agent.py`
Uses the current slide image, `style.json`, `premise.json`, `arc.json`, `slide_description.json`, and all prior narrations to write `slide_description_narration.json`.

6. `tts_agent.py`
Synthesizes one MP3 per slide into `audio/slide_001.mp3`, etc. Long narration can be chunked and merged back into a single MP3 per slide.

7. `video_agent.py`
Builds one video segment per slide and concatenates them into `Lecture_17_AI_screenplays.mp4` inside the project folder.

## Chaining required by the rubric

The implementation intentionally preserves the required agent context:

- Slide descriptions are generated one slide at a time with the current slide image plus all prior slide descriptions.
- Narrations are generated one slide at a time with the current slide image plus `style.json`, `premise.json`, `arc.json`, the full `slide_description.json`, and all prior slide narrations.
- The title slide narration is forced to introduce the speaker and preview the lecture topic.

## Setup

Requirements:

- Python 3.10+
- `ffmpeg` and `ffprobe` on your `PATH`
- One supported model provider configured in `.env` or your shell

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Create `.env` in the repo root if you want API-backed generation:

```bash
GEMINI_API_KEY=your_gemini_key
```

or:

```bash
OPENAI_API_KEY=your_openai_key
```

Optional overrides:

```bash
LECTURE_TEXT_PROVIDER=gemini
LECTURE_TEXT_MODEL=gemini-2.5-flash
LECTURE_TTS_PROVIDER=say
LECTURE_TTS_MODEL=gpt-4o-mini-tts
LECTURE_TTS_VOICE=alloy
```

Provider defaults:

- Text generation defaults to OpenAI if `OPENAI_API_KEY` exists, otherwise Gemini if `GEMINI_API_KEY` or `GOOGLE_API_KEY` exists.
- TTS defaults to OpenAI if `OPENAI_API_KEY` exists, otherwise Gemini if `GEMINI_API_KEY` or `GOOGLE_API_KEY` exists, otherwise macOS `say`.

## Run

Run the full pipeline from the repo root:

```bash
python3 run_lecture_pipeline.py
```

Verbose mode:

```bash
python3 run_lecture_pipeline.py --verbose
```

Resume an existing project from a later stage:

```bash
python3 run_lecture_pipeline.py --project-dir project_YYYYMMDD_HHMMSS --start-at audio --verbose
```

Reuse an existing root `style.json`:

```bash
python3 run_lecture_pipeline.py --skip-style
```

## Expected local outputs

Each new run creates:

```text
projects/project_YYYYMMDD_HHMMSS/
├── arc.json
├── audio/
│   └── slide_001.mp3 ...
├── Lecture_17_AI_screenplays.mp4
├── premise.json
├── slide_description.json
├── slide_description_narration.json
└── slide_images/
    └── slide_001.png ...
```

This repo includes a sample JSON-only project folder for grading convenience:

```text
projects/project_20260409_131907/
├── arc.json
├── audio/
├── premise.json
├── slide_description.json
├── slide_description_narration.json
└── slide_images/
```

## Repository structure

```text
.
├── Lecture_17_AI_screenplays.pdf
├── README.md
├── lecture_11_section_2_captions.txt
├── lecture_agents/
├── projects/
├── requirements.txt
├── run_lecture_pipeline.py
└── style.json
```
