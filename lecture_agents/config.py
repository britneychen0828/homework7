from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class PipelineConfig:
    repo_root: Path
    pdf_path: Path
    captions_path: Path
    style_path: Path
    projects_root: Path
    project_dir: Path
    slide_images_dir: Path
    audio_dir: Path
    temp_dir: Path
    slide_description_path: Path
    premise_path: Path
    arc_path: Path
    narration_path: Path
    final_video_path: Path
    image_dpi: int = 144
    image_format: str = "PNG"
    text_provider: str = field(default_factory=lambda: os.getenv("LECTURE_TEXT_PROVIDER", "").strip().lower())
    text_model: str = field(default_factory=lambda: os.getenv("LECTURE_TEXT_MODEL", "").strip())
    tts_provider: str = field(default_factory=lambda: os.getenv("LECTURE_TTS_PROVIDER", "").strip().lower())
    tts_model: str = field(default_factory=lambda: os.getenv("LECTURE_TTS_MODEL", "").strip())
    tts_voice: str = field(default_factory=lambda: os.getenv("LECTURE_TTS_VOICE", "").strip())
    openai_api_key: str | None = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    gemini_api_key: str | None = field(
        default_factory=lambda: os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    )

    @classmethod
    def create(
        cls,
        repo_root: Path,
        pdf_name: str = "Lecture_17_AI_screenplays.pdf",
        captions_name: str = "lecture_11_section_2_captions.txt",
        style_name: str = "style.json",
        project_name: str | None = None,
    ) -> "PipelineConfig":
        pdf_path = repo_root / pdf_name
        captions_path = repo_root / captions_name
        style_path = repo_root / style_name
        projects_root = repo_root / "projects"
        project_dir = projects_root / (project_name or timestamped_project_name())
        slide_images_dir = project_dir / "slide_images"
        audio_dir = project_dir / "audio"
        temp_dir = project_dir / "temp"
        pdf_stem = pdf_path.stem
        return cls(
            repo_root=repo_root,
            pdf_path=pdf_path,
            captions_path=captions_path,
            style_path=style_path,
            projects_root=projects_root,
            project_dir=project_dir,
            slide_images_dir=slide_images_dir,
            audio_dir=audio_dir,
            temp_dir=temp_dir,
            slide_description_path=project_dir / "slide_description.json",
            premise_path=project_dir / "premise.json",
            arc_path=project_dir / "arc.json",
            narration_path=project_dir / "slide_description_narration.json",
            final_video_path=project_dir / f"{pdf_stem}.mp4",
        )

    def ensure_directories(self) -> None:
        self.projects_root.mkdir(parents=True, exist_ok=True)
        self.project_dir.mkdir(parents=True, exist_ok=True)
        self.slide_images_dir.mkdir(parents=True, exist_ok=True)
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def resolved_text_provider(self) -> str:
        if self.text_provider:
            return self.text_provider
        if self.openai_api_key:
            return "openai"
        if self.gemini_api_key:
            return "gemini"
        return "openai"

    def resolved_text_model(self) -> str:
        if self.text_model:
            return self.text_model
        if self.resolved_text_provider() == "gemini":
            return "gemini-2.5-flash"
        return "gpt-4.1-mini"

    def resolved_tts_provider(self) -> str:
        if self.tts_provider:
            return self.tts_provider
        if self.openai_api_key:
            return "openai"
        if self.gemini_api_key:
            return "gemini"
        return "say"

    def resolved_tts_model(self) -> str:
        if self.tts_model:
            return self.tts_model
        if self.resolved_tts_provider() == "openai":
            return "gpt-4o-mini-tts"
        if self.resolved_tts_provider() == "gemini":
            return "gemini-2.5-flash-preview-tts"
        return ""

    def resolved_tts_voice(self) -> str:
        if self.tts_voice:
            return self.tts_voice
        if self.resolved_tts_provider() == "openai":
            return "alloy"
        if self.resolved_tts_provider() == "gemini":
            return "kore"
        return ""


def timestamped_project_name() -> str:
    from datetime import datetime

    return f"project_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
