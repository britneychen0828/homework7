from __future__ import annotations

import argparse
import logging
from pathlib import Path

from dotenv import load_dotenv

from lecture_agents.arc_agent import generate_arc
from lecture_agents.config import PipelineConfig
from lecture_agents.description_agent import generate_slide_descriptions
from lecture_agents.io_utils import read_json
from lecture_agents.llm import build_json_client
from lecture_agents.logging_utils import configure_logging
from lecture_agents.narration_agent import generate_slide_narrations
from lecture_agents.premise_agent import generate_premise
from lecture_agents.rasterize import rasterize_pdf_to_images
from lecture_agents.style_agent import build_style_profile
from lecture_agents.tts_agent import synthesize_all_slides
from lecture_agents.validators import (
    PipelineValidationError,
    require_binary,
    require_file,
    require_matching_counts,
    require_text_provider_credentials,
    require_tts_provider_support,
)
from lecture_agents.video_agent import build_final_video


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a lecture PDF into a narrated video.")
    parser.add_argument("--repo-root", default=".", help="Repository root containing the PDF and captions file.")
    parser.add_argument(
        "--project-dir",
        default=None,
        help="Optional existing or desired project folder name under projects/, for example project_20260407_221206.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    parser.add_argument(
        "--skip-style",
        action="store_true",
        help="Reuse existing style.json instead of regenerating it from captions.",
    )
    parser.add_argument(
        "--stop-after",
        choices=["descriptions", "premise", "arc", "narration", "audio"],
        default=None,
        help="Optional checkpoint to stop after a specific stage and keep generated artifacts.",
    )
    parser.add_argument(
        "--start-at",
        choices=["style", "descriptions", "premise", "arc", "narration", "audio", "video"],
        default="style",
        help="Resume from a later stage. For stages after style, use --project-dir to target an existing project.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    load_dotenv(repo_root / ".env")
    configure_logging(verbose=args.verbose)
    if args.start_at != "style" and not args.project_dir:
        raise SystemExit("--start-at after style requires --project-dir so the pipeline knows which existing project to reuse.")
    config = PipelineConfig.create(repo_root=repo_root, project_name=args.project_dir)

    try:
        validate_local_inputs(config)
        config.ensure_directories()
        LOGGER.info("Project directory: %s", config.project_dir)

        text_provider = config.resolved_text_provider()
        text_model = config.resolved_text_model()
        tts_provider = config.resolved_tts_provider()
        tts_model = config.resolved_tts_model()
        tts_voice = config.resolved_tts_voice()
        require_text_provider_credentials(
            text_provider,
            openai_api_key=config.openai_api_key,
            gemini_api_key=config.gemini_api_key,
        )
        require_tts_provider_support(tts_provider, openai_api_key=config.openai_api_key)
        LOGGER.info("Using text provider=%s model=%s", text_provider, text_model)
        LOGGER.info("Using tts provider=%s model=%s voice=%s", tts_provider, tts_model or "n/a", tts_voice or "n/a")
        llm_client = build_json_client(
            provider=text_provider,
            model=text_model,
            openai_api_key=config.openai_api_key,
            gemini_api_key=config.gemini_api_key,
        )

        if args.skip_style and config.style_path.exists():
            LOGGER.info("Reusing existing %s", config.style_path.name)
            style_profile = read_json(config.style_path)
        elif config.captions_path.exists():
            LOGGER.info("Building %s from %s with the text model", config.style_path.name, config.captions_path.name)
            style_profile = build_style_profile(config.captions_path, config.style_path, client=llm_client)
        elif config.style_path.exists():
            LOGGER.warning(
                "Missing %s, so the pipeline is reusing existing %s.",
                config.captions_path.name,
                config.style_path.name,
            )
            style_profile = read_json(config.style_path)
        else:
            raise PipelineValidationError(
                f"Missing both captions transcript file ({config.captions_path}) and reusable style profile ({config.style_path})."
            )

        slide_paths = load_or_build_slide_paths(config=config, start_at=args.start_at)
        slide_descriptions = load_or_build_slide_descriptions(
            config=config,
            start_at=args.start_at,
            slide_paths=slide_paths,
            client=llm_client,
        )
        if args.stop_after == "descriptions":
            LOGGER.info("Stopping after slide description stage by request.")
            return 0
        premise = load_or_build_premise(
            config=config,
            start_at=args.start_at,
            slide_descriptions=slide_descriptions,
            client=llm_client,
        )
        if args.stop_after == "premise":
            LOGGER.info("Stopping after premise stage by request.")
            return 0
        arc = load_or_build_arc(
            config=config,
            start_at=args.start_at,
            premise=premise,
            slide_descriptions=slide_descriptions,
            client=llm_client,
        )
        if args.stop_after == "arc":
            LOGGER.info("Stopping after arc stage by request.")
            return 0
        narration_document = load_or_build_narration(
            config=config,
            start_at=args.start_at,
            slide_paths=slide_paths,
            style_profile=style_profile,
            premise=premise,
            arc=arc,
            slide_descriptions=slide_descriptions,
            client=llm_client,
        )
        if args.stop_after == "narration":
            LOGGER.info("Stopping after narration stage by request.")
            return 0
        audio_paths = synthesize_all_slides(
            narration_document=narration_document,
            audio_dir=config.audio_dir,
            temp_dir=config.temp_dir / "tts_chunks",
            provider=tts_provider,
            api_key=config.openai_api_key,
            gemini_api_key=config.gemini_api_key,
            model=tts_model,
            voice=tts_voice,
        )
        LOGGER.info("Wrote slide audio files to %s", config.audio_dir)
        if args.stop_after == "audio":
            LOGGER.info("Stopping after audio stage by request.")
            return 0
        require_matching_counts(slide_paths, audio_paths)
        build_final_video(
            slide_paths=slide_paths,
            audio_paths=audio_paths,
            temp_dir=config.temp_dir / "video_segments",
            output_path=config.final_video_path,
        )
        LOGGER.info("Pipeline complete. Final video: %s", config.final_video_path)
        return 0
    except PipelineValidationError as exc:
        LOGGER.error("%s", exc)
        return 2
    except Exception:
        LOGGER.exception("Pipeline failed")
        return 1


def validate_local_inputs(config: PipelineConfig) -> None:
    require_file(config.pdf_path, "lecture PDF")
    if not config.captions_path.exists() and not config.style_path.exists():
        raise PipelineValidationError(
            f"Missing both captions transcript file ({config.captions_path}) and style.json ({config.style_path})."
        )
    require_binary("ffmpeg")
    require_binary("ffprobe")


def load_or_build_slide_paths(*, config: PipelineConfig, start_at: str) -> list[Path]:
    if start_at in {"style", "descriptions"}:
        return rasterize_pdf_to_images(
            config.pdf_path,
            config.slide_images_dir,
            dpi=config.image_dpi,
        )
    slide_paths = sorted(config.slide_images_dir.glob("slide_*.png"))
    if not slide_paths:
        raise PipelineValidationError(
            f"No slide images found in {config.slide_images_dir}. Rerun from descriptions or earlier."
        )
    LOGGER.info("Reusing %d slide images from %s", len(slide_paths), config.slide_images_dir)
    return slide_paths


def load_or_build_slide_descriptions(*, config: PipelineConfig, start_at: str, slide_paths: list[Path], client):
    if start_at in {"style", "descriptions"}:
        result = generate_slide_descriptions(
            slide_paths=slide_paths,
            client=client,
            output_path=config.slide_description_path,
        )
        LOGGER.info("Wrote slide descriptions to %s", config.slide_description_path)
        return result
    require_file(config.slide_description_path, "slide description JSON")
    LOGGER.info("Reusing %s", config.slide_description_path)
    return read_json(config.slide_description_path)


def load_or_build_premise(*, config: PipelineConfig, start_at: str, slide_descriptions: dict, client):
    if start_at in {"style", "descriptions", "premise"}:
        result = generate_premise(
            slide_descriptions=slide_descriptions,
            client=client,
            output_path=config.premise_path,
        )
        LOGGER.info("Wrote premise to %s", config.premise_path)
        return result
    require_file(config.premise_path, "premise JSON")
    LOGGER.info("Reusing %s", config.premise_path)
    return read_json(config.premise_path)


def load_or_build_arc(*, config: PipelineConfig, start_at: str, premise: dict, slide_descriptions: dict, client):
    if start_at in {"style", "descriptions", "premise", "arc"}:
        result = generate_arc(
            premise=premise,
            slide_descriptions=slide_descriptions,
            client=client,
            output_path=config.arc_path,
        )
        LOGGER.info("Wrote arc to %s", config.arc_path)
        return result
    require_file(config.arc_path, "arc JSON")
    LOGGER.info("Reusing %s", config.arc_path)
    return read_json(config.arc_path)


def load_or_build_narration(
    *,
    config: PipelineConfig,
    start_at: str,
    slide_paths: list[Path],
    style_profile: dict,
    premise: dict,
    arc: dict,
    slide_descriptions: dict,
    client,
):
    if start_at in {"style", "descriptions", "premise", "arc", "narration"}:
        result = generate_slide_narrations(
            slide_paths=slide_paths,
            style_profile=style_profile,
            premise=premise,
            arc=arc,
            slide_descriptions=slide_descriptions,
            client=client,
            output_path=config.narration_path,
        )
        LOGGER.info("Wrote slide narration bundle to %s", config.narration_path)
        return result
    require_file(config.narration_path, "slide narration JSON")
    LOGGER.info("Reusing %s", config.narration_path)
    return read_json(config.narration_path)


if __name__ == "__main__":
    raise SystemExit(main())
