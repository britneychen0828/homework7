from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path


LOGGER = logging.getLogger(__name__)


def build_final_video(
    *,
    slide_paths: list[Path],
    audio_paths: list[Path],
    temp_dir: Path,
    output_path: Path,
) -> Path:
    segment_paths: list[Path] = []
    temp_dir.mkdir(parents=True, exist_ok=True)

    for index, (slide_path, audio_path) in enumerate(zip(slide_paths, audio_paths), start=1):
        duration = probe_duration_seconds(audio_path)
        segment_path = temp_dir / f"segment_{index:03d}.mp4"
        LOGGER.info("Building video segment %03d with %.2f seconds", index, duration)
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loop",
                "1",
                "-i",
                str(slide_path),
                "-i",
                str(audio_path),
                "-c:v",
                "libx264",
                "-t",
                f"{duration:.3f}",
                "-pix_fmt",
                "yuv420p",
                "-vf",
                "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-shortest",
                str(segment_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        segment_paths.append(segment_path)

    concat_manifest = temp_dir / "video_concat.txt"
    concat_manifest.write_text(
        "".join(f"file '{path.resolve()}'\n" for path in segment_paths),
        encoding="utf-8",
    )
    LOGGER.info("Concatenating %d segments into %s", len(segment_paths), output_path.name)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_manifest),
            "-c",
            "copy",
            str(output_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return output_path


def probe_duration_seconds(path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    return float(payload["format"]["duration"])
