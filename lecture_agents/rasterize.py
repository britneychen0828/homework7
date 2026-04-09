from __future__ import annotations

import logging
from pathlib import Path

import pypdfium2 as pdfium

from .io_utils import zero_padded_name


LOGGER = logging.getLogger(__name__)


def rasterize_pdf_to_images(pdf_path: Path, output_dir: Path, dpi: int) -> list[Path]:
    LOGGER.info("Rasterizing %s into slide images", pdf_path.name)
    pdf = pdfium.PdfDocument(str(pdf_path))
    scale = dpi / 72
    slide_paths: list[Path] = []

    for index in range(len(pdf)):
        page = pdf[index]
        bitmap = page.render(scale=scale)
        image = bitmap.to_pil()
        path = output_dir / zero_padded_name("slide", index + 1, "png")
        image.save(path)
        slide_paths.append(path)
        LOGGER.info("Rendered slide %03d -> %s", index + 1, path.name)

    return slide_paths
