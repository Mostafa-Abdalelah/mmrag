from __future__ import annotations

from pathlib import Path

import pymupdf
from PIL import Image


def render_pages(pdf_path: Path, dpi: int) -> list[Image.Image]:
    zoom = dpi / 72.0
    matrix = pymupdf.Matrix(zoom, zoom)
    images: list[Image.Image] = []
    with pymupdf.open(pdf_path) as doc:
        for page in doc:
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            images.append(Image.frombytes("RGB", (pix.width, pix.height), pix.samples))
    return images


def render_page(pdf_path: Path, *, page: int, dpi: int) -> Image.Image:
    zoom = dpi / 72.0
    matrix = pymupdf.Matrix(zoom, zoom)
    with pymupdf.open(pdf_path) as doc:
        pix = doc[page - 1].get_pixmap(matrix=matrix, alpha=False)
        return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
