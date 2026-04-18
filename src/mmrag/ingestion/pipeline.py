from __future__ import annotations

from pathlib import Path

import pymupdf
from PIL import Image

from mmrag.corpus.models import Document
from mmrag.ingestion.protocols import Embedder
from mmrag.ingestion.storage import (
    embeddings_path_for,
    save_embeddings,
)


class IngestionPipeline:
    def __init__(self, *, embedder: Embedder, embeddings_dir: Path, render_dpi: int) -> None:
        self._embedder = embedder
        self._dir = embeddings_dir
        self._dpi = render_dpi

    def ingest_document(self, doc: Document) -> Path:
        target = embeddings_path_for(self._dir, doc_id=doc.doc_id, sha256=doc.sha256)
        if target.exists():
            return target

        pages = _render_pages(doc.source_path, self._dpi)
        embeddings = [
            self._embedder.embed_page(doc_id=doc.doc_id, page=i + 1, image=img)
            for i, img in enumerate(pages)
        ]
        return save_embeddings(
            self._dir,
            doc_id=doc.doc_id,
            sha256=doc.sha256,
            embeddings=embeddings,
        )


def _render_pages(pdf_path: Path, dpi: int) -> list[Image.Image]:
    zoom = dpi / 72.0
    matrix = pymupdf.Matrix(zoom, zoom)
    images: list[Image.Image] = []
    with pymupdf.open(pdf_path) as doc:
        for page in doc:
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            images.append(Image.frombytes("RGB", (pix.width, pix.height), pix.samples))
    return images
