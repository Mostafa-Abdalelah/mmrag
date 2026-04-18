from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pymupdf

from mmrag.corpus.hashing import sha256_file
from mmrag.corpus.models import Document


def scan_pdf_directory(pdf_dir: Path) -> list[Document]:
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    return [_build_document(p) for p in pdfs]


def _build_document(pdf_path: Path) -> Document:
    with pymupdf.open(pdf_path) as doc:
        n_pages = doc.page_count
    return Document(
        doc_id=pdf_path.stem,
        source_path=pdf_path.resolve(),
        sha256=sha256_file(pdf_path),
        n_pages=n_pages,
        ingested_at=datetime.now(timezone.utc),
    )
