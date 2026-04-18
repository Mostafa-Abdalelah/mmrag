from pathlib import Path

import numpy as np

from mmrag.corpus.models import Document
from mmrag.ingestion.models import PageEmbedding
from mmrag.ingestion.pipeline import IngestionPipeline
from mmrag.ingestion.storage import embeddings_path_for


class StubEmbedder:
    def __init__(self) -> None:
        self.calls = 0

    def embed_page(self, *, doc_id, page, image):
        self.calls += 1
        return PageEmbedding(
            doc_id=doc_id,
            page=page,
            patches=np.zeros((4, 128), dtype=np.float16),
            pooled=np.zeros(128, dtype=np.float32),
        )

    def embed_query(self, text):
        return np.zeros((len(text), 128), dtype=np.float16)


def _doc_from_pdf(path: Path) -> Document:
    from datetime import datetime, timezone

    import pymupdf

    from mmrag.corpus.hashing import sha256_file

    with pymupdf.open(path) as d:
        n = d.page_count
    return Document(
        doc_id=path.stem,
        source_path=path.resolve(),
        sha256=sha256_file(path),
        n_pages=n,
        ingested_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


def test_pipeline_embeds_every_page(sample_pdfs: dict[str, Path], tmp_path: Path) -> None:
    doc = _doc_from_pdf(sample_pdfs["three"])
    embedder = StubEmbedder()
    pipeline = IngestionPipeline(embedder=embedder, embeddings_dir=tmp_path, render_dpi=72)

    pipeline.ingest_document(doc)

    assert embedder.calls == 3
    out = embeddings_path_for(tmp_path, doc_id=doc.doc_id, sha256=doc.sha256)
    assert out.exists()


def test_pipeline_is_idempotent(sample_pdfs: dict[str, Path], tmp_path: Path) -> None:
    doc = _doc_from_pdf(sample_pdfs["one"])
    embedder = StubEmbedder()
    pipeline = IngestionPipeline(embedder=embedder, embeddings_dir=tmp_path, render_dpi=72)

    pipeline.ingest_document(doc)
    first_calls = embedder.calls

    pipeline.ingest_document(doc)
    assert embedder.calls == first_calls
