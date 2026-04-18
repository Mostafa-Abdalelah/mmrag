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


class _NullParser:
    def parse(self, pdf_path):
        return []


class _NullTextEmbedder:
    def embed_texts(self, texts):
        return np.zeros((len(texts), 384), dtype=np.float32)


class _NullVectorIndex:
    def upsert_dense(self, chunks, vectors): pass
    def upsert_multivector(self, chunks, vectors): pass
    def count(self): return 0


class _NullBm25:
    def add(self, chunks): pass
    def search(self, query, *, k): return []
    def save(self): pass
    def load(self): pass


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
    pipeline = IngestionPipeline(
        embedder=embedder,
        parser=_NullParser(),
        text_embedder=_NullTextEmbedder(),
        vector_index=_NullVectorIndex(),
        bm25_index=_NullBm25(),
        embeddings_dir=tmp_path,
        render_dpi=72,
        chunk_max_chars=500,
    )

    pipeline.ingest_document(doc)

    assert embedder.calls == 3
    out = embeddings_path_for(tmp_path, doc_id=doc.doc_id, sha256=doc.sha256)
    assert out.exists()


def test_pipeline_is_idempotent(sample_pdfs: dict[str, Path], tmp_path: Path) -> None:
    doc = _doc_from_pdf(sample_pdfs["one"])
    embedder = StubEmbedder()
    pipeline = IngestionPipeline(
        embedder=embedder,
        parser=_NullParser(),
        text_embedder=_NullTextEmbedder(),
        vector_index=_NullVectorIndex(),
        bm25_index=_NullBm25(),
        embeddings_dir=tmp_path,
        render_dpi=72,
        chunk_max_chars=500,
    )

    pipeline.ingest_document(doc)
    first_calls = embedder.calls
    out = embeddings_path_for(tmp_path, doc_id=doc.doc_id, sha256=doc.sha256)
    mtime1 = out.stat().st_mtime_ns

    pipeline.ingest_document(doc)
    assert embedder.calls == first_calls
    assert out.stat().st_mtime_ns == mtime1
