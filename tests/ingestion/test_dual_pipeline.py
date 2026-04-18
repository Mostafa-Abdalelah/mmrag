from pathlib import Path

import numpy as np

from mmrag.corpus.models import Document
from mmrag.index.bm25_store import Bm25Index
from mmrag.index.qdrant_store import QdrantIndex
from mmrag.ingestion.models import PageEmbedding
from mmrag.ingestion.pipeline import IngestionPipeline
from mmrag.ingestion.structural import ParsedBlock


class StubEmbedder:
    def embed_page(self, *, doc_id, page, image):
        return PageEmbedding(
            doc_id=doc_id, page=page,
            patches=np.zeros((4, 128), dtype=np.float16),
            pooled=np.zeros(128, dtype=np.float32),
        )
    def embed_query(self, text):
        return np.zeros((len(text), 128), dtype=np.float16)


class StubParser:
    def parse(self, pdf_path):
        return [
            ParsedBlock(page=1, kind="text", text="body paragraph",
                        bbox=(0.0, 0.0, 10.0, 10.0)),
            ParsedBlock(page=1, kind="table", text="| a | b |\n|---|---|\n| 1 | 2 |",
                        bbox=(0.0, 20.0, 10.0, 40.0)),
        ]


class StubTextEmbedder:
    def embed_texts(self, texts):
        return np.stack([np.full(384, float(i + 1), dtype=np.float32)
                         for i in range(len(texts))])


def _doc_from_pdf(path: Path) -> Document:
    from datetime import datetime, timezone
    import pymupdf
    from mmrag.corpus.hashing import sha256_file
    with pymupdf.open(path) as d:
        n = d.page_count
    return Document(
        doc_id=path.stem, source_path=path.resolve(),
        sha256=sha256_file(path), n_pages=n,
        ingested_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


def test_dual_pipeline_fills_qdrant_and_bm25(
    sample_pdfs: dict[str, Path], tmp_path: Path
) -> None:
    doc = _doc_from_pdf(sample_pdfs["one"])
    qi = QdrantIndex(tmp_path / "qdrant", dense_dim=384)
    bi = Bm25Index(tmp_path / "bm25.pkl")

    pipeline = IngestionPipeline(
        embedder=StubEmbedder(),
        parser=StubParser(),
        text_embedder=StubTextEmbedder(),
        vector_index=qi,
        bm25_index=bi,
        embeddings_dir=tmp_path / "emb",
        render_dpi=72,
        chunk_max_chars=500,
    )
    pipeline.ingest_document(doc)
    bi.save()

    assert qi.count() >= 2
    hits = bi.search("body paragraph", k=1)
    assert hits[0].content.startswith("body paragraph")
