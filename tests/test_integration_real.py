from pathlib import Path

import numpy as np
import pytest

from mmrag.corpus.loader import scan_pdf_directory
from mmrag.corpus.registry import DocumentRegistry
from mmrag.ingestion.pipeline import IngestionPipeline
from mmrag.ingestion.storage import embeddings_path_for, load_embeddings
from mmrag.ingestion.visual import ColPaliEmbedder


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

NTSB_CORPUS = Path(__file__).resolve().parent.parent / "data" / "pdfs"


@pytest.fixture(scope="session")
def real_embedder() -> ColPaliEmbedder:
    return ColPaliEmbedder(model_name="vidore/colpali-v1.3", device="mps")


@pytest.fixture(scope="session")
def smallest_ntsb_pdf() -> Path:
    pdfs = sorted(NTSB_CORPUS.glob("*.pdf"), key=lambda p: p.stat().st_size)
    if not pdfs:
        pytest.skip(f"no NTSB PDFs in {NTSB_CORPUS}")
    return pdfs[0]


@pytest.mark.slow
def test_real_pipeline_embeds_ntsb_first_three_pages(
    real_embedder: ColPaliEmbedder, smallest_ntsb_pdf: Path, tmp_path: Path
) -> None:
    import pymupdf
    trimmed = tmp_path / "trimmed.pdf"
    with pymupdf.open(smallest_ntsb_pdf) as src:
        out = pymupdf.open()
        out.insert_pdf(src, from_page=0, to_page=min(2, src.page_count - 1))
        out.save(trimmed); out.close()

    registry = DocumentRegistry(tmp_path / "manifest.json")
    pipeline = IngestionPipeline(
        embedder=real_embedder,
        parser=_NullParser(),
        text_embedder=_NullTextEmbedder(),
        vector_index=_NullVectorIndex(),
        bm25_index=_NullBm25(),
        embeddings_dir=tmp_path / "embeddings",
        render_dpi=100,
        chunk_max_chars=500,
    )
    docs = scan_pdf_directory(trimmed.parent)
    for d in docs:
        pipeline.ingest_document(d)
        registry.add(d)

    assert len(docs) == 1
    doc = docs[0]
    npz = embeddings_path_for(tmp_path / "embeddings", doc_id=doc.doc_id, sha256=doc.sha256)
    assert npz.exists()

    embs = load_embeddings(tmp_path / "embeddings", doc_id=doc.doc_id, sha256=doc.sha256)
    assert len(embs) == doc.n_pages
    for e in embs:
        assert e.patches.ndim == 2 and e.patches.shape[1] == 128
        assert e.pooled.shape == (128,)
        assert np.isfinite(e.patches).all() and np.isfinite(e.pooled).all()


@pytest.mark.slow
def test_real_pipeline_is_idempotent_on_ntsb(
    real_embedder: ColPaliEmbedder, smallest_ntsb_pdf: Path, tmp_path: Path
) -> None:
    import pymupdf
    trimmed = tmp_path / "trimmed.pdf"
    with pymupdf.open(smallest_ntsb_pdf) as src:
        out = pymupdf.open()
        out.insert_pdf(src, from_page=0, to_page=0)
        out.save(trimmed); out.close()

    pipeline = IngestionPipeline(
        embedder=real_embedder,
        parser=_NullParser(),
        text_embedder=_NullTextEmbedder(),
        vector_index=_NullVectorIndex(),
        bm25_index=_NullBm25(),
        embeddings_dir=tmp_path / "embeddings",
        render_dpi=100,
        chunk_max_chars=500,
    )
    [doc] = scan_pdf_directory(trimmed.parent)
    pipeline.ingest_document(doc)
    out = embeddings_path_for(tmp_path / "embeddings", doc_id=doc.doc_id, sha256=doc.sha256)
    mtime1 = out.stat().st_mtime_ns
    pipeline.ingest_document(doc)
    assert out.stat().st_mtime_ns == mtime1


@pytest.mark.slow
def test_real_embedder_produces_query_vectors(real_embedder: ColPaliEmbedder) -> None:
    q = real_embedder.embed_query("What caused the accident?")
    assert q.ndim == 2
    assert q.shape[1] == 128
    assert q.shape[0] > 0
