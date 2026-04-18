from pathlib import Path

import pytest

NTSB = Path(__file__).resolve().parent.parent.parent / "data" / "pdfs"


@pytest.mark.slow
def test_colpali_retriever_returns_top_hit(tmp_path: Path, monkeypatch) -> None:
    import pymupdf
    pdfs = sorted(NTSB.glob("*.pdf"), key=lambda p: p.stat().st_size)
    if not pdfs:
        pytest.skip("no NTSB corpus")
    src = pdfs[0]
    sample_dir = tmp_path / "pdfs"; sample_dir.mkdir()
    trimmed = sample_dir / f"{src.stem}_head.pdf"
    with pymupdf.open(src) as s:
        out = pymupdf.open()
        out.insert_pdf(s, from_page=0, to_page=1)
        out.save(trimmed); out.close()

    data = tmp_path / "data"
    monkeypatch.setenv("MMRAG_DATA_DIR", str(data))

    from typer.testing import CliRunner
    from mmrag.cli import app
    r = CliRunner().invoke(app, ["ingest", str(sample_dir)])
    assert r.exit_code == 0, r.stdout

    from mmrag.index.qdrant_store import QdrantIndex
    from mmrag.ingestion.visual import ColPaliEmbedder
    from mmrag.retrieval.colpali_retriever import ColPaliRetriever

    index = QdrantIndex(data / "qdrant", dense_dim=384)
    embedder = ColPaliEmbedder(model_name="vidore/colpali-v1.3", device="mps")
    retriever = ColPaliRetriever(embedder=embedder, index=index)

    hits = retriever.search("probable cause of the accident", k=3)
    index.close()

    assert hits, "no hits returned"
    assert all(h.doc_id == trimmed.stem for h in hits)
    assert all(h.modality == "page_image" for h in hits)
    assert hits[0].score > 0
    assert hits[0].score >= hits[-1].score
