from pathlib import Path

import pytest

NTSB = Path(__file__).resolve().parent.parent.parent / "data" / "pdfs"


@pytest.mark.slow
def test_real_dual_pipeline_fills_qdrant_and_bm25(tmp_path: Path, monkeypatch) -> None:
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

    # CLI calls qi.close() at end of ingest — lock is released; safe to open new client
    from qdrant_client import QdrantClient
    client = QdrantClient(path=str(data / "qdrant"))
    collections = client.get_collections().collections
    total = sum(client.count(c.name).count for c in collections)
    assert total >= 2, f"Expected >=2 points in Qdrant, got {total}"
    client.close()

    from mmrag.index.bm25_store import Bm25Index
    bi = Bm25Index(data / "bm25.pkl"); bi.load()
    assert bi.search("accident", k=1), "BM25 empty after real ingest"
