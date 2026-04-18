import os
from pathlib import Path

import pytest

NTSB = Path(__file__).resolve().parent.parent.parent / "data" / "pdfs"


@pytest.mark.slow
def test_real_gemini_answers_about_ntsb_page(tmp_path: Path, monkeypatch) -> None:
    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("MMRAG_GEMINI_API_KEY")):
        from mmrag.config import Settings
        if not Settings().gemini_api_key:
            pytest.skip("GEMINI_API_KEY not set")
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

    from mmrag.config import Settings
    from mmrag.corpus.registry import DocumentRegistry
    from mmrag.generation.gemini_generator import GeminiGenerator
    from mmrag.index.qdrant_store import QdrantIndex
    from mmrag.ingestion.visual import ColPaliEmbedder
    from mmrag.retrieval.colpali_retriever import ColPaliRetriever

    settings = Settings()
    index = QdrantIndex(data / "qdrant", dense_dim=384)
    retriever = ColPaliRetriever(
        embedder=ColPaliEmbedder(model_name="vidore/colpali-v1.3", device="mps"),
        index=index,
    )
    hits = retriever.search("what does this document describe?", k=2)
    assert hits

    gen = GeminiGenerator(
        model=settings.gemini_model,
        api_key=settings.gemini_api_key,
        registry=DocumentRegistry(settings.manifest_path),
        render_dpi=150,
    )
    ans = gen.answer(query="what does this document describe?", hits=hits)
    index.close()

    assert ans.text
    assert any(c.doc_id == trimmed.stem for c in ans.citations) or \
           "couldn't find evidence" in ans.text.lower()
