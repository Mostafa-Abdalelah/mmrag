from pathlib import Path

import numpy as np
from typer.testing import CliRunner

from mmrag.cli import app
from mmrag.ingestion.models import PageEmbedding
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
        return [ParsedBlock(page=1, kind="text", text="hello world",
                            bbox=(0.0, 0.0, 10.0, 10.0))]


class StubTextEmbedder:
    def embed_texts(self, texts):
        return np.zeros((len(texts), 384), dtype=np.float32)


def test_ingest_command_registers_documents(
    sample_pdfs: dict[str, Path], tmp_path: Path, monkeypatch
) -> None:
    data_dir = tmp_path / "data"
    monkeypatch.setenv("MMRAG_DATA_DIR", str(data_dir))

    from mmrag import cli
    monkeypatch.setattr(cli, "_build_embedder", lambda s: StubEmbedder())
    monkeypatch.setattr(cli, "_build_parser", lambda s: StubParser())
    monkeypatch.setattr(cli, "_build_text_embedder", lambda s: StubTextEmbedder())

    runner = CliRunner()
    result = runner.invoke(app, ["ingest", str(sample_pdfs["one"].parent)])

    assert result.exit_code == 0, result.stdout
    assert (data_dir / "manifest.json").exists()
    assert any((data_dir / "embeddings").glob("*.npz"))
    assert (data_dir / "qdrant").exists()
    assert (data_dir / "bm25.pkl").exists()


class _StubEmbedderForQuery:
    def embed_page(self, *, doc_id, page, image):
        raise NotImplementedError
    def embed_query(self, text):
        import numpy as np
        return np.random.rand(4, 128).astype(np.float32)


def test_query_command_prints_hits(
    sample_pdfs: dict[str, Path], tmp_path: Path, monkeypatch
) -> None:
    import numpy as np
    from mmrag.index.qdrant_store import QdrantIndex
    from mmrag.index.schema import Chunk

    data_dir = tmp_path / "data"
    monkeypatch.setenv("MMRAG_DATA_DIR", str(data_dir))

    (data_dir / "qdrant").mkdir(parents=True, exist_ok=True)
    idx = QdrantIndex(data_dir / "qdrant", dense_dim=384)
    chunk = Chunk(chunk_id="doc1-page-1", doc_id="doc1", page=1, bbox=None,
                  modality="page_image", content="page", source_hash="h")
    idx.upsert_multivector([chunk], [np.random.rand(8, 128).astype(np.float32)])
    idx.close()

    from mmrag import cli
    monkeypatch.setattr(cli, "_build_embedder",
                        lambda s: _StubEmbedderForQuery())

    runner = CliRunner()
    result = runner.invoke(app, ["query", "anything", "--k", "1"])
    assert result.exit_code == 0, result.stdout
    assert "doc1" in result.stdout
    assert "p1" in result.stdout
