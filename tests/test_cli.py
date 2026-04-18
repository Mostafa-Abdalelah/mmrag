from pathlib import Path

import numpy as np
from typer.testing import CliRunner

from mmrag.cli import app
from mmrag.ingestion.models import PageEmbedding


class StubEmbedder:
    def embed_page(self, *, doc_id, page, image):
        return PageEmbedding(
            doc_id=doc_id,
            page=page,
            patches=np.zeros((4, 128), dtype=np.float16),
            pooled=np.zeros(128, dtype=np.float32),
        )

    def embed_query(self, text):
        return np.zeros((len(text), 128), dtype=np.float16)


def test_ingest_command_registers_documents(
    sample_pdfs: dict[str, Path], tmp_path: Path, monkeypatch
) -> None:
    data_dir = tmp_path / "data"
    monkeypatch.setenv("MMRAG_DATA_DIR", str(data_dir))

    from mmrag import cli
    monkeypatch.setattr(cli, "_build_embedder", lambda settings: StubEmbedder())

    runner = CliRunner()
    result = runner.invoke(app, ["ingest", str(sample_pdfs["one"].parent)])

    assert result.exit_code == 0, result.stdout
    assert (data_dir / "manifest.json").exists()
    assert any((data_dir / "embeddings").glob("*.npz"))
