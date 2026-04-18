from pathlib import Path

import numpy as np

from mmrag.index.qdrant_store import QdrantIndex
from mmrag.index.schema import Chunk


def _c(i: int, modality: str = "text") -> Chunk:
    return Chunk(
        chunk_id=f"c{i}", doc_id="d", page=1, bbox=None,
        modality=modality, content=f"text {i}", source_hash="h",
    )


def test_upsert_dense_persists_count(tmp_path: Path) -> None:
    idx = QdrantIndex(tmp_path / "qdrant", dense_dim=384)
    chunks = [_c(i) for i in range(3)]
    vectors = np.random.rand(3, 384).astype(np.float32)
    idx.upsert_dense(chunks, vectors)
    assert idx.count() == 3


def test_upsert_multivector_keyed_by_chunk_id(tmp_path: Path) -> None:
    idx = QdrantIndex(tmp_path / "qdrant", dense_dim=384)
    chunks = [_c(i, "page_image") for i in range(2)]
    mv = [np.random.rand(1030, 128).astype(np.float32) for _ in range(2)]
    idx.upsert_multivector(chunks, mv)
    assert idx.count() == 2
