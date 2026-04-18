from pathlib import Path

import numpy as np

from mmrag.ingestion.models import PageEmbedding
from mmrag.ingestion.storage import (
    embeddings_path_for,
    load_embeddings,
    save_embeddings,
)


def _emb(page: int) -> PageEmbedding:
    return PageEmbedding(
        doc_id="doc_a",
        page=page,
        patches=np.full((4, 128), float(page), dtype=np.float16),
        pooled=np.full(128, float(page), dtype=np.float32),
    )


def test_roundtrip_preserves_arrays(tmp_path: Path) -> None:
    embs = [_emb(1), _emb(2)]
    save_embeddings(tmp_path, doc_id="doc_a", sha256="abc", embeddings=embs)

    loaded = load_embeddings(tmp_path, doc_id="doc_a", sha256="abc")

    assert [e.page for e in loaded] == [1, 2]
    assert np.array_equal(loaded[0].patches, embs[0].patches)
    assert np.array_equal(loaded[0].pooled, embs[0].pooled)


def test_path_is_content_hash_keyed(tmp_path: Path) -> None:
    p1 = embeddings_path_for(tmp_path, doc_id="d", sha256="h1")
    p2 = embeddings_path_for(tmp_path, doc_id="d", sha256="h2")
    assert p1 != p2
    assert p1.suffix == ".npz"
