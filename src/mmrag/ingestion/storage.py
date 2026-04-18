from __future__ import annotations

from pathlib import Path

import numpy as np

from mmrag.ingestion.models import PageEmbedding


def embeddings_path_for(root: Path, *, doc_id: str, sha256: str) -> Path:
    return root / f"{doc_id}__{sha256[:16]}.npz"


def save_embeddings(
    root: Path,
    *,
    doc_id: str,
    sha256: str,
    embeddings: list[PageEmbedding],
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = embeddings_path_for(root, doc_id=doc_id, sha256=sha256)

    pages = np.array([e.page for e in embeddings], dtype=np.int32)
    patches = np.stack([e.patches for e in embeddings])
    pooled = np.stack([e.pooled for e in embeddings])

    np.savez(path, pages=pages, patches=patches, pooled=pooled)
    return path


def load_embeddings(
    root: Path,
    *,
    doc_id: str,
    sha256: str,
) -> list[PageEmbedding]:
    path = embeddings_path_for(root, doc_id=doc_id, sha256=sha256)
    data = np.load(path)
    return [
        PageEmbedding(
            doc_id=doc_id,
            page=int(p),
            patches=data["patches"][i],
            pooled=data["pooled"][i],
        )
        for i, p in enumerate(data["pages"])
    ]
