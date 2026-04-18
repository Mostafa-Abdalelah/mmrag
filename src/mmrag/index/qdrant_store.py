from __future__ import annotations

from pathlib import Path
from uuid import uuid5, NAMESPACE_URL

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from mmrag.index.schema import Chunk

_COLL = "mmrag"


def _pid(chunk_id: str) -> str:
    return str(uuid5(NAMESPACE_URL, chunk_id))


class QdrantIndex:
    def __init__(self, path: Path, *, dense_dim: int) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self._client = QdrantClient(path=str(path))
        self._dim = dense_dim
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        existing = {c.name for c in self._client.get_collections().collections}
        if _COLL in existing:
            return
        self._client.create_collection(
            collection_name=_COLL,
            vectors_config={
                "dense": qm.VectorParams(size=self._dim, distance=qm.Distance.COSINE),
                "colpali": qm.VectorParams(
                    size=128,
                    distance=qm.Distance.COSINE,
                    multivector_config=qm.MultiVectorConfig(
                        comparator=qm.MultiVectorComparator.MAX_SIM,
                    ),
                ),
            },
        )

    def upsert_dense(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        _zero_mv = np.zeros((1, 128), dtype=np.float32).tolist()
        points = [
            qm.PointStruct(
                id=_pid(c.chunk_id),
                vector={"dense": v.tolist(), "colpali": _zero_mv},
                payload=c.model_dump(),
            )
            for c, v in zip(chunks, vectors, strict=True)
        ]
        self._client.upsert(collection_name=_COLL, points=points)

    def upsert_multivector(self, chunks: list[Chunk], vectors: list[np.ndarray]) -> None:
        points = [
            qm.PointStruct(
                id=_pid(c.chunk_id),
                vector={"colpali": v.astype(np.float32).tolist()},
                payload=c.model_dump(),
            )
            for c, v in zip(chunks, vectors, strict=True)
        ]
        self._client.upsert(collection_name=_COLL, points=points)

    def count(self) -> int:
        return self._client.count(collection_name=_COLL, exact=True).count

    def search_multivector(
        self,
        query_vectors: np.ndarray,
        *,
        k: int,
    ) -> list[tuple[dict, float]]:
        hits = self._client.query_points(
            collection_name=_COLL,
            query=query_vectors.astype(np.float32).tolist(),
            using="colpali",
            limit=k,
            with_payload=True,
            query_filter=qm.Filter(
                must=[qm.FieldCondition(
                    key="modality",
                    match=qm.MatchValue(value="page_image"),
                )]
            ),
        ).points
        return [(h.payload, float(h.score)) for h in hits]

    def close(self) -> None:
        self._client.close()
