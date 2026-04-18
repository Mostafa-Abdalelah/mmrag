from __future__ import annotations

from mmrag.index.qdrant_store import QdrantIndex
from mmrag.index.schema import BBox
from mmrag.ingestion.protocols import Embedder
from mmrag.retrieval.models import Hit


class ColPaliRetriever:
    def __init__(self, *, embedder: Embedder, index: QdrantIndex) -> None:
        self._embedder = embedder
        self._index = index

    def search(self, query: str, *, k: int) -> list[Hit]:
        qv = self._embedder.embed_query(query)
        raw = self._index.search_multivector(qv, k=k)
        return [_to_hit(payload, score) for payload, score in raw]


def _to_hit(payload: dict, score: float) -> Hit:
    bbox_raw = payload.get("bbox")
    bbox = BBox(**bbox_raw) if bbox_raw else None
    return Hit(
        chunk_id=payload["chunk_id"],
        doc_id=payload["doc_id"],
        page=payload["page"],
        modality=payload["modality"],
        score=score,
        bbox=bbox,
    )
