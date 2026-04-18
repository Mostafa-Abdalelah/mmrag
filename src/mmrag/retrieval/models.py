from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from mmrag.index.schema import BBox, Modality


class Hit(BaseModel):
    model_config = ConfigDict(frozen=True)

    chunk_id: str
    doc_id: str
    page: int
    modality: Modality
    score: float
    bbox: BBox | None
