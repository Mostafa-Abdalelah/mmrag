from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

Modality = Literal["page_image", "text", "table", "figure_caption"]


class BBox(BaseModel):
    model_config = ConfigDict(frozen=True)

    x0: float
    y0: float
    x1: float
    y1: float

    @model_validator(mode="after")
    def _check_order(self) -> "BBox":
        if self.x1 <= self.x0 or self.y1 <= self.y0:
            raise ValueError("BBox requires x1>x0 and y1>y0")
        return self

    def width(self) -> float:
        return self.x1 - self.x0

    def height(self) -> float:
        return self.y1 - self.y0


class Chunk(BaseModel):
    model_config = ConfigDict(frozen=True)

    chunk_id: str
    doc_id: str
    page: int
    bbox: BBox | None
    modality: Modality
    content: str
    source_hash: str
