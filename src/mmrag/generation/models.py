from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class Citation(BaseModel):
    model_config = ConfigDict(frozen=True)

    doc_id: str
    page: int
    quote: str | None = None


class Answer(BaseModel):
    model_config = ConfigDict(frozen=True)

    text: str
    citations: list[Citation]
