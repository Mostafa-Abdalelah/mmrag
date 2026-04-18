from __future__ import annotations

from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict


class Document(BaseModel):
    model_config = ConfigDict(frozen=True)

    doc_id: str
    source_path: Path
    sha256: str
    n_pages: int
    ingested_at: datetime


class DocumentManifest(BaseModel):
    model_config = ConfigDict(frozen=True)

    documents: list[Document]

    def get_by_hash(self, sha256: str) -> Document | None:
        for d in self.documents:
            if d.sha256 == sha256:
                return d
        return None

    def save(self, path: Path) -> None:
        path.write_text(self.model_dump_json(indent=2))

    @classmethod
    def load(cls, path: Path) -> DocumentManifest:
        return cls.model_validate_json(path.read_text())
