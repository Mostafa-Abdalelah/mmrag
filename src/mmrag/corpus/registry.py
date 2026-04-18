from __future__ import annotations

from pathlib import Path

from mmrag.corpus.models import Document, DocumentManifest


class DocumentRegistry:
    def __init__(self, manifest_path: Path) -> None:
        self._path = manifest_path
        self._manifest = self._load_or_empty()

    def _load_or_empty(self) -> DocumentManifest:
        if self._path.exists():
            return DocumentManifest.load(self._path)
        return DocumentManifest(documents=[])

    def manifest(self) -> DocumentManifest:
        return self._manifest

    def has_hash(self, sha256: str) -> bool:
        return self._manifest.get_by_hash(sha256) is not None

    def add(self, doc: Document) -> None:
        self._manifest = DocumentManifest(
            documents=[*self._manifest.documents, doc]
        )
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._manifest.save(self._path)
