from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from mmrag.corpus.models import Document, DocumentManifest


def test_document_is_frozen() -> None:
    d = Document(
        doc_id="abc",
        source_path=Path("/tmp/a.pdf"),
        sha256="deadbeef",
        n_pages=3,
        ingested_at=datetime.now(timezone.utc),
    )
    with pytest.raises(ValidationError):
        d.n_pages = 99  # type: ignore[misc]


def test_manifest_round_trips_json(tmp_path: Path) -> None:
    d = Document(
        doc_id="abc",
        source_path=Path("/tmp/a.pdf"),
        sha256="deadbeef",
        n_pages=3,
        ingested_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    m = DocumentManifest(documents=[d])

    path = tmp_path / "manifest.json"
    m.save(path)
    loaded = DocumentManifest.load(path)

    assert loaded == m


def test_manifest_get_by_hash() -> None:
    d = Document(
        doc_id="abc",
        source_path=Path("/tmp/a.pdf"),
        sha256="deadbeef",
        n_pages=3,
        ingested_at=datetime.now(timezone.utc),
    )
    m = DocumentManifest(documents=[d])
    assert m.get_by_hash("deadbeef") is d
    assert m.get_by_hash("missing") is None
