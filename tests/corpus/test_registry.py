from datetime import datetime, timezone
from pathlib import Path

from mmrag.corpus.models import Document
from mmrag.corpus.registry import DocumentRegistry


def _make_doc(doc_id: str, sha: str) -> Document:
    return Document(
        doc_id=doc_id,
        source_path=Path(f"/tmp/{doc_id}.pdf"),
        sha256=sha,
        n_pages=1,
        ingested_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


def test_registry_starts_empty(tmp_path: Path) -> None:
    reg = DocumentRegistry(tmp_path / "manifest.json")
    assert reg.manifest().documents == []


def test_registry_adds_and_persists(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    reg = DocumentRegistry(path)
    reg.add(_make_doc("a", "h1"))
    reg.add(_make_doc("b", "h2"))

    reloaded = DocumentRegistry(path)
    assert [d.doc_id for d in reloaded.manifest().documents] == ["a", "b"]


def test_registry_has_hash(tmp_path: Path) -> None:
    reg = DocumentRegistry(tmp_path / "manifest.json")
    reg.add(_make_doc("a", "h1"))
    assert reg.has_hash("h1")
    assert not reg.has_hash("h2")
