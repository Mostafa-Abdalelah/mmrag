from pathlib import Path

from mmrag.index.bm25_store import Bm25Index
from mmrag.index.schema import Chunk


def _c(i: int, text: str) -> Chunk:
    return Chunk(
        chunk_id=f"c{i}", doc_id="d", page=1, bbox=None,
        modality="text", content=text, source_hash="h",
    )


def test_bm25_roundtrip(tmp_path: Path) -> None:
    idx = Bm25Index(tmp_path / "bm25.pkl")
    idx.add([_c(1, "stall speed warning horn"),
             _c(2, "engine fire suppression system")])
    idx.save()

    loaded = Bm25Index(tmp_path / "bm25.pkl")
    loaded.load()

    hits = loaded.search("fire suppression", k=2)
    assert hits[0].chunk_id == "c2"
    assert len(hits) == 2


def test_bm25_respects_k(tmp_path: Path) -> None:
    idx = Bm25Index(tmp_path / "bm25.pkl")
    idx.add([_c(i, f"token{i} word") for i in range(10)])
    hits = idx.search("token3", k=3)
    assert len(hits) == 3
