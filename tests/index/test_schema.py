import pytest
from pydantic import ValidationError

from mmrag.index.schema import BBox, Chunk


def test_bbox_is_frozen_and_ordered() -> None:
    b = BBox(x0=10.0, y0=20.0, x1=100.0, y1=200.0)
    assert b.width() == 90.0
    assert b.height() == 180.0
    with pytest.raises(ValidationError):
        b.x0 = 99.0  # type: ignore[misc]


def test_bbox_rejects_inverted_coords() -> None:
    with pytest.raises(ValidationError):
        BBox(x0=100.0, y0=20.0, x1=10.0, y1=200.0)


def test_chunk_requires_valid_modality() -> None:
    with pytest.raises(ValidationError):
        Chunk(
            chunk_id="c1",
            doc_id="d",
            page=1,
            bbox=None,
            modality="bogus",  # type: ignore[arg-type]
            content="hi",
            source_hash="h",
        )


def test_chunk_accepts_all_modalities() -> None:
    for m in ("page_image", "text", "table", "figure_caption"):
        c = Chunk(
            chunk_id=f"c-{m}", doc_id="d", page=1, bbox=None,
            modality=m, content="x", source_hash="h",
        )
        assert c.modality == m


def test_chunk_is_frozen() -> None:
    c = Chunk(chunk_id="c", doc_id="d", page=1, bbox=None,
              modality="text", content="x", source_hash="h")
    with pytest.raises(ValidationError):
        c.page = 99  # type: ignore[misc]
