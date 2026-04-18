import pytest
from pydantic import ValidationError

from mmrag.retrieval.models import Hit


def test_hit_is_frozen() -> None:
    h = Hit(chunk_id="c1", doc_id="d", page=3, modality="page_image",
            score=0.87, bbox=None)
    with pytest.raises(ValidationError):
        h.score = 0.0  # type: ignore[misc]


def test_hit_rejects_bad_modality() -> None:
    with pytest.raises(ValidationError):
        Hit(chunk_id="c1", doc_id="d", page=1, modality="bogus",  # type: ignore[arg-type]
            score=0.5, bbox=None)


def test_hit_bbox_round_trips_from_dict() -> None:
    h = Hit.model_validate({
        "chunk_id": "c1", "doc_id": "d", "page": 1,
        "modality": "page_image", "score": 0.9,
        "bbox": {"x0": 0.0, "y0": 0.0, "x1": 10.0, "y1": 10.0},
    })
    assert h.bbox is not None
    assert h.bbox.width() == 10.0
