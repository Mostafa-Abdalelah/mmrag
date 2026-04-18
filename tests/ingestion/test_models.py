import numpy as np
import pytest
from pydantic import ValidationError

from mmrag.ingestion.models import PageEmbedding


def test_page_embedding_is_frozen() -> None:
    emb = PageEmbedding(
        doc_id="a",
        page=1,
        patches=np.zeros((10, 128), dtype=np.float16),
        pooled=np.zeros(128, dtype=np.float32),
    )
    with pytest.raises(ValidationError):
        emb.page = 99  # type: ignore[misc]


def test_page_embedding_validates_shapes() -> None:
    with pytest.raises(ValidationError):
        PageEmbedding(
            doc_id="a",
            page=1,
            patches=np.zeros((10, 64), dtype=np.float16),
            pooled=np.zeros(128, dtype=np.float32),
        )


def test_page_embedding_accepts_correct_shapes() -> None:
    emb = PageEmbedding(
        doc_id="a",
        page=1,
        patches=np.zeros((1030, 128), dtype=np.float16),
        pooled=np.zeros(128, dtype=np.float32),
    )
    assert emb.patches.shape == (1030, 128)
    assert emb.pooled.shape == (128,)
