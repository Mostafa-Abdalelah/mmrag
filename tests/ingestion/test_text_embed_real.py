import pytest

from mmrag.ingestion.text_embed import BgeTextEmbedder


@pytest.mark.slow
def test_real_bge_embeds_batches() -> None:
    vecs = BgeTextEmbedder().embed_texts(["hello world", "another sentence"])
    assert vecs.shape == (2, 384)
    import numpy as np
    norms = np.linalg.norm(vecs, axis=1)
    assert (abs(norms - 1.0) < 1e-3).all()
