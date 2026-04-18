import numpy as np
from PIL import Image

from mmrag.ingestion.protocols import Embedder


class FakeEmbedder:
    def embed_page(self, *, doc_id, page, image):
        from mmrag.ingestion.models import PageEmbedding
        arr = np.zeros((4, 128), dtype=np.float16)
        arr[:, 0] = page
        return PageEmbedding(
            doc_id=doc_id,
            page=page,
            patches=arr,
            pooled=np.full(128, page, dtype=np.float32),
        )

    def embed_query(self, text):
        return np.zeros((len(text), 128), dtype=np.float16)


def test_fake_embedder_satisfies_protocol() -> None:
    e = FakeEmbedder()
    assert isinstance(e, Embedder)


def test_fake_embedder_returns_expected_shape() -> None:
    e = FakeEmbedder()
    img = Image.new("RGB", (128, 128), color="white")
    emb = e.embed_page(doc_id="d", page=7, image=img)

    assert emb.patches.shape == (4, 128)
    assert emb.patches[0, 0] == 7
    assert emb.pooled[0] == 7
