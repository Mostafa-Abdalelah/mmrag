import numpy as np

from mmrag.ingestion.protocols import TextEmbedder


class FakeTextEmbedder:
    def embed_texts(self, texts):
        return np.stack([np.full(384, float(len(t)), dtype=np.float32) for t in texts])


def test_fake_satisfies_protocol() -> None:
    assert isinstance(FakeTextEmbedder(), TextEmbedder)


def test_fake_returns_expected_shape() -> None:
    vecs = FakeTextEmbedder().embed_texts(["a", "bb"])
    assert vecs.shape == (2, 384)
    assert vecs[0, 0] == 1.0
    assert vecs[1, 0] == 2.0
