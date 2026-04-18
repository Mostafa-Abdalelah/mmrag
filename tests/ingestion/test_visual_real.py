import pytest
from PIL import Image

from mmrag.config import Settings
from mmrag.ingestion.visual import ColPaliEmbedder


@pytest.mark.slow
def test_real_colpali_embeds_single_page() -> None:
    settings = Settings()
    embedder = ColPaliEmbedder(
        model_name=settings.colpali_model,
        device=settings.colpali_device,
    )
    img = Image.new("RGB", (896, 896), color="white")
    emb = embedder.embed_page(doc_id="fake", page=1, image=img)

    assert emb.patches.ndim == 2
    assert emb.patches.shape[1] == 128
    assert emb.pooled.shape == (128,)
