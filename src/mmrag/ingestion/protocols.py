from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from PIL import Image

from mmrag.ingestion.models import PageEmbedding


@runtime_checkable
class Embedder(Protocol):
    def embed_page(self, *, doc_id: str, page: int, image: Image.Image) -> PageEmbedding: ...

    def embed_query(self, text: str) -> np.ndarray: ...
