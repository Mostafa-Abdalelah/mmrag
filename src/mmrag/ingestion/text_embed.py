from __future__ import annotations

from functools import cached_property

import numpy as np


class BgeTextEmbedder:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        self._model_name = model_name

    @cached_property
    def _model(self):
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(self._model_name, device="cpu")

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        arr = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return arr.astype(np.float32)
