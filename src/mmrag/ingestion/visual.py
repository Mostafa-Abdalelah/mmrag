from __future__ import annotations

from functools import cached_property

import numpy as np
import torch
from PIL import Image

from mmrag.ingestion.models import PageEmbedding


class ColPaliEmbedder:
    def __init__(self, model_name: str, device: str) -> None:
        self._model_name = model_name
        self._device = device

    @cached_property
    def _model(self):
        from colpali_engine.models import ColPali
        return ColPali.from_pretrained(
            self._model_name,
            torch_dtype=torch.float16,
            device_map=self._device,
        ).eval()

    @cached_property
    def _processor(self):
        from colpali_engine.models import ColPaliProcessor
        return ColPaliProcessor.from_pretrained(self._model_name)

    def embed_page(self, *, doc_id: str, page: int, image: Image.Image) -> PageEmbedding:
        with torch.no_grad():
            batch = self._processor.process_images([image]).to(self._device)
            embeddings = self._model(**batch)
        patches = embeddings[0].to(torch.float16).cpu().numpy()
        pooled = patches.astype(np.float32).mean(axis=0)
        return PageEmbedding(doc_id=doc_id, page=page, patches=patches, pooled=pooled)

    def embed_query(self, text: str) -> np.ndarray:
        with torch.no_grad():
            batch = self._processor.process_queries([text]).to(self._device)
            embeddings = self._model(**batch)
        return embeddings[0].to(torch.float16).cpu().numpy()
