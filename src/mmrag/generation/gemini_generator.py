from __future__ import annotations

import json
import os
from functools import cached_property
from pathlib import Path

from PIL import Image

from mmrag.corpus.registry import DocumentRegistry
from mmrag.generation.models import Answer, Citation
from mmrag.ingestion.rendering import render_page
from mmrag.retrieval.models import Hit

_PROMPT = """\
You are answering a question using only the provided PDF page images as evidence.

Question: {query}

Rules:
- Use only facts visible in the provided pages.
- If the pages do not support an answer, respond with text="I couldn't find evidence for this in the corpus." and citations=[].
- Return valid JSON matching this schema:
  {{"text": string, "citations": [{{"doc_id": string, "page": integer, "quote": string | null}}]}}
- Use the doc_id and page number labels shown under each image when citing.
"""


class GeminiGenerator:
    def __init__(
        self,
        *,
        model: str,
        api_key: str | None,
        registry: DocumentRegistry,
        render_dpi: int,
    ) -> None:
        self._model = model
        self._api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self._registry = registry
        self._dpi = render_dpi

    @cached_property
    def _client(self):
        from google import genai
        if not self._api_key:
            raise RuntimeError("GEMINI_API_KEY not set")
        return genai.Client(api_key=self._api_key)

    def answer(self, *, query: str, hits: list[Hit]) -> Answer:
        if not hits:
            return Answer(text="I couldn't find evidence for this in the corpus.", citations=[])
        parts = self._build_parts(query, hits)
        response = self._client.models.generate_content(
            model=self._model,
            contents=parts,
            config={"response_mime_type": "application/json"},
        )
        data = json.loads(response.text)
        return _parse_answer(data)

    def _build_parts(self, query: str, hits: list[Hit]) -> list:
        parts: list = [_PROMPT.format(query=query)]
        for h in hits:
            img = self._render_hit(h)
            parts.append(f"doc_id={h.doc_id} page={h.page}")
            parts.append(img)
        return parts

    def _render_hit(self, h: Hit) -> Image.Image:
        doc = self._registry.manifest().get_by_hash_or_id(h.doc_id)
        return render_page(doc.source_path, page=h.page, dpi=self._dpi)


def _parse_answer(data: dict) -> Answer:
    citations = [Citation(**c) for c in data.get("citations") or []]
    return Answer(text=data.get("text") or "", citations=citations)
