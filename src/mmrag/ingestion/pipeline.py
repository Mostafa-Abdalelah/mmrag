from __future__ import annotations

from pathlib import Path

import pymupdf
from PIL import Image

from mmrag.corpus.models import Document
from mmrag.index.bm25_store import Bm25Index
from mmrag.index.qdrant_store import QdrantIndex
from mmrag.index.schema import Chunk
from mmrag.ingestion.chunker import chunk_blocks
from mmrag.ingestion.protocols import Embedder, TextEmbedder
from mmrag.ingestion.storage import embeddings_path_for, save_embeddings
from mmrag.ingestion.structural import Parser


class IngestionPipeline:
    def __init__(
        self,
        *,
        embedder: Embedder,
        parser: Parser,
        text_embedder: TextEmbedder,
        vector_index: QdrantIndex,
        bm25_index: Bm25Index,
        embeddings_dir: Path,
        render_dpi: int,
        chunk_max_chars: int,
    ) -> None:
        self._embedder = embedder
        self._parser = parser
        self._text_embedder = text_embedder
        self._qi = vector_index
        self._bi = bm25_index
        self._dir = embeddings_dir
        self._dpi = render_dpi
        self._max_chars = chunk_max_chars

    def ingest_document(self, doc: Document) -> None:
        self._ingest_visual(doc)
        self._ingest_structural(doc)

    def _ingest_visual(self, doc: Document) -> None:
        target = embeddings_path_for(self._dir, doc_id=doc.doc_id, sha256=doc.sha256)
        if target.exists():
            return
        pages = _render_pages(doc.source_path, self._dpi)
        page_embs = [
            self._embedder.embed_page(doc_id=doc.doc_id, page=i + 1, image=img)
            for i, img in enumerate(pages)
        ]
        save_embeddings(self._dir, doc_id=doc.doc_id, sha256=doc.sha256, embeddings=page_embs)
        page_chunks = [
            Chunk(
                chunk_id=f"{doc.doc_id}-page-{e.page}",
                doc_id=doc.doc_id, page=e.page, bbox=None,
                modality="page_image",
                content=f"page {e.page} of {doc.doc_id}",
                source_hash=doc.sha256,
            )
            for e in page_embs
        ]
        mv = [e.patches.astype("float32") for e in page_embs]
        self._qi.upsert_multivector(page_chunks, mv)

    def _ingest_structural(self, doc: Document) -> None:
        blocks = self._parser.parse(doc.source_path)
        chunks = chunk_blocks(
            blocks, doc_id=doc.doc_id, source_hash=doc.sha256,
            max_chars=self._max_chars,
        )
        if not chunks:
            return
        vectors = self._text_embedder.embed_texts([c.content for c in chunks])
        self._qi.upsert_dense(chunks, vectors)
        self._bi.add(chunks)


def _render_pages(pdf_path: Path, dpi: int) -> list[Image.Image]:
    zoom = dpi / 72.0
    matrix = pymupdf.Matrix(zoom, zoom)
    images: list[Image.Image] = []
    with pymupdf.open(pdf_path) as doc:
        for page in doc:
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            images.append(Image.frombytes("RGB", (pix.width, pix.height), pix.samples))
    return images
