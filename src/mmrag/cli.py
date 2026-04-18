from __future__ import annotations

from pathlib import Path

import typer

from mmrag.config import Settings
from mmrag.corpus.loader import scan_pdf_directory
from mmrag.corpus.registry import DocumentRegistry
from mmrag.index.bm25_store import Bm25Index
from mmrag.index.qdrant_store import QdrantIndex
from mmrag.ingestion.pipeline import IngestionPipeline
from mmrag.ingestion.protocols import Embedder, TextEmbedder
from mmrag.ingestion.structural import DoclingParser, Parser
from mmrag.ingestion.text_embed import BgeTextEmbedder
from mmrag.ingestion.visual import ColPaliEmbedder
from mmrag.retrieval.colpali_retriever import ColPaliRetriever

app = typer.Typer(add_completion=False)


@app.callback()
def _main() -> None:
    pass


def _build_embedder(settings: Settings) -> Embedder:
    return ColPaliEmbedder(
        model_name=settings.colpali_model,
        device=settings.colpali_device,
    )


def _build_parser(settings: Settings) -> Parser:
    return DoclingParser()


def _build_text_embedder(settings: Settings) -> TextEmbedder:
    return BgeTextEmbedder(model_name=settings.bge_model)


@app.command()
def ingest(pdf_dir: Path) -> None:
    settings = Settings()
    registry = DocumentRegistry(settings.manifest_path)
    qi = QdrantIndex(settings.qdrant_path, dense_dim=settings.dense_dim)
    bi = Bm25Index(settings.bm25_path)
    if settings.bm25_path.exists():
        bi.load()

    pipeline = IngestionPipeline(
        embedder=_build_embedder(settings),
        parser=_build_parser(settings),
        text_embedder=_build_text_embedder(settings),
        vector_index=qi,
        bm25_index=bi,
        embeddings_dir=settings.embeddings_dir,
        render_dpi=settings.pdf_render_dpi,
        chunk_max_chars=settings.chunk_max_chars,
    )

    docs = scan_pdf_directory(pdf_dir)
    for doc in docs:
        if registry.has_hash(doc.sha256):
            typer.echo(f"[cache] {doc.doc_id}")
            continue
        pipeline.ingest_document(doc)
        registry.add(doc)
        typer.echo(f"[ok]    {doc.doc_id} ({doc.n_pages} pages)")
    bi.save()
    qi.close()


@app.command()
def query(text: str, k: int = 5) -> None:
    settings = Settings()
    index = QdrantIndex(settings.qdrant_path, dense_dim=settings.dense_dim)
    retriever = ColPaliRetriever(
        embedder=_build_embedder(settings),
        index=index,
    )
    hits = retriever.search(text, k=k)
    index.close()
    for h in hits:
        typer.echo(f"{h.score:.4f}  {h.doc_id}  p{h.page}  [{h.modality}]")


if __name__ == "__main__":
    app()
