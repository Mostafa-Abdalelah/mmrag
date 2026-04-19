# mmrag — Multi-Modal Document Intelligence

ColPali-based Retrieval-Augmented Generation for visually rich PDFs. Preserves tables, charts, figures, and layout by indexing pages as images with a vision-language late-interaction retriever, and answers questions with a vision-capable LLM grounded in the retrieved pages.

## Highlights

- **Visual retrieval that actually sees the page.** ColPali (`vidore/colpali-v1.3`) embeds each page as a multi-vector and scores queries with MaxSim — no OCR, no lossy text extraction for layout.
- **Dual-path ingestion.** A structural path (Docling → section-aware chunks → bge-small dense + BM25 sparse) runs alongside the visual path, so both textual and visual signal are indexed per document.
- **Grounded answers with citations.** Gemini 2.5 Flash sees the actual retrieved page images and must cite `doc_id` + page in its JSON response; empty-evidence queries return an explicit refusal.
- **Local-first.** Qdrant runs in embedded file-backed mode, with a sidecar sparse index, and everything is wired to run on Apple Silicon (M1 / MPS) with no server processes.
- **Protocol-based layering.** Every boundary (`Embedder`, `Parser`, `VectorIndex`, `TextEmbedder`, `Retriever`, `Generator`) is a `Protocol` — swapping a component is a config change, not a code change.

## Architecture

```
Layer 0  Corpus Management      → DocumentRegistry, SHA-256 provenance, content-hash cache
Layer 1  Dual-Path Ingestion    → ColPali page-image embeddings  +  Docling structural chunks
Layer 2  Retrieval              → ColPali MaxSim search in Qdrant (text hybrid indexed, not yet fused)
Layer 3  Generation             → Gemini 2.5 Flash answer with inline citations
```

## Quickstart

Prerequisites: Python 3.11, [uv](https://docs.astral.sh/uv/), an Apple Silicon Mac (MPS) or NVIDIA GPU, and a `GEMINI_API_KEY` for the `ask` command.

```bash
uv sync
```

Drop your PDFs in `data/pdfs/` and ingest:

```bash
# full corpus
uv run rag-cli ingest data/pdfs

# demo mode: first N pages of each PDF (M1 friendly)
uv run rag-cli ingest data/pdfs --pages 100

# or cap the number of docs too
uv run rag-cli ingest data/pdfs --pages 100 --max-docs 50
```

Then retrieve and ask:

```bash
uv run rag-cli query "probable cause of the accident" --k 50
uv run rag-cli ask "What is the probable cause described on page 47 of AAR2101?" --k 50
```

Re-running `ingest` on an already-processed corpus is a no-op — every doc hash prints `[cache]`.

## Commands

| Command                     | What it does                                                                   |
|-----------------------------|--------------------------------------------------------------------------------|
| `rag-cli ingest <pdf_dir>`  | Dual-path ingest into Qdrant + sparse index under `$MMRAG_DATA_DIR/` (default `data/`) |
| `rag-cli query "<text>"`    | Top-K pages by ColPali MaxSim (`--k` default 50)                                |
| `rag-cli ask "<text>"`      | Retrieve top-K then answer via Gemini with inline citations                    |

## Configuration

All knobs are environment variables (canonical `MMRAG_*`, except `GEMINI_API_KEY` which is read directly):

| Variable                  | Default                        | Notes                                  |
|---------------------------|--------------------------------|----------------------------------------|
| `MMRAG_DATA_DIR`          | `data`                         | Root for manifest, embeddings, indexes |
| `MMRAG_COLPALI_MODEL`     | `vidore/colpali-v1.3`          | Visual multi-vector encoder            |
| `MMRAG_COLPALI_DEVICE`    | `mps`                          | `cpu` / `cuda` / `mps`                 |
| `MMRAG_BGE_MODEL`         | `BAAI/bge-small-en-v1.5`       | Dense text encoder (CPU)               |
| `MMRAG_PDF_RENDER_DPI`    | `150`                          | PDF → page image DPI for ColPali       |
| `MMRAG_ANSWER_RENDER_DPI` | `150`                          | PDF → page image DPI for Gemini        |
| `MMRAG_CHUNK_MAX_CHARS`   | `1200`                         | Cap for structural text chunks         |
| `MMRAG_GEMINI_MODEL`      | `gemini-2.5-flash`             | Vision-capable answerer                |
| `GEMINI_API_KEY`          | *(required for `ask`)*         | Unprefixed; also read from `.env`      |

## Tests

```bash
uv run pytest -m "not slow"   # fast unit + integration (no model weights loaded)
uv run pytest -m slow         # exercises real ColPali, bge-small, Docling, Qdrant, Gemini
```

50+ fast tests and 7+ slow tests — the slow suite drives real models on real NTSB PDFs end-to-end.

## Project Layout

```
src/mmrag/
  corpus/      # DocumentRegistry, content hashing, PDF loader
  ingestion/   # ColPali visual + Docling structural paths, pipeline, storage
  index/       # Chunk schema, Qdrant multivector+dense index, sparse sidecar
  retrieval/   # ColPaliRetriever, Hit model, Retriever Protocol
  generation/  # GeminiGenerator, Answer/Citation models, Generator Protocol
  cli.py       # rag-cli {ingest, query, ask}
  config.py    # pydantic-settings (env-driven)
tests/         # mirrors src/ layout; `@pytest.mark.slow` for model-backed paths
```


