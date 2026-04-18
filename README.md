# mmrag — Multi-Modal Document Intelligence

ColPali-based RAG for visually rich PDFs. See [CLAUDE.md](./CLAUDE.md) for scope, [design spec](./docs/superpowers/specs/2026-04-18-multimodal-rag-design.md) for architecture.

## Quickstart

```bash
uv sync
uv run rag-cli ingest data/pdfs/
```

## Tests

```bash
uv run pytest                  # fast tests only
uv run pytest -m slow          # include real ColPali load
```
