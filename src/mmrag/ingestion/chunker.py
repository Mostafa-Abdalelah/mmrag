from __future__ import annotations

import hashlib

from mmrag.index.schema import BBox, Chunk, Modality
from mmrag.ingestion.structural import ParsedBlock


def chunk_blocks(
    blocks: list[ParsedBlock],
    *,
    doc_id: str,
    source_hash: str,
    max_chars: int,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    buffer: list[ParsedBlock] = []
    for blk in blocks:
        if blk.kind in ("table", "figure_caption"):
            _flush_text(buffer, chunks, doc_id, source_hash)
            buffer = []
            chunks.append(_make_chunk(doc_id, source_hash, blk.page, blk.bbox,
                                      _modality_of(blk.kind), blk.text))
            continue
        if buffer and buffer[0].page != blk.page:
            _flush_text(buffer, chunks, doc_id, source_hash)
            buffer = []
        if buffer:
            projected = "\n\n".join(b.text for b in buffer) + "\n\n" + blk.text
            if len(projected) > max_chars:
                _flush_text(buffer, chunks, doc_id, source_hash)
                buffer = []
        buffer.append(blk)
    _flush_text(buffer, chunks, doc_id, source_hash)
    return chunks


def _flush_text(
    buffer: list[ParsedBlock],
    chunks: list[Chunk],
    doc_id: str,
    source_hash: str,
) -> None:
    if not buffer:
        return
    text = "\n\n".join(b.text for b in buffer)
    page = buffer[0].page
    bbox = _union_bbox([b.bbox for b in buffer])
    chunks.append(_make_chunk(doc_id, source_hash, page, bbox, "text", text))


def _make_chunk(
    doc_id: str,
    source_hash: str,
    page: int,
    bbox_tuple: tuple[float, float, float, float] | None,
    modality: Modality,
    text: str,
) -> Chunk:
    fingerprint = hashlib.sha1(f"{doc_id}|{page}|{modality}|{text}".encode()).hexdigest()[:16]
    bbox = BBox(x0=bbox_tuple[0], y0=bbox_tuple[1], x1=bbox_tuple[2], y1=bbox_tuple[3]) \
        if bbox_tuple else None
    return Chunk(
        chunk_id=f"{doc_id}-p{page}-{fingerprint}",
        doc_id=doc_id,
        page=page,
        bbox=bbox,
        modality=modality,
        content=text,
        source_hash=source_hash,
    )


def _modality_of(kind: str) -> Modality:
    if kind == "table":
        return "table"
    if kind == "figure_caption":
        return "figure_caption"
    return "text"


def _union_bbox(
    bboxes: list[tuple[float, float, float, float] | None],
) -> tuple[float, float, float, float] | None:
    valid = [b for b in bboxes if b is not None]
    if not valid:
        return None
    x0 = min(b[0] for b in valid)
    y0 = min(b[1] for b in valid)
    x1 = max(b[2] for b in valid)
    y1 = max(b[3] for b in valid)
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)
