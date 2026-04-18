from mmrag.ingestion.chunker import chunk_blocks
from mmrag.ingestion.structural import ParsedBlock


def _blk(page: int, kind: str, text: str) -> ParsedBlock:
    return ParsedBlock(page=page, kind=kind, text=text, bbox=(0.0, 0.0, 10.0, 10.0))


def test_chunker_groups_text_up_to_max_chars() -> None:
    blocks = [_blk(1, "text", "word " * 50) for _ in range(10)]
    chunks = chunk_blocks(
        blocks, doc_id="d", source_hash="h",
        max_chars=1000,
    )
    assert all(len(c.content) <= 1000 for c in chunks)
    assert sum(c.content.count("word") for c in chunks) == 500


def test_chunker_keeps_tables_as_own_chunks() -> None:
    blocks = [
        _blk(1, "text", "intro para"),
        _blk(1, "table", "| a | b |\n|---|---|\n| 1 | 2 |"),
        _blk(1, "text", "outro para"),
    ]
    chunks = chunk_blocks(blocks, doc_id="d", source_hash="h", max_chars=500)
    table_chunks = [c for c in chunks if c.modality == "table"]
    assert len(table_chunks) == 1
    assert "| a | b |" in table_chunks[0].content


def test_chunker_never_crosses_pages() -> None:
    blocks = [_blk(1, "text", "A"), _blk(2, "text", "B")]
    chunks = chunk_blocks(blocks, doc_id="d", source_hash="h", max_chars=500)
    pages = {c.page for c in chunks}
    assert pages == {1, 2}


def test_chunker_produces_stable_chunk_ids() -> None:
    blocks = [_blk(1, "text", "same text")]
    a = chunk_blocks(blocks, doc_id="d", source_hash="h", max_chars=500)
    b = chunk_blocks(blocks, doc_id="d", source_hash="h", max_chars=500)
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]
