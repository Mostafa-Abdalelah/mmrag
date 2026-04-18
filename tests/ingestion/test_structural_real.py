from pathlib import Path

import pytest

from mmrag.ingestion.structural import DoclingParser, ParsedBlock

NTSB = Path(__file__).resolve().parent.parent.parent / "data" / "pdfs"


def _smallest_pdf() -> Path:
    pdfs = sorted(NTSB.glob("*.pdf"), key=lambda p: p.stat().st_size)
    if not pdfs:
        pytest.skip("no NTSB corpus available")
    return pdfs[0]


@pytest.mark.slow
def test_docling_parses_real_ntsb_pdf(tmp_path: Path) -> None:
    import pymupdf
    src = _smallest_pdf()
    trimmed = tmp_path / "head.pdf"
    with pymupdf.open(src) as s:
        out = pymupdf.open()
        out.insert_pdf(s, from_page=0, to_page=min(1, s.page_count - 1))
        out.save(trimmed); out.close()

    blocks = DoclingParser().parse(trimmed)
    assert blocks, "Docling returned zero blocks"
    assert all(isinstance(b, ParsedBlock) for b in blocks)
    assert any(b.kind == "text" for b in blocks)
