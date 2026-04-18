from pathlib import Path

from mmrag.corpus.loader import scan_pdf_directory


def test_scan_returns_one_document_per_pdf(sample_pdfs: dict[str, Path]) -> None:
    dir_ = sample_pdfs["one"].parent
    docs = scan_pdf_directory(dir_)

    ids = sorted(d.doc_id for d in docs)
    assert ids == ["sample_1page", "sample_3page"]


def test_scan_populates_page_count(sample_pdfs: dict[str, Path]) -> None:
    dir_ = sample_pdfs["one"].parent
    docs = scan_pdf_directory(dir_)
    page_counts = {d.doc_id: d.n_pages for d in docs}

    assert page_counts == {"sample_1page": 1, "sample_3page": 3}


def test_scan_populates_sha256(sample_pdfs: dict[str, Path]) -> None:
    dir_ = sample_pdfs["one"].parent
    docs = scan_pdf_directory(dir_)
    for d in docs:
        assert len(d.sha256) == 64
