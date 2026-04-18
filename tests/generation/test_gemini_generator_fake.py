from mmrag.generation.models import Answer, Citation
from mmrag.generation.protocols import Generator
from mmrag.retrieval.models import Hit


class FakeGenerator:
    def answer(self, *, query, hits):
        return Answer(
            text=f"pretend answer to {query!r}",
            citations=[Citation(doc_id=h.doc_id, page=h.page) for h in hits],
        )


def test_fake_satisfies_protocol() -> None:
    assert isinstance(FakeGenerator(), Generator)


def test_fake_passes_through_hits() -> None:
    h = Hit(chunk_id="c", doc_id="d", page=3,
            modality="page_image", score=0.9, bbox=None)
    a = FakeGenerator().answer(query="why?", hits=[h])
    assert a.citations[0].doc_id == "d"
    assert a.citations[0].page == 3
