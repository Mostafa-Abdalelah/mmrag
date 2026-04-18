import pytest
from pydantic import ValidationError

from mmrag.generation.models import Answer, Citation


def test_answer_is_frozen() -> None:
    a = Answer(text="because of X", citations=[
        Citation(doc_id="d", page=1),
    ])
    with pytest.raises(ValidationError):
        a.text = "changed"  # type: ignore[misc]


def test_answer_refusal_has_empty_citations() -> None:
    a = Answer(text="I couldn't find evidence in the corpus.", citations=[])
    assert a.citations == []


def test_citation_accepts_optional_quote() -> None:
    c = Citation(doc_id="d", page=2, quote="relevant sentence")
    assert c.quote == "relevant sentence"
