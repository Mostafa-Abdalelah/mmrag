from __future__ import annotations

from typing import Protocol, runtime_checkable

from mmrag.generation.models import Answer
from mmrag.retrieval.models import Hit


@runtime_checkable
class Generator(Protocol):
    def answer(self, *, query: str, hits: list[Hit]) -> Answer: ...
