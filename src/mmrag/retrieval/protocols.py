from __future__ import annotations

from typing import Protocol, runtime_checkable

from mmrag.retrieval.models import Hit


@runtime_checkable
class Retriever(Protocol):
    def search(self, query: str, *, k: int) -> list[Hit]: ...
