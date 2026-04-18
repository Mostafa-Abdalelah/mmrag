from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from mmrag.index.schema import Chunk


@runtime_checkable
class VectorIndex(Protocol):
    def upsert_dense(self, chunks: list[Chunk], vectors: np.ndarray) -> None: ...

    def upsert_multivector(self, chunks: list[Chunk], vectors: list[np.ndarray]) -> None: ...

    def count(self) -> int: ...
