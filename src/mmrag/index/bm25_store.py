from __future__ import annotations

import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from mmrag.index.schema import Chunk

_TOKEN = re.compile(r"[A-Za-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(text)]


class Bm25Index:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._chunks: list[Chunk] = []
        self._bm25: BM25Okapi | None = None

    def add(self, chunks: list[Chunk]) -> None:
        self._chunks = [*self._chunks, *chunks]
        corpus = [_tokenize(c.content) for c in self._chunks]
        self._bm25 = BM25Okapi(corpus)

    def search(self, query: str, *, k: int) -> list[Chunk]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        order = scores.argsort()[::-1][:k]
        return [self._chunks[i] for i in order]

    def save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("wb") as f:
            pickle.dump({"chunks": self._chunks}, f)

    def load(self) -> None:
        with self._path.open("rb") as f:
            state = pickle.load(f)
        self._chunks = state["chunks"]
        corpus = [_tokenize(c.content) for c in self._chunks]
        self._bm25 = BM25Okapi(corpus) if corpus else None
