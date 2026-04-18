from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

Kind = Literal["text", "table", "figure_caption"]


@dataclass(frozen=True)
class ParsedBlock:
    page: int
    kind: Kind
    text: str
    bbox: tuple[float, float, float, float] | None


@runtime_checkable
class Parser(Protocol):
    def parse(self, pdf_path: Path) -> list[ParsedBlock]: ...


class DoclingParser:
    @cached_property
    def _converter(self):
        from docling.document_converter import DocumentConverter
        return DocumentConverter()

    def parse(self, pdf_path: Path) -> list[ParsedBlock]:
        result = self._converter.convert(str(pdf_path))
        doc = result.document
        blocks: list[ParsedBlock] = []
        for item, _level in doc.iterate_items():
            kind = _kind_for(item)
            if kind is None:
                continue
            text = _text_of(item)
            if not text.strip():
                continue
            page, bbox = _provenance(item)
            blocks.append(ParsedBlock(page=page, kind=kind, text=text, bbox=bbox))
        return blocks


def _kind_for(item) -> Kind | None:
    label = getattr(item, "label", "") or ""
    name = label.lower() if isinstance(label, str) else str(label).lower()
    if "table" in name:
        return "table"
    if "caption" in name or "figure" in name:
        return "figure_caption"
    if "text" in name or "paragraph" in name or "section" in name or "title" in name or "header" in name or "list" in name:
        return "text"
    return None


def _text_of(item) -> str:
    if hasattr(item, "export_to_markdown"):
        return item.export_to_markdown()
    return getattr(item, "text", "") or ""


def _provenance(item) -> tuple[int, tuple[float, float, float, float] | None]:
    prov = getattr(item, "prov", None) or []
    if not prov:
        return 1, None
    p = prov[0]
    page = getattr(p, "page_no", 1) or 1
    bbox = getattr(p, "bbox", None)
    if bbox is None:
        return page, None
    return page, (float(bbox.l), float(bbox.t), float(bbox.r), float(bbox.b))
