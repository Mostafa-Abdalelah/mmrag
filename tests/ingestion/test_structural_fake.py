from pathlib import Path

from mmrag.ingestion.structural import ParsedBlock, Parser


class FakeParser:
    def parse(self, pdf_path: Path) -> list[ParsedBlock]:
        return [
            ParsedBlock(page=1, kind="text", text="hello", bbox=(0.0, 0.0, 100.0, 20.0)),
            ParsedBlock(page=1, kind="table", text="| a | b |\n|---|---|\n| 1 | 2 |",
                        bbox=(0.0, 40.0, 100.0, 120.0)),
        ]


def test_fake_parser_satisfies_protocol() -> None:
    assert isinstance(FakeParser(), Parser)


def test_fake_parser_returns_blocks(tmp_path: Path) -> None:
    blocks = FakeParser().parse(tmp_path / "x.pdf")
    assert len(blocks) == 2
    assert blocks[0].kind == "text"
    assert blocks[1].kind == "table"
