from pathlib import Path

from mmrag.corpus.hashing import sha256_file


def test_sha256_file_is_deterministic(tmp_path: Path) -> None:
    f = tmp_path / "x.bin"
    f.write_bytes(b"hello world")

    assert sha256_file(f) == sha256_file(f)


def test_sha256_file_matches_known_value(tmp_path: Path) -> None:
    f = tmp_path / "x.txt"
    f.write_bytes(b"hello world")

    expected = "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"

    assert sha256_file(f) == expected


def test_sha256_file_differs_for_different_content(tmp_path: Path) -> None:
    a = tmp_path / "a.bin"
    b = tmp_path / "b.bin"
    a.write_bytes(b"alpha")
    b.write_bytes(b"beta")

    assert sha256_file(a) != sha256_file(b)
