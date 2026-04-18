from pathlib import Path

import pytest

from tests.fixtures.build_sample_pdfs import ensure_sample_pdfs


@pytest.fixture(scope="session")
def sample_pdfs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("sample_pdfs")
    return ensure_sample_pdfs(root)
