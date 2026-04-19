from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MMRAG_", env_file=".env", extra="ignore")

    colpali_model: str = "vidore/colpali-v1.3"
    colpali_device: str = "mps"
    pdf_render_dpi: int = 150
    bge_model: str = "BAAI/bge-small-en-v1.5"
    dense_dim: int = 384
    chunk_max_chars: int = 1200
    gemini_model: str = "gemini-2.5-flash"
    gemini_api_key: str | None = Field(default=None, validation_alias="GEMINI_API_KEY")
    answer_render_dpi: int = 150
    data_dir: Path = Field(default_factory=lambda: Path("data"))

    @property
    def embeddings_dir(self) -> Path:
        return self.data_dir / "embeddings"

    @property
    def manifest_path(self) -> Path:
        return self.data_dir / "manifest.json"

    @property
    def qdrant_path(self) -> Path:
        return self.data_dir / "qdrant"

    @property
    def bm25_path(self) -> Path:
        return self.data_dir / "bm25.pkl"
