from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MMRAG_", env_file=".env", extra="ignore")

    colpali_model: str = "vidore/colpali-v1.3"
    colpali_device: str = "mps"
    pdf_render_dpi: int = 150
    data_dir: Path = Field(default_factory=lambda: Path("data"))

    @property
    def embeddings_dir(self) -> Path:
        return self.data_dir / "embeddings"

    @property
    def manifest_path(self) -> Path:
        return self.data_dir / "manifest.json"
