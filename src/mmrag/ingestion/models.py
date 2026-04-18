from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, field_validator

PATCH_DIM = 128


class PageEmbedding(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    doc_id: str
    page: int
    patches: np.ndarray
    pooled: np.ndarray

    @field_validator("patches")
    @classmethod
    def _validate_patches(cls, v: Any) -> np.ndarray:
        if not isinstance(v, np.ndarray) or v.ndim != 2 or v.shape[1] != PATCH_DIM:
            raise ValueError(f"patches must be (N, {PATCH_DIM}) ndarray, got {getattr(v, 'shape', v)}")
        return v

    @field_validator("pooled")
    @classmethod
    def _validate_pooled(cls, v: Any) -> np.ndarray:
        if not isinstance(v, np.ndarray) or v.shape != (PATCH_DIM,):
            raise ValueError(f"pooled must be ({PATCH_DIM},) ndarray, got {getattr(v, 'shape', v)}")
        return v
