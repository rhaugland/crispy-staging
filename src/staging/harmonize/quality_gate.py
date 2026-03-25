from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from staging.config import settings


@dataclass
class QualityCheckResult:
    passed: bool
    issues: list[str] = field(default_factory=list)
    failed_indices: list[int] = field(default_factory=list)


class QualityGate:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.max_retries = settings.quality_gate_max_retries

    async def check(self, staged_images: list[np.ndarray], original_images: list[np.ndarray]) -> QualityCheckResult:
        return await self._check_with_vision(staged_images, original_images)

    async def _check_with_vision(self, staged: list[np.ndarray], originals: list[np.ndarray]) -> QualityCheckResult:
        raise NotImplementedError("Wire up Claude Vision quality check")

    def _adjust_params(self, attempt: int) -> dict:
        if attempt == 0:
            return {"color_temp_shift": 0, "exposure_shift": 0.0, "mask_expand_px": 0}
        if attempt == 1:
            return {"color_temp_shift": 500, "exposure_shift": 0.5, "mask_expand_px": 10}
        return {"color_temp_shift": -500, "exposure_shift": -0.5, "mask_expand_px": 10}

    def should_retry(self, attempt: int) -> bool:
        return attempt < self.max_retries
