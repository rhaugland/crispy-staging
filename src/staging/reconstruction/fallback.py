from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from staging.models import CameraPose

MIN_FEATURE_OVERLAP = 0.30


class AlignmentError(Exception):
    pass


@dataclass
class FallbackResult:
    camera_poses: list[CameraPose]
    floor_plane_normal: np.ndarray
    floor_plane_offset: float
    feature_overlap_pct: float


class DepthFallback:
    """Fallback reconstruction using DepthAnything v2 when SfM fails."""

    def estimate_depth(self, image: np.ndarray) -> np.ndarray:
        return self._run_depth_model(image)

    def _run_depth_model(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Wire up DepthAnything v2 model")

    def align_views(
        self,
        images: list[np.ndarray],
        depths: list[np.ndarray],
    ) -> FallbackResult:
        result = self._align_from_depths(images, depths)

        if result.feature_overlap_pct < MIN_FEATURE_OVERLAP:
            raise AlignmentError(
                f"Feature overlap {result.feature_overlap_pct:.0%} is below "
                f"minimum {MIN_FEATURE_OVERLAP:.0%}. Photos don't share enough "
                "visual content. Please re-shoot with more overlap between angles."
            )

        return result

    def _align_from_depths(
        self,
        images: list[np.ndarray],
        depths: list[np.ndarray],
    ) -> FallbackResult:
        raise NotImplementedError("Wire up feature matching + pose estimation")
