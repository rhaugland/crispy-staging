# src/staging/analysis/floor.py
from __future__ import annotations

import numpy as np

from staging.models import CameraPose, FloorPlan


class FloorSegmenter:
    def segment_floor(self, image: np.ndarray) -> np.ndarray:
        return self._run_sam(image)

    def _run_sam(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Wire up SAM 2 / Grounded-SAM")

    def merge_views(
        self,
        masks: list[np.ndarray],
        poses: list[CameraPose],
        plane_normal: np.ndarray,
        plane_offset: float,
    ) -> FloorPlan:
        polygon = self._project_and_merge(masks, poses, plane_normal, plane_offset)
        return FloorPlan(
            polygon=polygon,
            plane_normal=plane_normal,
            plane_offset=plane_offset,
        )

    def _project_and_merge(
        self,
        masks: list[np.ndarray],
        poses: list[CameraPose],
        plane_normal: np.ndarray,
        plane_offset: float,
    ) -> np.ndarray:
        raise NotImplementedError("Wire up projection + convex hull")
