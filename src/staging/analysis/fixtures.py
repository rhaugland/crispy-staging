# src/staging/analysis/fixtures.py
from __future__ import annotations

import numpy as np

from staging.models import CameraPose, ExclusionZone, FloorPlan

FIXTURE_LABELS = ["window", "door", "fireplace", "built-in shelving", "radiator", "vent"]


class FixtureDetector:
    def __init__(self, api_key: str):
        self.api_key = api_key

    async def detect(
        self,
        images: list[np.ndarray],
        poses: list[CameraPose | None],
        floor_plane_normal: np.ndarray,
        floor_plane_offset: float,
    ) -> list[ExclusionZone]:
        return await self._detect_and_project(images, poses, floor_plane_normal, floor_plane_offset)

    async def _detect_and_project(
        self,
        images: list[np.ndarray],
        poses: list[CameraPose | None],
        floor_plane_normal: np.ndarray,
        floor_plane_offset: float,
    ) -> list[ExclusionZone]:
        raise NotImplementedError("Wire up Grounded-SAM fixture detection")


def compute_placeable_ratio(floor: FloorPlan, zones: list[ExclusionZone]) -> float:
    total = floor.area()
    if total == 0:
        return 0.0

    excluded = 0.0
    for zone in zones:
        pts = zone.polygon
        n = len(pts)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += pts[i][0] * pts[j][1]
            area -= pts[j][0] * pts[i][1]
        excluded += abs(area) / 2.0

    return max(0.0, (total - excluded) / total)
