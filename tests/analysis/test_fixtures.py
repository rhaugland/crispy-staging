# tests/analysis/test_fixtures.py
import numpy as np
import pytest
from unittest.mock import patch, AsyncMock
from staging.analysis.fixtures import FixtureDetector, compute_placeable_ratio
from staging.models import ExclusionZone, FloorPlan


@pytest.mark.asyncio
async def test_detect_returns_exclusion_zones():
    detector = FixtureDetector(api_key="test-key")
    fake_zones = [
        ExclusionZone(label="window", polygon=np.array([[1, 0], [3, 0], [3, 0.5], [1, 0.5]])),
        ExclusionZone(label="door", polygon=np.array([[0, 1], [0.8, 1], [0.8, 2], [0, 2]])),
    ]
    with patch.object(detector, "_detect_and_project", new_callable=AsyncMock, return_value=fake_zones):
        zones = await detector.detect(
            images=[np.zeros((1080, 1920, 3), dtype=np.uint8)],
            poses=[None],
            floor_plane_normal=np.array([0, 1, 0]),
            floor_plane_offset=0.0,
        )
    assert len(zones) == 2
    assert zones[0].label == "window"


@pytest.mark.asyncio
async def test_empty_room_no_fixtures():
    detector = FixtureDetector(api_key="test-key")
    with patch.object(detector, "_detect_and_project", new_callable=AsyncMock, return_value=[]):
        zones = await detector.detect(
            images=[np.zeros((1080, 1920, 3), dtype=np.uint8)],
            poses=[None],
            floor_plane_normal=np.array([0, 1, 0]),
            floor_plane_offset=0.0,
        )
    assert len(zones) == 0


def test_compute_placeable_area_ratio():
    floor = FloorPlan(
        polygon=np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float),
        plane_normal=np.array([0, 1, 0]),
        plane_offset=0.0,
    )
    zones = [
        ExclusionZone(label="door", polygon=np.array([[0, 0], [2, 0], [2, 2], [0, 2]], dtype=float)),
    ]
    ratio = compute_placeable_ratio(floor, zones)
    assert 0.9 < ratio < 1.0
