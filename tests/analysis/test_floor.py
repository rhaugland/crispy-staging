# tests/analysis/test_floor.py
import numpy as np
import pytest
from unittest.mock import patch
from staging.analysis.floor import FloorSegmenter
from staging.models import CameraPose, FloorPlan


def _make_pose(pos):
    return CameraPose(
        position=np.array(pos, dtype=float),
        rotation=np.eye(3),
        focal_length=35.0,
        image_width=1920,
        image_height=1080,
    )


def test_segment_single_image_returns_mask():
    seg = FloorSegmenter()
    fake_mask = (np.zeros((1080, 1920), dtype=np.uint8))
    fake_mask[540:1080, :] = 255
    with patch.object(seg, "segment_floor", return_value=fake_mask):
        mask = seg.segment_floor(image=np.zeros((1080, 1920, 3), dtype=np.uint8))
    assert mask.shape == (1080, 1920)
    assert mask.sum() > 0


def test_merge_multi_view_produces_polygon():
    seg = FloorSegmenter()
    masks = [np.ones((1080, 1920), dtype=bool) for _ in range(3)]
    poses = [_make_pose([0, 0, 0]), _make_pose([2, 0, 0]), _make_pose([1, 0, 2])]
    plane_normal = np.array([0, 1, 0], dtype=float)

    fake_polygon = np.array([[0, 0], [4, 0], [4, 3], [0, 3]], dtype=float)
    with patch.object(seg, "_project_and_merge", return_value=fake_polygon):
        floor_plan = seg.merge_views(
            masks=masks, poses=poses, plane_normal=plane_normal, plane_offset=0.0
        )
    assert isinstance(floor_plan, FloorPlan)
    assert floor_plan.area() > 0
