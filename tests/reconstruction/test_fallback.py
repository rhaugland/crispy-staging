import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from staging.reconstruction.fallback import (
    DepthFallback,
    FallbackResult,
    AlignmentError,
)


def test_estimate_depth_returns_depth_map():
    fallback = DepthFallback()
    fake_depth = np.random.rand(1080, 1920).astype(np.float32)
    with patch.object(fallback, "_run_depth_model", return_value=fake_depth):
        depth = fallback.estimate_depth(image=np.zeros((1080, 1920, 3), dtype=np.uint8))
    assert depth.shape == (1080, 1920)


def test_align_views_success():
    fallback = DepthFallback()
    depths = [np.random.rand(1080, 1920).astype(np.float32) for _ in range(3)]
    images = [np.zeros((1080, 1920, 3), dtype=np.uint8) for _ in range(3)]
    fake_result = FallbackResult(
        camera_poses=[MagicMock() for _ in range(3)],
        floor_plane_normal=np.array([0, 1, 0]),
        floor_plane_offset=0.0,
        feature_overlap_pct=0.45,
    )
    with patch.object(fallback, "_align_from_depths", return_value=fake_result):
        result = fallback.align_views(images=images, depths=depths)
    assert result.feature_overlap_pct >= 0.3


def test_align_views_low_overlap_raises():
    fallback = DepthFallback()
    depths = [np.random.rand(1080, 1920).astype(np.float32) for _ in range(3)]
    images = [np.zeros((1080, 1920, 3), dtype=np.uint8) for _ in range(3)]
    fake_result = FallbackResult(
        camera_poses=[],
        floor_plane_normal=np.array([0, 1, 0]),
        floor_plane_offset=0.0,
        feature_overlap_pct=0.15,
    )
    with patch.object(fallback, "_align_from_depths", return_value=fake_result):
        with pytest.raises(AlignmentError, match="overlap"):
            fallback.align_views(images=images, depths=depths)
