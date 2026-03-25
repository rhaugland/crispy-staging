import numpy as np
import pytest
from staging.reconstruction.quality import (
    validate_reconstruction,
    ReconstructionQualityError,
    compute_reprojection_error,
)
from staging.models import CameraPose, FloorPlan


def _make_pose(pos, focal=35.0):
    return CameraPose(
        position=np.array(pos, dtype=float),
        rotation=np.eye(3),
        focal_length=focal,
        image_width=1920,
        image_height=1080,
    )


def test_valid_reconstruction():
    poses = [_make_pose([0, 0, 0]), _make_pose([2, 0, 0]), _make_pose([1, 0, 2])]
    floor = FloorPlan(
        polygon=np.array([[0, 0], [4, 0], [4, 3], [0, 3]], dtype=float),
        plane_normal=np.array([0, 1, 0], dtype=float),
        plane_offset=0.0,
    )
    result = validate_reconstruction(poses=poses, floor=floor, reprojection_errors=[1.2, 0.8, 2.1])
    assert result.passed is True


def test_reprojection_error_too_high():
    poses = [_make_pose([0, 0, 0]), _make_pose([2, 0, 0]), _make_pose([1, 0, 2])]
    floor = FloorPlan(
        polygon=np.array([[0, 0], [4, 0], [4, 3], [0, 3]], dtype=float),
        plane_normal=np.array([0, 1, 0], dtype=float),
        plane_offset=0.0,
    )
    with pytest.raises(ReconstructionQualityError, match="reprojection"):
        validate_reconstruction(poses=poses, floor=floor, reprojection_errors=[6.0, 7.0, 8.0])


def test_floor_plan_too_small():
    poses = [_make_pose([0, 0, 0]), _make_pose([2, 0, 0]), _make_pose([1, 0, 2])]
    floor = FloorPlan(
        polygon=np.array([[0, 0], [0.1, 0], [0.1, 0.1], [0, 0.1]], dtype=float),
        plane_normal=np.array([0, 1, 0], dtype=float),
        plane_offset=0.0,
    )
    with pytest.raises(ReconstructionQualityError, match="floor"):
        validate_reconstruction(poses=poses, floor=floor, reprojection_errors=[1.0, 1.0, 1.0])
