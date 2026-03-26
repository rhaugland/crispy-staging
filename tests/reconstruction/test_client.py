import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from staging.reconstruction.client import (
    ColmapReconstructor,
    ReconstructionResult,
    CameraPoseResult,
)


def _mock_reconstruction():
    """Create a mock pycolmap Reconstruction object."""
    mock_recon = MagicMock()
    mock_recon.num_reg_images.return_value = 3
    mock_recon.num_points3D.return_value = 100

    # Mock cameras
    mock_cam = MagicMock()
    mock_cam.focal_length = 500.0
    mock_cam.width = 1024
    mock_cam.height = 683
    mock_recon.cameras = {1: mock_cam}

    # Mock images
    mock_images = {}
    for i in range(3):
        img = MagicMock()
        img.name = f"{i+1}.jpg"
        img.camera_id = 1

        cfw = MagicMock()
        rot = MagicMock()
        rot.matrix.return_value = np.eye(3)
        cfw.rotation = rot
        cfw.translation = np.array([float(i), 0.0, 0.0])
        img.cam_from_world.return_value = cfw
        img.points2D = []
        mock_images[i] = img

    mock_recon.images = mock_images

    # Mock 3D points
    mock_recon.points3D = {}

    return mock_recon


def test_reconstruct_returns_result():
    recon = ColmapReconstructor(quality="low")
    mock_result = _mock_reconstruction()

    with (
        patch("pycolmap.extract_features"),
        patch("pycolmap.match_exhaustive"),
        patch("pycolmap.incremental_mapping", return_value={"0": mock_result}),
    ):
        result = recon.reconstruct(image_dir=MagicMock())

    assert isinstance(result, ReconstructionResult)
    assert len(result.camera_poses) == 3
    assert result.camera_poses[0].image_name == "1.jpg"


def test_reconstruct_no_result_raises():
    recon = ColmapReconstructor(quality="low")

    with (
        patch("pycolmap.extract_features"),
        patch("pycolmap.match_exhaustive"),
        patch("pycolmap.incremental_mapping", return_value={}),
    ):
        with pytest.raises(RuntimeError, match="no valid reconstruction"):
            recon.reconstruct(image_dir=MagicMock())


def test_camera_pose_extraction():
    recon = ColmapReconstructor(quality="low")
    mock_result = _mock_reconstruction()

    with (
        patch("pycolmap.extract_features"),
        patch("pycolmap.match_exhaustive"),
        patch("pycolmap.incremental_mapping", return_value={"0": mock_result}),
    ):
        result = recon.reconstruct(image_dir=MagicMock())

    pose = result.camera_poses[0]
    assert pose.focal_length == 500.0
    assert pose.image_width == 1024
    assert pose.image_height == 683
    assert pose.position.shape == (3,)
    assert pose.rotation.shape == (3, 3)
