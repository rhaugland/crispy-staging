import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from staging.render.scene import BlenderScene
from staging.models import CameraPose, FurniturePlacement


def test_scene_creation():
    with patch("staging.render.scene.bpy", MagicMock()):
        scene = BlenderScene()
        scene.clear()


def test_import_mesh():
    mock_bpy = MagicMock()
    with patch("staging.render.scene.bpy", mock_bpy):
        scene = BlenderScene()
        scene.import_mesh(mesh_path="/tmp/room.glb")
        mock_bpy.ops.import_scene.gltf.assert_called_once()


def test_place_furniture():
    mock_bpy = MagicMock()
    with patch("staging.render.scene.bpy", mock_bpy):
        scene = BlenderScene()
        placement = FurniturePlacement(
            asset_id="sofa-01",
            position=np.array([2.0, 0.0, 1.5]),
            rotation_y=90.0,
        )
        scene.place_furniture(
            placement=placement,
            asset_path="/tmp/assets/sofa-01.glb",
        )
        mock_bpy.ops.import_scene.gltf.assert_called()


def test_set_camera():
    mock_bpy = MagicMock()
    with patch("staging.render.scene.bpy", mock_bpy):
        scene = BlenderScene()
        pose = CameraPose(
            position=np.array([1.0, 1.5, 0.0]),
            rotation=np.eye(3),
            focal_length=35.0,
            image_width=1920,
            image_height=1080,
        )
        scene.set_camera(pose)
