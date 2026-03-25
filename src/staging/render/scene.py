from __future__ import annotations
import math
from pathlib import Path
import numpy as np

try:
    import bpy
except ImportError:
    bpy = None

from staging.models import CameraPose, FurniturePlacement


class BlenderScene:
    def clear(self):
        bpy.ops.object.select_all(action="SELECT")
        bpy.ops.object.delete()

    def import_mesh(self, mesh_path: str):
        bpy.ops.import_scene.gltf(filepath=mesh_path)

    def place_furniture(self, placement: FurniturePlacement, asset_path: str):
        bpy.ops.import_scene.gltf(filepath=asset_path)
        obj = bpy.context.selected_objects[0]
        obj.location = (
            float(placement.position[0]),
            float(placement.position[2]),
            float(placement.position[1]),
        )
        obj.rotation_euler[2] = math.radians(placement.rotation_y)
        obj.scale = (placement.scale, placement.scale, placement.scale)

    def set_camera(self, pose: CameraPose):
        cam_data = bpy.data.cameras.new("StagingCamera")
        cam_data.lens = pose.focal_length
        cam_data.sensor_width = 36.0
        cam_obj = bpy.data.objects.new("StagingCamera", cam_data)
        bpy.context.scene.collection.objects.link(cam_obj)
        bpy.context.scene.camera = cam_obj
        cam_obj.location = (
            float(pose.position[0]),
            float(pose.position[2]),
            float(pose.position[1]),
        )
        rot = pose.rotation
        cam_obj.rotation_euler = self._matrix_to_euler(rot)
        bpy.context.scene.render.resolution_x = pose.image_width
        bpy.context.scene.render.resolution_y = pose.image_height

    def _matrix_to_euler(self, rot: np.ndarray) -> tuple[float, float, float]:
        sy = math.sqrt(rot[0, 0] ** 2 + rot[1, 0] ** 2)
        singular = sy < 1e-6
        if not singular:
            x = math.atan2(rot[2, 1], rot[2, 2])
            y = math.atan2(-rot[2, 0], sy)
            z = math.atan2(rot[1, 0], rot[0, 0])
        else:
            x = math.atan2(-rot[1, 2], rot[1, 1])
            y = math.atan2(-rot[2, 0], sy)
            z = 0
        return (x, y, z)
