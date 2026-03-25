from __future__ import annotations
import math
from typing import NamedTuple

import numpy as np

try:
    import bpy
except ImportError:
    bpy = None


class LightEstimate(NamedTuple):
    direction: np.ndarray  # unit vector (3,) pointing toward light source
    color: tuple[float, float, float]  # linear RGB, each in [0, 1]
    strength: float  # watts / area, Cycles energy units


class LightingEstimator:
    """Estimate scene lighting from an environment map or a simple heuristic,
    then set up Blender world and sun lights accordingly."""

    DEFAULT_STRENGTH = 5.0
    DEFAULT_COLOR = (1.0, 0.98, 0.95)  # warm white

    def estimate_from_image(self, image: np.ndarray) -> LightEstimate:
        """Rough brightest-region heuristic on an HDR or LDR numpy image (H, W, 3)."""
        gray = image.mean(axis=2)
        flat_idx = int(np.argmax(gray))
        h, w = gray.shape
        row, col = divmod(flat_idx, w)
        # Convert pixel coords to spherical direction (equirectangular assumed)
        theta = math.pi * (row / h)        # polar angle [0, pi]
        phi = 2.0 * math.pi * (col / w)   # azimuth [0, 2pi]
        dx = math.sin(theta) * math.cos(phi)
        dy = math.sin(theta) * math.sin(phi)
        dz = math.cos(theta)
        direction = np.array([dx, dy, dz], dtype=float)
        mean_brightness = float(gray[row, col])
        strength = max(1.0, mean_brightness / 255.0 * 10.0)
        return LightEstimate(
            direction=direction,
            color=self.DEFAULT_COLOR,
            strength=strength,
        )

    def apply_to_scene(self, estimate: LightEstimate) -> None:
        """Create a sun lamp in the active Blender scene from a LightEstimate."""
        # World ambient
        world = bpy.context.scene.world
        if world is None:
            world = bpy.data.worlds.new("StagingWorld")
            bpy.context.scene.world = world
        world.use_nodes = True
        bg_node = world.node_tree.nodes.get("Background")
        if bg_node is not None:
            bg_node.inputs["Strength"].default_value = 0.5

        # Sun lamp
        lamp_data = bpy.data.lights.new(name="StagingSun", type="SUN")
        lamp_data.energy = estimate.strength
        lamp_data.color = estimate.color
        lamp_obj = bpy.data.objects.new("StagingSun", lamp_data)
        bpy.context.scene.collection.objects.link(lamp_obj)

        # Orient the sun lamp opposite to the light direction (points toward scene)
        d = estimate.direction
        lamp_obj.rotation_euler = (
            math.atan2(math.sqrt(d[0] ** 2 + d[1] ** 2), -d[2]),
            0.0,
            math.atan2(d[1], d[0]),
        )

    def apply_default(self) -> None:
        """Apply a neutral three-point lighting rig when no image is available."""
        self.apply_to_scene(
            LightEstimate(
                direction=np.array([0.5, -0.5, 0.7]),
                color=self.DEFAULT_COLOR,
                strength=self.DEFAULT_STRENGTH,
            )
        )
