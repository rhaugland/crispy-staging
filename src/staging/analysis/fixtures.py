# src/staging/analysis/fixtures.py
from __future__ import annotations

import logging

import numpy as np

from staging.models import CameraPose, ExclusionZone, FloorPlan

logger = logging.getLogger(__name__)

FIXTURE_LABELS = ["window", "door", "fireplace", "built-in shelving", "radiator", "vent"]

# Grounded-SAM — only available on GPU workers
try:
    from groundingdino.util.inference import load_model as load_gdino, predict as gdino_predict
except ImportError:
    load_gdino = None
    gdino_predict = None


class FixtureDetector:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return
        if load_gdino is None:
            raise ImportError(
                "GroundingDINO is required for fixture detection. "
                "Install from: https://github.com/IDEA-Research/GroundingDINO"
            )
        self._model = load_gdino(
            "groundingdino/config/GroundingDINO_SwinT_OGC.py",
            "weights/groundingdino_swint_ogc.pth",
        )

    async def detect(
        self,
        images: list[np.ndarray],
        poses: list[CameraPose | None],
        floor_plane_normal: np.ndarray,
        floor_plane_offset: float,
    ) -> list[ExclusionZone]:
        return await self._detect_and_project(
            images, poses, floor_plane_normal, floor_plane_offset
        )

    async def _detect_and_project(
        self,
        images: list[np.ndarray],
        poses: list[CameraPose | None],
        floor_plane_normal: np.ndarray,
        floor_plane_offset: float,
    ) -> list[ExclusionZone]:
        self._load_model()

        prompt = " . ".join(FIXTURE_LABELS) + " ."
        zones = []

        for img, pose in zip(images, poses):
            if pose is None:
                continue

            from PIL import Image as PILImage
            import torchvision.transforms as T

            pil_img = PILImage.fromarray(img)
            transform = T.Compose([
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            img_tensor = transform(pil_img)

            boxes, logits, phrases = gdino_predict(
                model=self._model,
                image=img_tensor,
                caption=prompt,
                box_threshold=0.3,
                text_threshold=0.25,
            )

            h, w = img.shape[:2]
            for box, phrase in zip(boxes, phrases):
                # box is [cx, cy, bw, bh] normalized
                cx, cy, bw, bh = box.tolist()
                x1 = (cx - bw / 2) * w
                x2 = (cx + bw / 2) * w
                y_bottom = (cy + bh / 2) * h

                # Project bottom edge of bbox to floor plane
                polygon = self._project_bbox_to_floor(
                    x1, x2, y_bottom, pose, floor_plane_normal, floor_plane_offset
                )

                if polygon is not None:
                    zones.append(ExclusionZone(
                        label=phrase,
                        polygon=polygon,
                    ))

        return zones

    @staticmethod
    def _project_bbox_to_floor(
        x1: float,
        x2: float,
        y_bottom: float,
        pose: CameraPose,
        plane_normal: np.ndarray,
        plane_offset: float,
    ) -> np.ndarray | None:
        """Project the bottom edge of a 2D bounding box onto the floor plane."""
        fx = pose.focal_length
        cx = pose.image_width / 2.0
        cy = pose.image_height / 2.0
        R = pose.rotation
        C = pose.position
        n = plane_normal

        points_2d = []
        for x in [x1, x2]:
            # Pixel to camera-frame ray
            dir_cam = np.array([(x - cx) / fx, (y_bottom - cy) / fx, 1.0])
            dir_world = R.T @ dir_cam

            denom = np.dot(dir_world, n)
            if abs(denom) < 1e-8:
                continue

            t = (plane_offset - np.dot(n, C)) / denom
            if t <= 0:
                continue

            point_3d = C + t * dir_world

            # Project to 2D on ground plane
            if abs(n[1]) > 0.9:
                points_2d.append([point_3d[0], point_3d[2]])
            else:
                points_2d.append([point_3d[0], point_3d[2]])

        if len(points_2d) < 2:
            return None

        # Create a rectangle from the two projected points with some depth
        p1, p2 = np.array(points_2d[0]), np.array(points_2d[1])
        edge = p2 - p1
        perp = np.array([-edge[1], edge[0]])
        if np.linalg.norm(perp) > 0:
            perp = perp / np.linalg.norm(perp) * 0.3  # 30cm depth

        return np.array([p1, p2, p2 + perp, p1 + perp], dtype=float)


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
