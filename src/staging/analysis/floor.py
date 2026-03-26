# src/staging/analysis/floor.py
from __future__ import annotations

import logging

import numpy as np
from scipy.spatial import ConvexHull

from staging.models import CameraPose, FloorPlan

logger = logging.getLogger(__name__)

# SAM 2 — only available on GPU workers
try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except ImportError:
    build_sam2 = None
    SAM2ImagePredictor = None


class FloorSegmenter:
    def __init__(self):
        self._predictor = None

    def _load_model(self):
        if self._predictor is not None:
            return
        if SAM2ImagePredictor is None:
            raise ImportError(
                "SAM 2 is required for floor segmentation. "
                "Install with: pip install segment-anything-2"
            )
        model = build_sam2("sam2_hiera_s", "sam2_hiera_small.pt")
        self._predictor = SAM2ImagePredictor(model)

    def segment_floor(self, image: np.ndarray) -> np.ndarray:
        """Segment the floor region from an image.

        Args:
            image: (H, W, 3) RGB uint8 image.

        Returns:
            Binary mask (H, W) uint8, 255 = floor.
        """
        self._load_model()
        h, w = image.shape[:2]

        self._predictor.set_image(image)

        # Seed points: center-bottom region (likely floor)
        points = np.array([
            [w // 2, int(h * 0.85)],       # center bottom
            [w // 4, int(h * 0.80)],       # left bottom
            [3 * w // 4, int(h * 0.80)],   # right bottom
        ])
        labels = np.array([1, 1, 1])  # all foreground (floor)

        masks, scores, _ = self._predictor.predict(
            point_coords=points,
            point_labels=labels,
            multimask_output=True,
        )

        # Pick the mask with highest score
        best_idx = int(np.argmax(scores))
        mask = masks[best_idx]

        return (mask * 255).astype(np.uint8)

    def merge_views(
        self,
        masks: list[np.ndarray],
        poses: list[CameraPose],
        plane_normal: np.ndarray,
        plane_offset: float,
    ) -> FloorPlan:
        """Project floor masks from multiple views onto the ground plane and merge."""
        polygon = self._project_and_merge(masks, poses, plane_normal, plane_offset)
        return FloorPlan(
            polygon=polygon,
            plane_normal=plane_normal,
            plane_offset=plane_offset,
        )

    def _project_and_merge(
        self,
        masks: list[np.ndarray],
        poses: list[CameraPose],
        plane_normal: np.ndarray,
        plane_offset: float,
    ) -> np.ndarray:
        """Project floor mask pixels to the ground plane and compute convex hull.

        For each mask pixel that is floor, cast a ray from the camera through
        that pixel and intersect with the ground plane (dot(n, p) = offset).
        Collect all intersection points and return the convex hull.
        """
        all_points = []

        for mask, pose in zip(masks, poses):
            h, w = mask.shape[:2]
            fx = pose.focal_length
            cx = pose.image_width / 2.0
            cy = pose.image_height / 2.0
            R = pose.rotation       # (3, 3) world-to-camera
            C = pose.position       # (3,) camera center in world

            # Sample floor pixels (subsample for speed)
            ys, xs = np.where(mask > 127)
            if len(ys) == 0:
                continue

            step = max(1, len(ys) // 500)  # keep ~500 samples
            xs = xs[::step]
            ys = ys[::step]

            # Pixel to camera-frame ray directions
            dirs_cam = np.stack([
                (xs - cx) / fx,
                (ys - cy) / fx,
                np.ones_like(xs, dtype=float),
            ], axis=1)  # (N, 3)

            # Transform to world frame: ray_world = R^T @ ray_cam
            dirs_world = (R.T @ dirs_cam.T).T  # (N, 3)

            # Ray-plane intersection: t = (offset - dot(n, C)) / dot(n, d)
            n = plane_normal
            denom = dirs_world @ n  # (N,)
            numer = plane_offset - np.dot(n, C)

            valid = np.abs(denom) > 1e-8
            t = np.full(len(denom), np.inf)
            t[valid] = numer / denom[valid]

            # Only forward intersections
            valid &= t > 0

            if not np.any(valid):
                continue

            # Compute 3D intersection points
            points_3d = C[None, :] + t[valid, None] * dirs_world[valid]

            # Project to 2D on the ground plane
            # Use the two axes perpendicular to the normal
            if abs(n[1]) > 0.9:  # Y-up
                points_2d = points_3d[:, [0, 2]]  # XZ plane
            elif abs(n[2]) > 0.9:  # Z-up
                points_2d = points_3d[:, [0, 1]]  # XY plane
            else:
                points_2d = points_3d[:, [0, 2]]  # default XZ

            all_points.append(points_2d)

        if not all_points:
            logger.warning("No floor points projected from any view")
            return np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)

        merged = np.vstack(all_points)

        # Compute convex hull
        if len(merged) < 3:
            return merged.astype(float)

        try:
            hull = ConvexHull(merged)
            return merged[hull.vertices].astype(float)
        except Exception:
            return merged[:4].astype(float)
