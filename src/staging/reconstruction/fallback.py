from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from staging.models import CameraPose

logger = logging.getLogger(__name__)

MIN_FEATURE_OVERLAP = 0.30

# DepthAnything v2 — only available on GPU workers
try:
    from transformers import pipeline as hf_pipeline
except ImportError:
    hf_pipeline = None


class AlignmentError(Exception):
    pass


@dataclass
class FallbackResult:
    camera_poses: list[CameraPose]
    floor_plane_normal: np.ndarray
    floor_plane_offset: float
    feature_overlap_pct: float


class DepthFallback:
    """Fallback reconstruction using DepthAnything v2 when SfM fails."""

    def __init__(self):
        self._depth_pipe = None

    def _load_model(self):
        if self._depth_pipe is not None:
            return
        if hf_pipeline is None:
            raise ImportError(
                "transformers is required for depth estimation. "
                "Install with: pip install transformers torch"
            )
        self._depth_pipe = hf_pipeline(
            "depth-estimation",
            model="depth-anything/Depth-Anything-V2-Small-hf",
            device="cuda" if self._has_cuda() else "cpu",
        )

    @staticmethod
    def _has_cuda() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def estimate_depth(self, image: np.ndarray) -> np.ndarray:
        """Estimate relative depth from a single image.

        Args:
            image: (H, W, 3) RGB uint8.

        Returns:
            (H, W) float32 relative depth map.
        """
        self._load_model()
        from PIL import Image

        pil_img = Image.fromarray(image)
        result = self._depth_pipe(pil_img)
        depth = np.array(result["depth"], dtype=np.float32)

        # Normalize to [0, 1]
        d_min, d_max = depth.min(), depth.max()
        if d_max - d_min > 0:
            depth = (depth - d_min) / (d_max - d_min)

        return depth

    def align_views(
        self,
        images: list[np.ndarray],
        depths: list[np.ndarray],
    ) -> FallbackResult:
        """Align multiple depth maps using feature matching.

        Raises AlignmentError if feature overlap is below threshold.
        """
        result = self._align_from_depths(images, depths)

        if result.feature_overlap_pct < MIN_FEATURE_OVERLAP:
            raise AlignmentError(
                f"Feature overlap {result.feature_overlap_pct:.0%} is below "
                f"minimum {MIN_FEATURE_OVERLAP:.0%}. Photos don't share enough "
                "visual content. Please re-shoot with more overlap between angles."
            )

        return result

    def _align_from_depths(
        self,
        images: list[np.ndarray],
        depths: list[np.ndarray],
    ) -> FallbackResult:
        """Align views using ORB feature matching + depth-informed pose estimation."""
        try:
            import cv2
        except ImportError:
            raise ImportError("OpenCV is required for fallback alignment: pip install opencv-python")

        orb = cv2.ORB_create(nfeatures=2000)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Extract features from all images
        all_kps = []
        all_descs = []
        for img in images:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            kps, descs = orb.detectAndCompute(gray, None)
            all_kps.append(kps)
            all_descs.append(descs)

        # Match pairs and compute overlap
        total_matches = 0
        total_possible = 0
        camera_poses = []

        # First camera is at origin
        h, w = images[0].shape[:2]
        focal_est = max(w, h) * 1.2  # rough focal length estimate

        camera_poses.append(CameraPose(
            position=np.array([0.0, 0.0, 0.0]),
            rotation=np.eye(3),
            focal_length=focal_est,
            image_width=w,
            image_height=h,
        ))

        for i in range(1, len(images)):
            desc0, desc1 = all_descs[0], all_descs[i]
            if desc0 is None or desc1 is None:
                camera_poses.append(CameraPose(
                    position=np.array([float(i), 0.0, 0.0]),
                    rotation=np.eye(3),
                    focal_length=focal_est,
                    image_width=w,
                    image_height=h,
                ))
                continue

            matches = bf.match(desc0, desc1)
            total_matches += len(matches)
            total_possible += min(len(desc0), len(desc1))

            if len(matches) >= 8:
                # Estimate relative pose using essential matrix
                pts0 = np.float32([all_kps[0][m.queryIdx].pt for m in matches])
                pts1 = np.float32([all_kps[i][m.trainIdx].pt for m in matches])

                K = np.array([[focal_est, 0, w / 2], [0, focal_est, h / 2], [0, 0, 1]])
                E, mask = cv2.findEssentialMat(pts0, pts1, K, method=cv2.RANSAC, threshold=1.0)

                if E is not None:
                    _, R, t, _ = cv2.recoverPose(E, pts0, pts1, K)
                    position = -R.T @ t.flatten()
                    camera_poses.append(CameraPose(
                        position=position,
                        rotation=R,
                        focal_length=focal_est,
                        image_width=w,
                        image_height=h,
                    ))
                    continue

            # Fallback: simple offset
            camera_poses.append(CameraPose(
                position=np.array([float(i), 0.0, 0.0]),
                rotation=np.eye(3),
                focal_length=focal_est,
                image_width=w,
                image_height=h,
            ))

        overlap = total_matches / max(total_possible, 1)

        return FallbackResult(
            camera_poses=camera_poses,
            floor_plane_normal=np.array([0.0, 1.0, 0.0]),  # assume Y-up
            floor_plane_offset=0.0,
            feature_overlap_pct=overlap,
        )
