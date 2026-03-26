"""COLMAP-based 3D reconstruction from multi-view photos."""
from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# pycolmap is optional — only available when running reconstruction
try:
    import pycolmap
except ImportError:
    pycolmap = None


@dataclass
class ReconstructionResult:
    """Result of running SfM reconstruction."""
    camera_poses: list[CameraPoseResult]
    points3d: np.ndarray  # (N, 3) sparse point cloud
    reprojection_errors: list[float]  # per-image mean reprojection error


@dataclass
class CameraPoseResult:
    """Recovered camera pose for one image."""
    image_name: str
    position: np.ndarray      # (3,) camera center in world coords
    rotation: np.ndarray      # (3, 3) rotation matrix (world-to-camera)
    focal_length: float
    image_width: int
    image_height: int


class ColmapReconstructor:
    """Runs COLMAP SfM pipeline via pycolmap to recover camera poses and sparse geometry."""

    def __init__(self, quality: str = "medium"):
        """
        Args:
            quality: SIFT extraction quality — "low", "medium", or "high".
                     Higher = more features, slower.
        """
        if pycolmap is None:
            raise ImportError("pycolmap is required for reconstruction. Install with: pip install pycolmap")
        self.quality = quality

    def reconstruct(self, image_dir: Path, workspace: Path | None = None) -> ReconstructionResult:
        """Run full SfM pipeline on images in a directory.

        Args:
            image_dir: Directory containing input images (JPEG/PNG).
            workspace: Working directory for COLMAP database and outputs.
                       If None, uses a temp directory.

        Returns:
            ReconstructionResult with camera poses, sparse points, and errors.
        """
        if workspace is None:
            tmp = tempfile.mkdtemp(prefix="colmap_")
            workspace = Path(tmp)

        workspace.mkdir(parents=True, exist_ok=True)
        db_path = workspace / "database.db"

        logger.info(f"Running COLMAP on {image_dir} (workspace: {workspace})")

        # Step 1: Extract SIFT features
        logger.info("Extracting features...")
        sift_opts = pycolmap.SiftExtractionOptions()
        if self.quality == "low":
            sift_opts.max_num_features = 4096
        elif self.quality == "high":
            sift_opts.max_num_features = 16384
        else:
            sift_opts.max_num_features = 8192

        # Use shared camera model for all images (same phone/camera)
        reader_opts = pycolmap.ImageReaderOptions()
        reader_opts.camera_model = "SIMPLE_RADIAL"

        pycolmap.extract_features(
            database_path=str(db_path),
            image_path=str(image_dir),
            camera_mode=pycolmap.CameraMode.SINGLE,
            sift_options=sift_opts,
            reader_options=reader_opts,
        )

        # Step 2: Match features exhaustively (fine for 3-4 images)
        logger.info("Matching features...")
        match_opts = pycolmap.SiftMatchingOptions()
        match_opts.max_ratio = 0.9  # more permissive ratio test (default 0.8)
        match_opts.max_distance = 0.9

        pycolmap.match_exhaustive(
            database_path=str(db_path),
            sift_options=match_opts,
        )

        # Step 3: Incremental SfM
        logger.info("Running incremental mapping...")
        output_path = workspace / "sparse"
        output_path.mkdir(exist_ok=True)

        mapper_opts = pycolmap.IncrementalPipelineOptions()
        mapper_opts.min_num_matches = 5
        mapper_opts.mapper.init_min_num_inliers = 5
        mapper_opts.mapper.init_min_tri_angle = 0.5  # very low min angle for narrow baselines
        mapper_opts.mapper.abs_pose_min_num_inliers = 5
        mapper_opts.mapper.abs_pose_min_inlier_ratio = 0.05
        mapper_opts.mapper.init_max_forward_motion = 1.0  # allow any forward motion
        mapper_opts.mapper.init_max_reg_trials = 3
        mapper_opts.min_model_size = 2  # accept even 2-image reconstructions
        mapper_opts.multiple_models = True  # try multiple models to find one that works

        reconstructions = pycolmap.incremental_mapping(
            database_path=str(db_path),
            image_path=str(image_dir),
            output_path=str(output_path),
            options=mapper_opts,
        )

        if not reconstructions:
            raise RuntimeError(
                "COLMAP failed to reconstruct — no valid reconstruction found. "
                "Ensure photos have sufficient overlap and visual features."
            )

        # Use the largest reconstruction (most registered images)
        best_recon = max(reconstructions.values(), key=lambda r: r.num_reg_images())
        logger.info(
            f"Best reconstruction: {best_recon.num_reg_images()} images, "
            f"{best_recon.num_points3D()} 3D points"
        )

        return self._extract_result(best_recon)

    @staticmethod
    def _find_best_initial_pair(db_path: Path) -> tuple[int, int] | None:
        """Find the image pair with most inlier matches."""
        db = pycolmap.Database(str(db_path))
        pair_ids, inlier_counts = db.read_two_view_geometry_num_inliers()

        if not pair_ids:
            return None

        best_idx = max(range(len(pair_ids)), key=lambda i: inlier_counts[i])
        best_pair_id = pair_ids[best_idx]
        id1, id2 = db.pair_id_to_image_pair(best_pair_id)

        logger.info(
            f"Best pair: images {id1}-{id2} with {inlier_counts[best_idx]} inliers"
        )
        return (id1, id2)

    def _extract_result(self, recon) -> ReconstructionResult:
        """Extract camera poses and points from a pycolmap Reconstruction."""
        camera_poses = []
        reprojection_errors = []

        for image_id, image in recon.images.items():
            cam = recon.cameras[image.camera_id]

            # Camera center in world coordinates
            # COLMAP stores world-to-camera transform, so center = -R^T * t
            cfw = image.cam_from_world()
            R = cfw.rotation.matrix()
            t = cfw.translation
            position = -R.T @ t

            # Get focal length
            focal = cam.focal_length if hasattr(cam, 'focal_length') else cam.params[0]

            camera_poses.append(CameraPoseResult(
                image_name=image.name,
                position=np.array(position),
                rotation=np.array(R),
                focal_length=float(focal),
                image_width=cam.width,
                image_height=cam.height,
            ))

            # Compute mean reprojection error for this image
            errors = []
            for p2d in image.points2D:
                if p2d.has_point3D():
                    p3d = recon.points3D[p2d.point3D_id]
                    errors.append(p3d.error)
            if errors:
                reprojection_errors.append(float(np.mean(errors)))
            else:
                reprojection_errors.append(0.0)

        # Extract 3D points
        points = np.array([p.xyz for p in recon.points3D.values()])

        return ReconstructionResult(
            camera_poses=camera_poses,
            points3d=points,
            reprojection_errors=reprojection_errors,
        )
