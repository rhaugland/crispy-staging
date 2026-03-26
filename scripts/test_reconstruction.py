"""
End-to-end reconstruction test using COLMAP.

Usage:
    python3.11 scripts/test_reconstruction.py test-photos/

Reads all .jpg/.jpeg/.png files from the given directory,
runs COLMAP SfM to recover camera poses and sparse 3D points.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Add src to path so staging package is importable
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from staging.reconstruction.client import ColmapReconstructor


def main(photo_dir: str):
    photo_path = Path(photo_dir)
    if not photo_path.is_dir():
        print(f"Error: {photo_dir} is not a directory")
        sys.exit(1)

    # Collect photos
    extensions = {".jpg", ".jpeg", ".png"}
    photos = sorted(
        p for p in photo_path.iterdir()
        if p.suffix.lower() in extensions
    )

    if len(photos) < 3:
        print(f"Error: Need at least 3 photos, found {len(photos)}")
        sys.exit(1)

    if len(photos) > 4:
        print(f"Warning: Found {len(photos)} photos, using first 4")
        photos = photos[:4]

    print(f"Found {len(photos)} photos:")
    for p in photos:
        size_kb = p.stat().st_size / 1024
        print(f"  {p.name} ({size_kb:.0f} KB)")

    # Run reconstruction
    print("\n--- Running COLMAP SfM ---")
    reconstructor = ColmapReconstructor(quality="medium")

    try:
        result = reconstructor.reconstruct(image_dir=photo_path)
    except RuntimeError as e:
        print(f"\nReconstruction failed: {e}")
        sys.exit(1)

    # Report results
    print(f"\n--- Results ---")
    print(f"Registered cameras: {len(result.camera_poses)}")
    print(f"3D points recovered: {len(result.points3d)}")
    print(f"Mean reprojection errors: {[f'{e:.2f}px' for e in result.reprojection_errors]}")

    print("\n--- Camera Poses ---")
    for pose in result.camera_poses:
        print(f"  {pose.image_name}:")
        print(f"    Position: [{pose.position[0]:.3f}, {pose.position[1]:.3f}, {pose.position[2]:.3f}]")
        print(f"    Focal length: {pose.focal_length:.1f}px")
        print(f"    Image size: {pose.image_width}x{pose.image_height}")

    # Point cloud stats
    if len(result.points3d) > 0:
        pts = result.points3d
        print(f"\n--- Point Cloud Stats ---")
        print(f"  Min: [{pts[:, 0].min():.2f}, {pts[:, 1].min():.2f}, {pts[:, 2].min():.2f}]")
        print(f"  Max: [{pts[:, 0].max():.2f}, {pts[:, 1].max():.2f}, {pts[:, 2].max():.2f}]")
        print(f"  Centroid: [{pts.mean(0)[0]:.2f}, {pts.mean(0)[1]:.2f}, {pts.mean(0)[2]:.2f}]")

    print("\n--- Done ---")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3.11 scripts/test_reconstruction.py <photo-directory>")
        sys.exit(1)

    main(sys.argv[1])
