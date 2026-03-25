from __future__ import annotations

from dataclasses import dataclass

from staging.models import CameraPose, FloorPlan

MAX_REPROJECTION_ERROR_PX = 5.0
MIN_FLOOR_AREA_SQ_M = 2.0


class ReconstructionQualityError(Exception):
    pass


@dataclass
class QualityResult:
    passed: bool
    mean_reprojection_error: float
    floor_area: float


def compute_reprojection_error(errors: list[float]) -> float:
    return sum(errors) / len(errors) if errors else 0.0


def validate_reconstruction(
    poses: list[CameraPose],
    floor: FloorPlan,
    reprojection_errors: list[float],
) -> QualityResult:
    mean_err = compute_reprojection_error(reprojection_errors)
    floor_area = floor.area()

    if mean_err > MAX_REPROJECTION_ERROR_PX:
        raise ReconstructionQualityError(
            f"Mean reprojection error {mean_err:.1f}px exceeds threshold "
            f"{MAX_REPROJECTION_ERROR_PX}px. Please re-shoot with more overlap."
        )

    if floor_area < MIN_FLOOR_AREA_SQ_M:
        raise ReconstructionQualityError(
            f"Detected floor area {floor_area:.1f}m² is implausibly small. "
            "Reconstruction may have failed. Please re-shoot."
        )

    return QualityResult(
        passed=True,
        mean_reprojection_error=mean_err,
        floor_area=floor_area,
    )
