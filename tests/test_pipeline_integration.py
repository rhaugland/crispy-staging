import numpy as np
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from staging.pipeline import StagingPipeline
from staging.models import (
    Job, JobStatus, RoomType, Style, CameraPose, FloorPlan, FurniturePlacement,
)


@pytest.mark.asyncio
async def test_full_pipeline_mock_integration():
    """Test that pipeline correctly chains all stages together."""
    storage = MagicMock()
    storage.download = MagicMock(return_value=b"fake-image")

    mock_poses = [
        CameraPose(
            position=np.array([0, 1.5, 0]),
            rotation=np.eye(3),
            focal_length=35.0,
            image_width=1920,
            image_height=1080,
        )
        for _ in range(3)
    ]
    mock_floor = FloorPlan(
        polygon=np.array([[0, 0], [5, 0], [5, 4], [0, 4]], dtype=float),
        plane_normal=np.array([0, 1, 0]),
        plane_offset=0.0,
    )

    pipeline = StagingPipeline(
        storage=storage,
        reconstructor=MagicMock(),
        room_classifier=AsyncMock(),
        quality_gate=AsyncMock(),
    )

    job = Job(
        agent_id="a1",
        org_id="o1",
        photo_keys=["a.jpg", "b.jpg", "c.jpg"],
        room_type=RoomType.LIVING,
        style=Style.MODERN,
    )

    with (
        patch.object(
            pipeline, "_run_reconstruction", new_callable=AsyncMock,
            return_value={"poses": mock_poses, "floor": mock_floor, "mesh_path": "/tmp/mesh.glb"},
        ),
        patch.object(
            pipeline, "_run_analysis", new_callable=AsyncMock,
            return_value={"floor": mock_floor, "room_type": RoomType.LIVING, "exclusion_zones": []},
        ),
        patch.object(
            pipeline, "_run_layout", new_callable=AsyncMock,
            return_value={
                "placements": [
                    FurniturePlacement(
                        asset_id="sofa-1", position=np.array([2.5, 0, 2]), rotation_y=0,
                    )
                ]
            },
        ),
        patch.object(
            pipeline, "_run_render", new_callable=AsyncMock,
            return_value={"render_paths": ["/tmp/r0.png", "/tmp/r1.png", "/tmp/r2.png"]},
        ),
        patch.object(
            pipeline, "_run_harmonize", new_callable=AsyncMock,
            return_value=["results/staged_0.jpg", "results/staged_1.jpg", "results/staged_2.jpg"],
        ),
    ):
        result = await pipeline.run(job)

    assert result.status == JobStatus.COMPLETED
    assert len(result.result_keys) == 3
