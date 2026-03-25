# tests/test_pipeline.py
import numpy as np
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from staging.pipeline import StagingPipeline
from staging.models import Job, JobStatus, RoomType, Style


@pytest.fixture
def pipeline():
    return StagingPipeline(
        storage=MagicMock(),
        luma_client=AsyncMock(),
        room_classifier=AsyncMock(),
        quality_gate=AsyncMock(),
    )


@pytest.mark.asyncio
async def test_pipeline_updates_status_through_stages(pipeline):
    job = Job(agent_id="a1", org_id="o1", photo_keys=["a.jpg", "b.jpg", "c.jpg"], style=Style.MODERN)
    statuses = []

    original_update = pipeline._update_status

    async def track_status(j, status):
        statuses.append(status)
        await original_update(j, status)

    pipeline._update_status = track_status

    with patch.object(pipeline, "_run_reconstruction", new_callable=AsyncMock), \
         patch.object(pipeline, "_run_analysis", new_callable=AsyncMock), \
         patch.object(pipeline, "_run_layout", new_callable=AsyncMock), \
         patch.object(pipeline, "_run_render", new_callable=AsyncMock), \
         patch.object(pipeline, "_run_harmonize", new_callable=AsyncMock):
        await pipeline.run(job)

    assert JobStatus.RECONSTRUCTING in statuses
    assert JobStatus.COMPLETED in statuses


@pytest.mark.asyncio
async def test_pipeline_handles_reconstruction_failure(pipeline):
    job = Job(agent_id="a1", org_id="o1", photo_keys=["a.jpg", "b.jpg", "c.jpg"])

    with patch.object(pipeline, "_run_reconstruction", new_callable=AsyncMock, side_effect=RuntimeError("bad photos")):
        await pipeline.run(job)

    assert job.status == JobStatus.FAILED
    assert "bad photos" in job.error
