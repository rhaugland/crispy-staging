# src/staging/pipeline.py
from __future__ import annotations
import logging
from typing import Any
from staging.models import Job, JobStatus

logger = logging.getLogger(__name__)


class StagingPipeline:
    def __init__(self, storage, luma_client, room_classifier, quality_gate):
        self.storage = storage
        self.luma_client = luma_client
        self.room_classifier = room_classifier
        self.quality_gate = quality_gate

    async def run(self, job: Job) -> Job:
        try:
            await self._update_status(job, JobStatus.RECONSTRUCTING)
            reconstruction = await self._run_reconstruction(job)

            await self._update_status(job, JobStatus.ANALYZING)
            analysis = await self._run_analysis(job, reconstruction)

            await self._update_status(job, JobStatus.LAYING_OUT)
            layout = await self._run_layout(job, analysis)

            await self._update_status(job, JobStatus.RENDERING)
            renders = await self._run_render(job, reconstruction, layout)

            await self._update_status(job, JobStatus.HARMONIZING)
            results = await self._run_harmonize(job, renders)

            job.result_keys = results
            await self._update_status(job, JobStatus.COMPLETED)

        except Exception as e:
            logger.exception(f"Pipeline failed for job {job.id}")
            job.error = str(e)
            await self._update_status(job, JobStatus.FAILED)

        return job

    async def _update_status(self, job: Job, status: JobStatus):
        job.status = status
        logger.info(f"Job {job.id}: {status}")

    async def _run_reconstruction(self, job: Job) -> dict[str, Any]:
        raise NotImplementedError("Wire up reconstruction stage")

    async def _run_analysis(self, job: Job, reconstruction: dict) -> dict[str, Any]:
        raise NotImplementedError("Wire up analysis stage")

    async def _run_layout(self, job: Job, analysis: dict) -> dict[str, Any]:
        raise NotImplementedError("Wire up layout stage")

    async def _run_render(self, job: Job, reconstruction: dict, layout: dict) -> dict[str, Any]:
        raise NotImplementedError("Wire up render stage")

    async def _run_harmonize(self, job: Job, renders: dict) -> list[str]:
        raise NotImplementedError("Wire up harmonize stage")
