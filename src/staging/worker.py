from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Modal import is conditional — only available in Modal runtime
try:
    import modal

    app = modal.App("virtual-staging")
    gpu_image = (
        modal.Image.debian_slim(python_version="3.11")
        .pip_install(
            "numpy", "Pillow", "httpx", "pydantic", "pydantic-settings",
            "boto3", "torch", "torchvision",
        )
        .apt_install("blender")
    )
except ImportError:
    modal = None
    app = None
    gpu_image = None


class StagingWorker:
    """GPU worker that runs the full staging pipeline for a single job."""

    async def run_job(self, job_id: str):
        """Main entry point. Loads job from store, runs pipeline, saves results."""
        from staging.pipeline import StagingPipeline

        logger.info(f"Starting job {job_id}")

        # 1. Load job from Redis/DB
        # 2. Download input photos from S3
        # 3. Run pipeline
        # 4. Upload results to S3
        # 5. Update job status

        raise NotImplementedError("Wire up full job execution")


# Modal function definition (only if Modal is available)
if modal is not None:
    @app.function(gpu="A10G", image=gpu_image, timeout=900)
    async def process_staging_job(job_id: str):
        worker = StagingWorker()
        await worker.run_job(job_id)
