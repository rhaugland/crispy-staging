from __future__ import annotations

import asyncio
import io
import logging
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from api.schemas import CreateJobResponse, JobResultsResponse, JobStatusResponse
from staging.validation import ValidationError, validate_photos

router = APIRouter(prefix="/api/v1")

# In-memory job store (swap for Redis in production)
_jobs: dict[str, dict] = {}
UPLOAD_DIR = Path(__file__).parent.parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)


@router.post("/jobs", status_code=201, response_model=CreateJobResponse)
async def create_job(
    photos: list[UploadFile] = File(...),
    room_type: str = Form("living"),
    style: str = Form("modern"),
    room_status: str = Form("empty"),
):
    photo_bytes = [await p.read() for p in photos]
    try:
        validate_photos(photo_bytes)
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))

    job_id = str(uuid.uuid4())[:8]

    # Save uploaded photos
    job_dir = UPLOAD_DIR / job_id
    job_dir.mkdir(exist_ok=True)
    photo_keys = []
    for i, (p, raw) in enumerate(zip(photos, photo_bytes)):
        ext = Path(p.filename or f"photo_{i}.jpg").suffix or ".jpg"
        fname = f"photo_{i}{ext}"
        (job_dir / fname).write_bytes(raw)
        photo_keys.append(fname)

    _jobs[job_id] = {
        "id": job_id,
        "status": "pending",
        "room_type": room_type,
        "style": style,
        "room_status": room_status,
        "photo_keys": photo_keys,
        "error": None,
        "result_urls": [],
        "photos_total": len(photo_keys),
        "photos_done": 0,
    }

    # Kick off async reconstruction in background
    asyncio.create_task(_process_job(job_id))

    return CreateJobResponse(job_id=job_id)


@router.get("/jobs/{job_id}")
async def job_status(job_id: str):
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "id": job["id"],
        "status": job["status"],
        "error": job["error"],
        "photos_total": job.get("photos_total", 0),
        "photos_done": job.get("photos_done", 0),
    }


@router.get("/jobs/{job_id}/results")
async def job_results(job_id: str):
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Job status: {job['status']}")
    originals = [f"/api/v1/uploads/{job_id}/{k}" for k in job["photo_keys"]]
    return {"urls": job["result_urls"], "originals": originals}


@router.get("/uploads/{job_id}/{filename}")
async def serve_upload(job_id: str, filename: str):
    path = UPLOAD_DIR / job_id / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(str(path))


async def _process_job(job_id: str):
    """Background task: run staging on uploaded photos."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

    # Run blocking Replicate calls in a thread so the event loop stays free
    # for status polling requests
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _process_job_sync, job_id)


def _process_job_sync(job_id: str):
    """Synchronous staging work — runs in a thread."""
    import random
    import time
    from staging.replicate_staging import stage_photo

    job = _jobs[job_id]
    job_dir = UPLOAD_DIR / job_id

    try:
        job["status"] = "staging"

        job_seed = random.randint(0, 2**31 - 1)
        staged_urls = []
        reference_path = None
        for i, fname in enumerate(job["photo_keys"]):
            if i > 0:
                time.sleep(12)  # space out requests for rate limits
            photo_path = job_dir / fname
            staged_path = job_dir / f"staged_{i}.jpg"
            stage_photo(
                image_path=photo_path,
                room_type=job["room_type"],
                style=job["style"],
                seed=job_seed,
                output_path=staged_path,
                reference_image=reference_path,
                room_status=job.get("room_status", "empty"),
            )
            # First staged result becomes the style reference for all subsequent photos
            if i == 0:
                reference_path = staged_path
            staged_urls.append(f"/api/v1/uploads/{job_id}/staged_{i}.jpg")
            job["photos_done"] = i + 1

        job["status"] = "completed"
        job["result_urls"] = staged_urls

    except Exception as e:
        import traceback
        logger.error(f"Job {job_id} failed: {traceback.format_exc()}")
        job["status"] = "failed"
        job["error"] = str(e)
