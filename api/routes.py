from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from api.auth import verify_token
from api.schemas import CreateJobResponse, JobResultsResponse, JobStatusResponse
from staging.validation import ValidationError, validate_photos

router = APIRouter(prefix="/api/v1")


async def enqueue_job(
    photo_keys: list[str],
    room_type: str,
    style: str,
    agent_id: str,
    org_id: str,
) -> str:
    """Enqueue a staging job. Placeholder — wire up Redis queue."""
    raise NotImplementedError("Wire up job queue")


async def get_job(job_id: str) -> dict:
    """Get job status. Placeholder."""
    raise NotImplementedError("Wire up job store")


async def get_job_results(job_id: str) -> dict:
    """Get job results. Placeholder."""
    raise NotImplementedError("Wire up results")


@router.post("/jobs", status_code=201, response_model=CreateJobResponse)
async def create_job(
    photos: list[UploadFile] = File(...),
    room_type: str = Form("living"),
    style: str = Form("modern"),
    auth: dict = Depends(verify_token),
):
    photo_bytes = [await p.read() for p in photos]
    try:
        validate_photos(photo_bytes)
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))

    job_id = await enqueue_job(
        photo_keys=[],  # filled after S3 upload
        room_type=room_type,
        style=style,
        agent_id=auth["agent_id"],
        org_id=auth["org_id"],
    )
    return CreateJobResponse(job_id=job_id)


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def job_status(job_id: str, auth: dict = Depends(verify_token)):
    job = await get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobStatusResponse(**job)


@router.get("/jobs/{job_id}/results", response_model=JobResultsResponse)
async def job_results(job_id: str, auth: dict = Depends(verify_token)):
    results = await get_job_results(job_id)
    if results is None:
        raise HTTPException(status_code=404, detail="Results not found")
    return JobResultsResponse(**results)
