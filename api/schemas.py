from pydantic import BaseModel


class CreateJobResponse(BaseModel):
    job_id: str
    status: str = "pending"
    message: str = "Job queued for processing"


class JobStatusResponse(BaseModel):
    id: str
    status: str
    error: str | None = None


class JobResultsResponse(BaseModel):
    urls: list[str]
