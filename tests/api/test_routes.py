import io

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
from PIL import Image

from api.app import create_app
from api.routes import _jobs


@pytest.fixture
def client():
    app = create_app()
    _jobs.clear()
    return TestClient(app)


def _make_jpeg(w=1920, h=1080):
    img = Image.new("RGB", (w, h), "white")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf


def test_create_job(client):
    files = [("photos", ("p1.jpg", _make_jpeg(), "image/jpeg")) for _ in range(3)]
    with patch("api.routes._process_job", new_callable=AsyncMock):
        resp = client.post(
            "/api/v1/jobs",
            files=files,
            data={"room_type": "living", "style": "modern"},
        )
    assert resp.status_code == 201
    assert "job_id" in resp.json()


def test_create_job_too_few_photos(client):
    files = [("photos", ("p1.jpg", _make_jpeg(), "image/jpeg")) for _ in range(2)]
    with patch("api.routes._process_job", new_callable=AsyncMock):
        resp = client.post(
            "/api/v1/jobs",
            files=files,
            data={"room_type": "living", "style": "modern"},
        )
    assert resp.status_code == 422


def test_get_job_status(client):
    _jobs["job-123"] = {"id": "job-123", "status": "rendering", "error": None}
    resp = client.get("/api/v1/jobs/job-123")
    assert resp.status_code == 200
    assert resp.json()["status"] == "rendering"


def test_get_job_results(client):
    _jobs["job-123"] = {
        "id": "job-123",
        "status": "completed",
        "error": None,
        "result_urls": ["/uploads/job-123/staged_0.jpg", "/uploads/job-123/staged_1.jpg"],
    }
    resp = client.get("/api/v1/jobs/job-123/results")
    assert resp.status_code == 200
    assert len(resp.json()["urls"]) == 2
