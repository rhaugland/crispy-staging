import io

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
from PIL import Image

from api.app import create_app


@pytest.fixture
def client():
    app = create_app()
    return TestClient(app)


def _make_jpeg(w=1920, h=1080):
    img = Image.new("RGB", (w, h), "white")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf


def test_create_job(client):
    files = [("photos", ("p1.jpg", _make_jpeg(), "image/jpeg")) for _ in range(3)]
    with patch("api.routes.enqueue_job", new_callable=AsyncMock, return_value="job-123"):
        resp = client.post(
            "/api/v1/jobs",
            files=files,
            data={"room_type": "living", "style": "modern"},
            headers={"Authorization": "Bearer test-token"},
        )
    assert resp.status_code == 201
    assert "job_id" in resp.json()


def test_create_job_too_few_photos(client):
    files = [("photos", ("p1.jpg", _make_jpeg(), "image/jpeg")) for _ in range(2)]
    resp = client.post(
        "/api/v1/jobs",
        files=files,
        data={"room_type": "living", "style": "modern"},
        headers={"Authorization": "Bearer test-token"},
    )
    assert resp.status_code == 422


def test_get_job_status(client):
    with patch("api.routes.get_job", return_value={"id": "job-123", "status": "rendering"}):
        resp = client.get(
            "/api/v1/jobs/job-123",
            headers={"Authorization": "Bearer test-token"},
        )
    assert resp.status_code == 200
    assert resp.json()["status"] == "rendering"


def test_get_job_results(client):
    with patch(
        "api.routes.get_job_results",
        return_value={"urls": ["https://s3/staged_0.jpg", "https://s3/staged_1.jpg"]},
    ):
        resp = client.get(
            "/api/v1/jobs/job-123/results",
            headers={"Authorization": "Bearer test-token"},
        )
    assert resp.status_code == 200
    assert len(resp.json()["urls"]) == 2
