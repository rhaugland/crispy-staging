import pytest
import httpx
from unittest.mock import AsyncMock, patch
from staging.reconstruction.client import LumaClient, ReconstructionResult


@pytest.fixture
def client():
    return LumaClient(api_key="test-key")


@pytest.mark.asyncio
async def test_submit_returns_capture_id(client):
    mock_response = httpx.Response(200, json={"id": "capture-abc"})
    with patch.object(client._client, "post", new_callable=AsyncMock, return_value=mock_response):
        capture_id = await client.submit(photo_urls=["https://s3/a.jpg", "https://s3/b.jpg", "https://s3/c.jpg"])
    assert capture_id == "capture-abc"


@pytest.mark.asyncio
async def test_poll_until_complete(client):
    responses = [
        httpx.Response(200, json={"status": "processing"}),
        httpx.Response(200, json={"status": "processing"}),
        httpx.Response(200, json={"status": "complete", "mesh_url": "https://mesh", "cameras": []}),
    ]
    call_count = 0

    async def mock_get(*args, **kwargs):
        nonlocal call_count
        resp = responses[call_count]
        call_count += 1
        return resp

    with patch.object(client._client, "get", side_effect=mock_get):
        result = await client.poll(capture_id="capture-abc", interval=0.01)

    assert result.mesh_url == "https://mesh"
    assert call_count == 3


@pytest.mark.asyncio
async def test_poll_failure_raises(client):
    mock_response = httpx.Response(200, json={"status": "failed", "error": "bad input"})
    with patch.object(client._client, "get", new_callable=AsyncMock, return_value=mock_response):
        with pytest.raises(RuntimeError, match="bad input"):
            await client.poll(capture_id="capture-abc", interval=0.01)
