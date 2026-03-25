from __future__ import annotations

import asyncio
from dataclasses import dataclass

import httpx


@dataclass
class ReconstructionResult:
    mesh_url: str
    cameras: list[dict]


class LumaClient:
    BASE_URL = "https://webapp.engineeringlumalabs.com/api/v3"

    def __init__(self, api_key: str):
        self.api_key = api_key
        self._client = httpx.AsyncClient(
            headers={"Authorization": f"luma-api-key={api_key}"},
            timeout=60.0,
        )

    @staticmethod
    def _raise_for_status(resp: httpx.Response) -> None:
        if resp.status_code >= 400:
            resp.raise_for_status()

    async def submit(self, photo_urls: list[str]) -> str:
        resp = await self._client.post(
            f"{self.BASE_URL}/captures",
            json={"images": [{"url": u} for u in photo_urls], "type": "room"},
        )
        self._raise_for_status(resp)
        return resp.json()["id"]

    async def poll(
        self, capture_id: str, interval: float = 5.0, timeout: float = 600.0
    ) -> ReconstructionResult:
        elapsed = 0.0
        while elapsed < timeout:
            resp = await self._client.get(f"{self.BASE_URL}/captures/{capture_id}")
            self._raise_for_status(resp)
            data = resp.json()

            if data["status"] == "complete":
                return ReconstructionResult(
                    mesh_url=data["mesh_url"],
                    cameras=data.get("cameras", []),
                )
            if data["status"] == "failed":
                raise RuntimeError(f"Reconstruction failed: {data.get('error', 'unknown')}")

            await asyncio.sleep(interval)
            elapsed += interval

        raise TimeoutError(f"Reconstruction timed out after {timeout}s")

    async def download_mesh(self, mesh_url: str) -> bytes:
        resp = await self._client.get(mesh_url)
        self._raise_for_status(resp)
        return resp.content

    async def close(self):
        await self._client.aclose()
