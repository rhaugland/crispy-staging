from __future__ import annotations

import base64
import io

import httpx
import numpy as np
from PIL import Image

from staging.models import RoomType

CLASSIFICATION_PROMPT = (
    "What type of room is shown in these photos? "
    "Respond with exactly one word: living, bedroom, dining, or office. "
    "If uncertain, respond with the closest match."
)

VALID_TYPES = {t.value for t in RoomType}


class RoomClassifier:
    def __init__(self, api_key: str):
        self.api_key = api_key

    async def classify(self, images: list[np.ndarray]) -> RoomType:
        raw_result = await self._call_vision_model(images)
        normalized = raw_result.strip().lower()

        if normalized in VALID_TYPES:
            return RoomType(normalized)
        return RoomType.LIVING

    async def _call_vision_model(self, images: list[np.ndarray]) -> str:
        image_contents = []
        for img_array in images[:2]:
            pil_img = Image.fromarray(img_array)
            buf = io.BytesIO()
            pil_img.save(buf, format="JPEG", quality=80)
            b64 = base64.b64encode(buf.getvalue()).decode()
            image_contents.append({
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg", "data": b64},
            })

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": "claude-sonnet-4-20250514",
                    "max_tokens": 10,
                    "messages": [{"role": "user", "content": [*image_contents, {"type": "text", "text": CLASSIFICATION_PROMPT}]}],
                },
                timeout=30.0,
            )
            resp.raise_for_status()
            return resp.json()["content"][0]["text"]
