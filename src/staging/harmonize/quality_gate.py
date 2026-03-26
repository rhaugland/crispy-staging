from __future__ import annotations

import base64
import io
import json
import logging
from dataclasses import dataclass, field

import httpx
import numpy as np
from PIL import Image

from staging.config import settings

logger = logging.getLogger(__name__)

QUALITY_CHECK_PROMPT = """\
You are a virtual staging quality inspector. Compare each pair of original and staged room photos.

Check for these issues:
1. STRUCTURAL CHANGES: Walls, floors, ceiling, windows, or doors appear modified
2. FLOATING/CLIPPING: Furniture appears to float above the floor or clip through walls
3. LIGHTING MISMATCH: Staged furniture has different lighting direction or color temperature than the room
4. PROHIBITED ITEMS: Curtains, hanging lights, or structural modifications were added
5. REALISM: Furniture looks unnatural, has artifacts, or has incorrect perspective

Respond with JSON only, no markdown:
{"passed": true/false, "issues": ["issue description with image index"], "failed_indices": [0-based indices of failed images]}

If everything looks good, respond: {"passed": true, "issues": [], "failed_indices": []}
"""


@dataclass
class QualityCheckResult:
    passed: bool
    issues: list[str] = field(default_factory=list)
    failed_indices: list[int] = field(default_factory=list)


class QualityGate:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.max_retries = settings.quality_gate_max_retries

    async def check(
        self, staged_images: list[np.ndarray], original_images: list[np.ndarray]
    ) -> QualityCheckResult:
        return await self._check_with_vision(staged_images, original_images)

    async def _check_with_vision(
        self, staged: list[np.ndarray], originals: list[np.ndarray]
    ) -> QualityCheckResult:
        image_contents = []
        for i, (orig, stg) in enumerate(zip(originals, staged)):
            image_contents.append({"type": "text", "text": f"--- Image {i} ---"})
            image_contents.append({"type": "text", "text": "Original:"})
            image_contents.append(self._encode_image(orig))
            image_contents.append({"type": "text", "text": "Staged:"})
            image_contents.append(self._encode_image(stg))

        image_contents.append({"type": "text", "text": QUALITY_CHECK_PROMPT})

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
                    "max_tokens": 512,
                    "messages": [
                        {"role": "user", "content": image_contents},
                    ],
                },
                timeout=60.0,
            )
            resp.raise_for_status()
            text = resp.json()["content"][0]["text"]

        return self._parse_response(text)

    @staticmethod
    def _encode_image(img_array: np.ndarray) -> dict:
        pil_img = Image.fromarray(img_array)
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=75)
        b64 = base64.b64encode(buf.getvalue()).decode()
        return {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/jpeg", "data": b64},
        }

    @staticmethod
    def _parse_response(text: str) -> QualityCheckResult:
        try:
            data = json.loads(text)
            return QualityCheckResult(
                passed=bool(data.get("passed", False)),
                issues=data.get("issues", []),
                failed_indices=data.get("failed_indices", []),
            )
        except (json.JSONDecodeError, KeyError):
            logger.warning(f"Could not parse quality check response: {text[:200]}")
            return QualityCheckResult(
                passed=False,
                issues=[f"Unparseable quality check response: {text[:200]}"],
                failed_indices=[],
            )

    def _adjust_params(self, attempt: int) -> dict:
        if attempt == 0:
            return {"color_temp_shift": 0, "exposure_shift": 0.0, "mask_expand_px": 0}
        if attempt == 1:
            return {"color_temp_shift": 500, "exposure_shift": 0.5, "mask_expand_px": 10}
        return {"color_temp_shift": -500, "exposure_shift": -0.5, "mask_expand_px": 10}

    def should_retry(self, attempt: int) -> bool:
        return attempt < self.max_retries
