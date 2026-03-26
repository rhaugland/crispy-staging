"""Virtual staging via Replicate — uses ControlNet interior design model."""
from __future__ import annotations

import logging
import os
import random
import time
import urllib.request
from pathlib import Path

import replicate

logger = logging.getLogger(__name__)

STAGING_PROMPT = (
    "A professionally staged {style} {room_type} with elegant furniture. "
    "Sofas, chairs, coffee table, rugs, floor lamps, throw pillows, potted plants, wall art. "
    "Keep the exact same room architecture. Photorealistic, interior design magazine, 8k."
)

STYLE_MAP = {
    "modern": "modern",
    "traditional": "traditional",
    "scandinavian": "scandinavian",
    "luxury": "luxury",
    "coastal": "coastal",
    "midcenturymodern": "mid-century modern",
    "farmhouse": "farmhouse",
    "industrial": "industrial",
    "minimalist": "minimalist",
    "contemporary": "contemporary",
    "boho": "bohemian",
    "rustic": "rustic",
}

ROOM_TYPE_MAP = {
    "living": "living room",
    "bedroom": "bedroom",
    "dining": "dining room",
    "kitchen": "kitchen",
    "office": "home office",
    "bathroom": "bathroom",
}


def build_prompt(room_type: str, style: str) -> str:
    """Build a staging prompt from room type and style."""
    return STAGING_PROMPT.format(
        style=STYLE_MAP.get(style, style),
        room_type=ROOM_TYPE_MAP.get(room_type, room_type),
    )


def stage_photo(
    image_path: Path,
    room_type: str = "living",
    style: str = "modern",
    seed: int | None = None,
    output_path: Path | None = None,
    reference_image: Path | None = None,
    **kwargs,
) -> Path:
    """Stage a room photo using adirik/interior-design ControlNet model.

    Uses dual ControlNets (segmentation + MLSD lines) to lock room structure
    while generating furniture via inpainting.
    """
    token = os.environ.get("REPLICATE_API_TOKEN")
    if not token:
        raise ValueError("REPLICATE_API_TOKEN environment variable not set")

    if output_path is None:
        output_path = image_path.parent / f"staged_{image_path.stem}.jpg"

    prompt = build_prompt(room_type, style)
    if seed is None:
        seed = random.randint(0, 2**32 - 1)

    logger.info(f"Staging {image_path.name} as {room_type}/{style} (seed={seed})")

    output = replicate.run(
        "adirik/interior-design:76604baddc85b1b4616e1c6475eca080da339c8875bd4996705440484a6eac38",
        input={
            "image": open(image_path, "rb"),
            "prompt": prompt,
            "negative_prompt": (
                "lowres, watermark, banner, logo, contactinfo, text, deformed, blurry, "
                "blur, out of focus, out of frame, surreal, ugly, beginner, amateur, "
                "empty room, no furniture, changing walls, changing floor, "
                "different room layout, structural changes"
            ),
            "prompt_strength": 0.3,
            "guidance_scale": 15,
            "num_inference_steps": 50,
            "seed": seed,
        },
    )

    # Download result
    result_url = None
    if isinstance(output, list) and output:
        result_url = str(output[0])
    elif hasattr(output, "url"):
        result_url = output.url
    elif hasattr(output, "__iter__"):
        for item in output:
            if hasattr(item, "url"):
                result_url = item.url
            else:
                result_url = str(item)
            break
    else:
        result_url = str(output)

    if not result_url:
        raise RuntimeError("Interior design model returned no output")

    logger.info(f"Downloading result from {result_url}")
    urllib.request.urlretrieve(result_url, str(output_path))
    logger.info(f"Saved staged image to {output_path}")
    return output_path
