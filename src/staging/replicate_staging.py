"""Virtual staging via Gemini (empty rooms) and Replicate SDXL (furnished rooms)."""
from __future__ import annotations

import base64
import json
import logging
import os
import random
import urllib.request
from pathlib import Path

import replicate

logger = logging.getLogger(__name__)

ROOM_TYPE_MAP = {
    "living": "living room",
    "bedroom": "bedroom",
    "dining": "dining room",
    "kitchen": "kitchen",
    "office": "home office",
    "bathroom": "bathroom",
}

# Specific furniture + style combos for SDXL prompts
ROOM_FURNITURE = {
    "living": "a large sofa, accent chairs, a coffee table, side tables, a rug, table lamps, throw pillows, and wall art",
    "bedroom": "a bed with headboard and bedding, nightstands, table lamps, a dresser, an accent chair, a rug, and wall art",
    "dining": "a dining table, dining chairs, a sideboard or buffet, a centerpiece, table settings, a rug, and wall art",
    "kitchen": "bar stools, a kitchen island arrangement, pendant lights, decorative bowls, plants, and countertop accessories",
    "office": "a desk, an office chair, a bookshelf, a desk lamp, storage, a rug, and wall art",
    "bathroom": "towels on racks, bath mat, countertop accessories, a plant, candles, and decorative storage",
}

STYLE_PROMPTS = {
    "modern": (
        "Furniture style: sleek low-profile furniture with clean geometric lines. "
        "Black, white, and gray upholstery. Chrome and glass accent tables. "
        "Contemporary art prints on walls. Simple geometric rug."
    ),
    "traditional": (
        "Furniture style: classic dark wood furniture with carved details. "
        "Burgundy, navy, and forest green fabric upholstery. Tufted cushions. "
        "Oriental area rug. Brass table lamps. Framed oil paintings on walls."
    ),
    "scandinavian": (
        "Furniture style: light oak and birch wood furniture with simple rounded forms. "
        "White and pale gray upholstery with soft blush accents. "
        "Sheepskin throws, knit blankets, ceramic vases on tables."
    ),
    "luxury": (
        "Furniture style: plush velvet furniture in emerald, navy, or burgundy. "
        "Gold and marble accent tables. Crystal table lamps. Silk cushions. "
        "Large-scale framed artwork. Thick pile area rug."
    ),
    "coastal": (
        "Furniture style: white and light blue slipcovered sofas. "
        "Rattan and wicker accent chairs. Driftwood coffee table. "
        "Seagrass area rug. Coral and shell decor on tables. Nautical art prints."
    ),
    "midcenturymodern": (
        "Furniture style: furniture with organic curves and tapered wooden legs. "
        "Warm walnut and teak wood tones. Mustard yellow, olive green, and burnt orange fabrics. "
        "Sunburst mirror on wall. Abstract art prints."
    ),
    "farmhouse": (
        "Furniture style: distressed white-painted wood furniture. "
        "Natural linen slipcovered sofas. Wooden farm table. Ladder shelf. "
        "Woven baskets. Gingham throw pillows. Galvanized metal table accessories."
    ),
    "industrial": (
        "Furniture style: raw metal and reclaimed wood furniture. Metal pipe shelving. "
        "Dark leather Chesterfield sofa. Riveted metal coffee table. "
        "Vintage factory clock on wall. Dark moody colored textiles."
    ),
    "minimalist": (
        "Furniture style: very few carefully chosen pieces. "
        "Low platform furniture in white, black, or light wood. "
        "One single statement art piece on wall. Simple geometric area rug."
    ),
    "contemporary": (
        "Furniture style: sculptural furniture forms with mixed materials. "
        "Marble and brass accent tables. Bouclé fabric upholstery. "
        "Muted earth tones with one bold accent color. Oversized art on wall."
    ),
    "boho": (
        "Furniture style: layered eclectic mix of colorful furniture. "
        "Low floor seating with colorful cushions. Macramé wall hanging. "
        "Moroccan poufs. Kilim area rug. Rattan peacock chair. Trailing potted plants."
    ),
    "rustic": (
        "Furniture style: heavy natural timber furniture. "
        "Thick wool blankets. Leather and hide upholstery. Wrought iron hardware. "
        "Warm amber and brown textiles. Handcrafted pottery on tables."
    ),
}

MODELS = {
    "sdxl": "stability-ai/sdxl:7762fd07cf82c948538e41f63f77d685e02b063e37e496e96eefd46c929f9bdc",
    "style_transfer": "fofr/style-transfer:f1023890703bc0a5a3a2c21b5e498833be5f6ef6e70e9daf6b9b3a4fd8309cf0",
}


def build_prompt(room_type: str, style: str) -> str:
    room_name = ROOM_TYPE_MAP.get(room_type, room_type)
    furniture = ROOM_FURNITURE.get(room_type, "furniture, rugs, lamps, and decor")
    style_desc = STYLE_PROMPTS.get(style, f"{style} interior design style.")

    return (
        f"Same exact room, only replace furniture and wall art. Add {furniture}. "
        f"{style_desc} "
        f"Everything else stays identical — walls, floors, ceiling, windows, doors, countertops, cabinets, "
        f"appliances, light fixtures, vents, sprinklers, outlets, switches, trim, molding. "
        f"Only furniture and paintings change. Photorealistic real estate photography."
    )


def _extract_url(output) -> str:
    """Pull a URL string from various Replicate output formats."""
    if isinstance(output, str):
        return output
    if isinstance(output, dict):
        for key in ("url", "output", "image"):
            if key in output:
                return str(output[key])
        for v in output.values():
            if isinstance(v, str) and v.startswith("http"):
                return v
    if isinstance(output, list) and output:
        return str(output[0])
    if hasattr(output, "url"):
        return output.url
    if hasattr(output, "__iter__"):
        for item in output:
            return str(item)
    return str(output)


def _describe_furniture(image_path: Path) -> str:
    """Ask Gemini to describe the furniture in a staged room for cross-room consistency."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return ""

    image_data = image_path.read_bytes()
    image_b64 = base64.b64encode(image_data).decode("utf-8")
    ext = image_path.suffix.lower()
    mime = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png"

    payload = json.dumps({
        "contents": [{
            "parts": [
                {"text": (
                    "Describe ONLY the furniture and decor in this room in one paragraph. "
                    "Include specific details: wood tone (e.g. light oak, dark walnut), "
                    "fabric colors, material types (velvet, linen, leather), metal finishes "
                    "(brass, chrome, matte black), rug colors/patterns, and art style. "
                    "Be very specific about colors and materials so another room could be "
                    "furnished to match."
                )},
                {"inline_data": {"mime_type": mime, "data": image_b64}},
            ]
        }],
        "generationConfig": {
            "responseModalities": ["TEXT"],
        },
    }).encode("utf-8")

    req = urllib.request.Request(
        f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3-pro-image-preview:generateContent?key={api_key}",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read().decode("utf-8"))
        candidates = result.get("candidates", [])
        for candidate in candidates:
            parts = candidate.get("content", {}).get("parts", [])
            for part in parts:
                if "text" in part:
                    logger.info(f"Furniture description: {part['text'][:200]}...")
                    return part["text"]
    except Exception as e:
        logger.warning(f"Failed to describe furniture: {e}")
    return ""


def _stage_gemini(
    image_path: Path,
    room_type: str,
    style: str,
    output_path: Path,
    furniture_description: str = "",
) -> Path:
    """Stage an empty room using Gemini image generation."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set")

    room_name = ROOM_TYPE_MAP.get(room_type, room_type)
    furniture = ROOM_FURNITURE.get(room_type, "furniture, rugs, lamps, and decor")
    style_desc = STYLE_PROMPTS.get(style, f"{style} interior design style.")

    prompt = (
        f"Edit this photo of an empty {room_name}. Add furniture ONLY — do not remove, replace, "
        f"or alter anything that already exists in the photo. "
        f"ABSOLUTE RULES — NEVER BREAK THESE: "
        f"1. Every countertop in this photo must remain exactly as-is — same material, same color, same shape, fully visible. "
        f"2. Every cabinet must remain exactly as-is. "
        f"3. Every sink, appliance, and fixture must remain exactly as-is. "
        f"4. Camera angle and perspective must not change at all. "
        f"5. Walls, floors, ceiling, windows, doors, trim, molding, vents, outlets — all unchanged. "
        f"Add ONLY these items into the empty space: {furniture}. "
        f"{style_desc} "
        f"Place furniture on the existing floors. Do not put furniture where countertops or cabinets are. "
        f"Photorealistic real estate photograph."
    )

    logger.info(f"Gemini: staging {image_path.name} as {room_name}/{style}")

    # Single image only — no reference image to avoid confusing Gemini
    image_data = image_path.read_bytes()
    image_b64 = base64.b64encode(image_data).decode("utf-8")
    ext = image_path.suffix.lower()
    mime = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png"

    payload = json.dumps({
        "contents": [{
            "parts": [
                {"text": prompt},
                {"inline_data": {"mime_type": mime, "data": image_b64}},
            ]
        }],
        "generationConfig": {
            "responseModalities": ["IMAGE"],
        },
    }).encode("utf-8")

    req = urllib.request.Request(
        f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3-pro-image-preview:generateContent?key={api_key}",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            result = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8", errors="replace")
        logger.error(f"Gemini API error {e.code}: {error_body}")
        raise RuntimeError(f"Gemini API error {e.code}: {error_body}") from e

    # Extract generated image from Gemini response
    candidates = result.get("candidates", [])
    for candidate in candidates:
        parts = candidate.get("content", {}).get("parts", [])
        for part in parts:
            inline = part.get("inlineData") or part.get("inline_data")
            if inline and "data" in inline:
                output_path.write_bytes(base64.b64decode(inline["data"]))
                logger.info(f"Gemini: saved staged image to {output_path}")
                return output_path

    logger.error(f"Gemini response structure: {json.dumps(result, default=str)[:500]}")
    raise RuntimeError("Gemini returned no image in response")


def _stage_sdxl(
    image_path: Path,
    room_type: str,
    style: str,
    seed: int,
    output_path: Path,
) -> Path:
    """Stage a furnished room using SDXL img2img."""
    prompt = build_prompt(room_type, style)

    output = replicate.run(
        MODELS["sdxl"],
        input={
            "image": open(image_path, "rb"),
            "prompt": prompt,
            "negative_prompt": (
                "changing room structure, changing walls, changing paint color, "
                "changing floors, changing ceiling, "
                "changing windows, changing window frames, changing doors, "
                "changing countertops, changing cabinets, changing sinks, "
                "changing appliances, changing light fixtures, "
                "removing vents, removing sprinklers, removing outlets, "
                "moving walls, different room shape, different room layout, "
                "renovation, remodel, new flooring, new paint, overdecorated, "
                "blurry, low quality, distorted, deformed, watermark, "
                "cartoon, illustration, painting, drawing, anime, CGI, unrealistic, oversaturated"
            ),
            "prompt_strength": 0.35,
            "num_inference_steps": 50,
            "guidance_scale": 8,
            "seed": seed,
            "width": 1024,
            "height": 1024,
        },
    )

    result_url = _extract_url(output)
    if not result_url:
        raise RuntimeError("SDXL returned no output")

    urllib.request.urlretrieve(result_url, str(output_path))
    return output_path


def _harmonize_style(
    staged_path: Path,
    reference_path: Path,
    style: str,
    seed: int,
    output_path: Path,
) -> Path:
    """Run a style-transfer pass so this room matches the reference room's aesthetic."""
    style_desc = STYLE_PROMPTS.get(style, f"{style} interior design style.")
    prompt = f"Interior room staged with cohesive furniture and decor. {style_desc} Photorealistic real estate photography."

    logger.info(f"Harmonizing {staged_path.name} to match {reference_path.name}")

    output = replicate.run(
        MODELS["style_transfer"],
        input={
            "style_image": open(reference_path, "rb"),
            "structure_image": open(staged_path, "rb"),
            "prompt": prompt,
            "negative_prompt": (
                "changing room structure, changing walls, changing floors, "
                "changing ceiling, changing windows, changing doors, "
                "blurry, low quality, distorted, deformed, watermark, "
                "cartoon, illustration, anime, CGI, unrealistic"
            ),
            "structure_depth_strength": 1.5,
            "structure_denoising_strength": 0.25,
            "model": "realistic",
            "seed": seed,
            "output_format": "jpg",
            "output_quality": 90,
        },
    )

    result_url = _extract_url(output)
    if not result_url:
        raise RuntimeError("Style transfer returned no output")

    urllib.request.urlretrieve(result_url, str(output_path))
    logger.info(f"Harmonized style saved to {output_path}")
    return output_path


def stage_photo(
    image_path: Path,
    room_type: str = "living",
    style: str = "modern",
    seed: int | None = None,
    output_path: Path | None = None,
    reference_image: Path | None = None,
    room_status: str = "empty",
    **kwargs,
) -> Path:
    if output_path is None:
        output_path = image_path.parent / f"staged_{image_path.stem}.jpg"

    if seed is None:
        seed = random.randint(0, 2**31 - 1)

    logger.info(f"Staging {image_path.name} as {room_type}/{style} (seed={seed}, room_status={room_status})")

    if room_status == "empty":
        # Gemini for empty rooms — image generation with structure preservation
        logger.info("Using Gemini for empty room staging")
        _stage_gemini(image_path, room_type, style, output_path)
    else:
        # SDXL for furnished rooms — subtle furniture swap
        logger.info("Using SDXL img2img for furnished room")
        _stage_sdxl(image_path, room_type, style, seed, output_path)

    logger.info(f"Saved staged image to {output_path}")
    return output_path
