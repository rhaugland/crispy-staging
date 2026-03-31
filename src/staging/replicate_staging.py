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
    "living": "exactly one 3-seat sofa, exactly two accent chairs, one coffee table, two side tables, one area rug, two table lamps, throw pillows, and one piece of wall art",
    "bedroom": "a bed with headboard and bedding, nightstands, table lamps, a dresser, an accent chair, a rug, and wall art",
    "dining": "a dining table, dining chairs, a sideboard or buffet, a centerpiece, table settings, a rug, and wall art",
    "kitchen": "bar stools, a kitchen island arrangement, pendant lights, decorative bowls, plants, and countertop accessories",
    "office": "a desk, an office chair, a bookshelf, a desk lamp, storage, a rug, and wall art",
    "bathroom": "towels on racks, bath mat, countertop accessories, a plant, candles, and decorative storage",
}

STYLE_PROMPTS = {
    "modern": (
        "Furniture: charcoal gray linen 3-seat track-arm sofa with square cushions, "
        "two matching charcoal gray linen square-back armchairs with chrome legs, "
        "round white lacquer coffee table, chrome arc floor lamp, "
        "black and white geometric area rug, abstract black and white art print on wall."
    ),
    "traditional": (
        "Furniture: navy blue tufted velvet rolled-arm 3-seat sofa, "
        "two matching burgundy velvet wingback armchairs with dark cherry wood legs, "
        "dark cherry wood rectangular coffee table, burgundy throw pillows, "
        "red and navy oriental area rug, brass candlestick table lamps, gold-framed oil painting on wall."
    ),
    "scandinavian": (
        "Furniture: white linen 3-seat sofa with rounded arms, "
        "two matching pale gray linen rounded armchairs with light oak legs, "
        "round light oak coffee table, pale blush throw pillows, light gray wool area rug, "
        "white ceramic vases, simple line art print in thin black frame on wall."
    ),
    "luxury": (
        "Furniture: emerald green velvet tufted 3-seat Chesterfield sofa, "
        "two matching navy velvet barrel-back armchairs with gold legs, "
        "round white marble coffee table with gold legs, crystal table lamps, "
        "cream silk throw pillows, ivory thick pile area rug, large gold-framed abstract art on wall."
    ),
    "coastal": (
        "Furniture: white slipcovered rolled-arm 3-seat sofa, light blue throw pillows, "
        "two matching round natural rattan armchairs with white cushions, "
        "rectangular driftwood coffee table, "
        "beige seagrass area rug, blue and white nautical art print on wall."
    ),
    "midcenturymodern": (
        "Furniture: mustard yellow upholstered 3-seat sofa with tapered walnut legs, "
        "two matching olive green upholstered armchairs with tapered walnut legs, "
        "round warm walnut coffee table, burnt orange throw pillows, "
        "cream and tan geometric area rug, abstract orange and gold art print on wall."
    ),
    "farmhouse": (
        "Furniture: cream linen slipcovered rolled-arm 3-seat sofa, "
        "two matching natural linen armchairs with distressed white oak legs, "
        "rectangular distressed white oak coffee table, sage green throw pillows, "
        "natural jute area rug, galvanized metal table lamp, "
        "vintage botanical print in white frame on wall."
    ),
    "industrial": (
        "Furniture: dark brown leather tufted 3-seat Chesterfield sofa, "
        "two matching dark brown leather club armchairs with riveted details, "
        "rectangular black iron and reclaimed wood coffee table, "
        "matte black metal tripod floor lamp, charcoal gray wool area rug, "
        "vintage black and white photography print on wall."
    ),
    "minimalist": (
        "Furniture: white low-profile 3-seat platform sofa with clean lines, "
        "one matching white low-profile armchair with clean lines, "
        "round light ash wood coffee table, single matte black floor lamp, "
        "off-white wool area rug, one large minimalist black and white art print on wall."
    ),
    "contemporary": (
        "Furniture: taupe bouclé curved 3-seat sofa, "
        "two matching taupe bouclé rounded armchairs with brass legs, "
        "round white marble coffee table with brass legs, terracotta accent pillow, "
        "warm beige wool area rug, large-scale earth-toned abstract art on wall."
    ),
    "boho": (
        "Furniture: low terracotta linen 3-seat floor sofa, colorful embroidered throw pillows, "
        "two matching round rattan peacock armchairs with colorful cushions, "
        "multicolor kilim area rug, "
        "macramé wall hanging, trailing pothos plant in woven basket."
    ),
    "rustic": (
        "Furniture: cognac brown leather 3-seat rolled-arm sofa, "
        "two matching cognac brown leather club armchairs, "
        "heavy rectangular dark pine wood coffee table, cream wool throw blanket, "
        "wrought iron table lamp, brown and tan cowhide area rug, "
        "landscape oil painting in wood frame on wall."
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
                    "In 2-3 sentences, describe the furniture style: wood tone, "
                    "fabric colors, material types, metal finishes, and rug color. "
                    "Example: 'Dark walnut wood with navy velvet upholstery. "
                    "Brass accents and cream wool rug.' "
                    "Do NOT mention walls, floors, countertops, cabinets, or room structure."
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
    seed: int = 0,
) -> Path:
    """Stage an empty room using Gemini image generation."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set")

    room_name = ROOM_TYPE_MAP.get(room_type, room_type)
    furniture = ROOM_FURNITURE.get(room_type, "furniture, rugs, lamps, and decor")
    style_desc = STYLE_PROMPTS.get(style, f"{style} interior design style.")

    furniture_match = ""
    if furniture_description:
        furniture_match = f" Use these specific colors and materials: {furniture_description}"

    prompt = (
        f"RULE 1 — PRESERVE EVERYTHING: Do not remove, replace, or alter anything already in this photo. "
        f"Every countertop stays exactly as-is — same material, color, shape, fully visible, fully opaque, fully solid. Do not make countertops transparent or translucent. "
        f"Every cabinet, sink, appliance, fixture stays exactly as-is. "
        f"Walls, floors, ceiling, windows, doors, trim, molding, vents, outlets — all unchanged. "
        f"The camera has NOT moved — same angle, same height, same position, same lens, same field of view. Do not zoom, crop, rotate, or shift the viewpoint at all. The output must be pixel-aligned with the input photo. "
        f"RULE 2 — ADD FURNITURE: Edit this photo of an empty {room_name}. "
        f"Place furniture ONLY in empty floor space, never where countertops or cabinets are. "
        f"Add exactly: {furniture}. "
        f"RULE 3 — EXACT STYLE: {style_desc}{furniture_match} "
        f"Use EXACTLY the colors and materials specified — do not adapt them to match the room. "
        f"RULE 4 — EXACT COUNT: Place exactly ONE sofa and exactly TWO accent chairs. No more, no fewer. "
        f"The sofa must be the exact same style described above in every photo. "
        f"Photorealistic real estate photograph."
    )

    logger.info(f"Gemini: staging {image_path.name} as {room_name}/{style}")

    image_data = image_path.read_bytes()
    image_b64 = base64.b64encode(image_data).decode("utf-8")
    ext = image_path.suffix.lower()
    mime = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png"

    payload = json.dumps({
        "contents": [{
            "parts": [
                {"text": prompt},
                {"inline_data": {"mime_type": mime, "data": image_b64}},
            ],
        }],
        "generationConfig": {
            "responseModalities": ["IMAGE"],
            "seed": seed,
        },
    }).encode("utf-8")

    req = urllib.request.Request(
        f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3-pro-image-preview:generateContent?key={api_key}",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    import time as _time
    last_error = None
    result = None
    for attempt in range(3):
        if attempt > 0:
            wait = 10 * attempt
            logger.info(f"Gemini: retry {attempt}/2 after {wait}s...")
            _time.sleep(wait)
            # rebuild request (urlopen consumes it)
            req = urllib.request.Request(
                f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3-pro-image-preview:generateContent?key={api_key}",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                result = json.loads(resp.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8", errors="replace")
            logger.warning(f"Gemini API error {e.code} (attempt {attempt+1}): {error_body}")
            if e.code in (503, 429, 500):
                last_error = RuntimeError(f"Gemini API error {e.code}: {error_body}")
                continue
            raise RuntimeError(f"Gemini API error {e.code}: {error_body}") from e
        except (urllib.error.URLError, TimeoutError) as e:
            logger.warning(f"Gemini timeout/network error (attempt {attempt+1}): {e}")
            last_error = RuntimeError(f"Gemini network error: {e}")
            continue
    else:
        raise last_error

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
            "structure_depth_strength": 2.0,
            "structure_denoising_strength": 0.20,
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
        _stage_gemini(image_path, room_type, style, output_path, seed=seed)
    else:
        # SDXL for furnished rooms — subtle furniture swap
        logger.info("Using SDXL img2img for furnished room")
        _stage_sdxl(image_path, room_type, style, seed, output_path)


    logger.info(f"Saved staged image to {output_path}")
    return output_path
