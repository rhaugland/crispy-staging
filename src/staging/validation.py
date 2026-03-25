from __future__ import annotations

import io
from dataclasses import dataclass

from PIL import Image

from staging.config import settings


class ValidationError(Exception):
    pass


@dataclass
class ValidationResult:
    valid: bool
    dimensions: list[tuple[int, int]]
    photos: list[bytes]


def validate_photos(photo_bytes_list: list[bytes]) -> ValidationResult:
    count = len(photo_bytes_list)
    if count < settings.min_photos:
        raise ValidationError(f"Upload at least {settings.min_photos} photos.")
    if count > settings.max_photos:
        raise ValidationError(f"Upload at most {settings.max_photos} photos.")

    dimensions: list[tuple[int, int]] = []
    processed: list[bytes] = []

    for i, raw in enumerate(photo_bytes_list):
        if len(raw) > settings.max_file_size_mb * 1024 * 1024:
            raise ValidationError(
                f"Photo {i + 1} exceeds max file size of {settings.max_file_size_mb}MB."
            )

        img = Image.open(io.BytesIO(raw))
        if img.format not in ("JPEG", "PNG"):
            raise ValidationError(f"Photo {i + 1}: unsupported format {img.format}. Use JPEG or PNG.")

        w, h = img.size
        min_w, min_h = settings.min_resolution
        max_w, max_h = settings.max_resolution

        if w < min_w or h < min_h:
            raise ValidationError(
                f"Photo {i + 1}: resolution {w}x{h} below minimum {min_w}x{min_h}."
            )

        if w > max_w or h > max_h:
            img.thumbnail((max_w, max_h), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=95)
            raw = buf.getvalue()
            w, h = img.size

        dimensions.append((w, h))
        processed.append(raw)

    return ValidationResult(valid=True, dimensions=dimensions, photos=processed)
