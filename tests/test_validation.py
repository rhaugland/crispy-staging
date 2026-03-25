import io
import pytest
from PIL import Image
from staging.validation import validate_photos, ValidationError


def _make_image(width: int, height: int, fmt: str = "JPEG") -> bytes:
    img = Image.new("RGB", (width, height), color="white")
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def test_valid_photos():
    photos = [_make_image(1920, 1080) for _ in range(3)]
    result = validate_photos(photos)
    assert result.valid is True


def test_too_few_photos():
    photos = [_make_image(1920, 1080) for _ in range(2)]
    with pytest.raises(ValidationError, match="at least 3"):
        validate_photos(photos)


def test_too_many_photos():
    photos = [_make_image(1920, 1080) for _ in range(5)]
    with pytest.raises(ValidationError, match="at most 4"):
        validate_photos(photos)


def test_resolution_too_low():
    photos = [_make_image(640, 480) for _ in range(3)]
    with pytest.raises(ValidationError, match="resolution"):
        validate_photos(photos)


def test_file_too_large():
    large = _make_image(8000, 6000)
    large_padded = large + b"\x00" * (21 * 1024 * 1024 - len(large))
    photos = [large_padded, _make_image(1920, 1080), _make_image(1920, 1080)]
    with pytest.raises(ValidationError, match="size"):
        validate_photos(photos)


def test_oversized_image_downscaled():
    photos = [_make_image(9000, 7000) for _ in range(3)]
    result = validate_photos(photos)
    assert result.valid is True
    for dim in result.dimensions:
        assert dim[0] <= 8000
        assert dim[1] <= 6000
