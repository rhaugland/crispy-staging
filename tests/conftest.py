# tests/conftest.py
import io
import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def sample_image():
    """A minimal 1920x1080 RGB image as numpy array."""
    return np.zeros((1080, 1920, 3), dtype=np.uint8)


@pytest.fixture
def sample_image_bytes():
    """A minimal 1920x1080 JPEG as bytes."""
    img = Image.new("RGB", (1920, 1080), "white")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def sample_rgba():
    """A minimal 1920x1080 RGBA image as numpy array."""
    return np.zeros((1080, 1920, 4), dtype=np.uint8)
