import numpy as np
import pytest
from staging.harmonize.composite import alpha_composite, apply_shadow


def test_alpha_composite_transparent_over_background():
    bg = np.full((100, 100, 3), 200, dtype=np.uint8)
    fg_rgba = np.zeros((100, 100, 4), dtype=np.uint8)
    fg_rgba[40:60, 40:60] = [255, 0, 0, 255]
    result = alpha_composite(background=bg, foreground_rgba=fg_rgba)
    assert result.shape == (100, 100, 3)
    assert np.array_equal(result[50, 50], [255, 0, 0])
    assert np.array_equal(result[0, 0], [200, 200, 200])


def test_alpha_composite_semi_transparent():
    bg = np.full((100, 100, 3), 0, dtype=np.uint8)
    fg_rgba = np.zeros((100, 100, 4), dtype=np.uint8)
    fg_rgba[50, 50] = [255, 255, 255, 128]
    result = alpha_composite(background=bg, foreground_rgba=fg_rgba)
    assert 120 <= result[50, 50, 0] <= 136


def test_apply_shadow_darkens():
    bg = np.full((100, 100, 3), 200, dtype=np.uint8)
    shadow = np.full((100, 100, 4), [0, 0, 0, 128], dtype=np.uint8)
    result = apply_shadow(image=bg, shadow_rgba=shadow)
    assert result[50, 50, 0] < 200
