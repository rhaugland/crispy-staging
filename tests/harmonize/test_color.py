import numpy as np
import pytest
from staging.harmonize.color import estimate_color_temp, adjust_color_temp


def test_estimate_color_temp_warm():
    warm = np.full((100, 100, 3), [220, 180, 140], dtype=np.uint8)
    temp = estimate_color_temp(warm)
    assert temp > 0


def test_estimate_color_temp_cool():
    cool = np.full((100, 100, 3), [140, 160, 220], dtype=np.uint8)
    temp = estimate_color_temp(cool)
    assert temp < 0


def test_adjust_shifts_color():
    neutral = np.full((100, 100, 3), [180, 180, 180], dtype=np.uint8)
    shifted = adjust_color_temp(neutral, shift=500)
    assert shifted[50, 50, 0] > neutral[50, 50, 0]
    assert shifted[50, 50, 2] < neutral[50, 50, 2]
