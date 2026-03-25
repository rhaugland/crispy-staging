from __future__ import annotations
import numpy as np


def estimate_color_temp(image: np.ndarray) -> float:
    mean_rgb = image.astype(np.float32).mean(axis=(0, 1))
    warmth = (mean_rgb[0] - mean_rgb[2])
    return float(warmth)


def adjust_color_temp(image: np.ndarray, shift: float) -> np.ndarray:
    result = image.astype(np.float32)
    factor = shift / 1000.0
    result[:, :, 0] += factor * 15
    result[:, :, 1] += factor * 5
    result[:, :, 2] -= factor * 15
    return np.clip(result, 0, 255).astype(np.uint8)


def match_color_to_reference(rendered: np.ndarray, reference: np.ndarray) -> np.ndarray:
    ref_temp = estimate_color_temp(reference)
    ren_temp = estimate_color_temp(rendered)
    shift = ref_temp - ren_temp
    result = adjust_color_temp(rendered, shift)
    ref_brightness = reference.astype(np.float32).mean()
    ren_brightness = result.astype(np.float32).mean()
    if ren_brightness > 0:
        brightness_ratio = ref_brightness / ren_brightness
        result = np.clip(result.astype(np.float32) * brightness_ratio, 0, 255).astype(np.uint8)
    return result
