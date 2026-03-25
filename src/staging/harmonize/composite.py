from __future__ import annotations
import numpy as np


def alpha_composite(background: np.ndarray, foreground_rgba: np.ndarray) -> np.ndarray:
    fg_rgb = foreground_rgba[:, :, :3].astype(np.float32)
    alpha = foreground_rgba[:, :, 3:4].astype(np.float32) / 255.0
    bg = background.astype(np.float32)
    result = fg_rgb * alpha + bg * (1.0 - alpha)
    return np.clip(result, 0, 255).astype(np.uint8)


def apply_shadow(image: np.ndarray, shadow_rgba: np.ndarray) -> np.ndarray:
    alpha = shadow_rgba[:, :, 3:4].astype(np.float32) / 255.0
    img = image.astype(np.float32)
    shadow_factor = 1.0 - (alpha * 0.5)
    result = img * shadow_factor
    return np.clip(result, 0, 255).astype(np.uint8)
