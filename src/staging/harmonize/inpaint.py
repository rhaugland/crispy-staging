from __future__ import annotations
import numpy as np


class EdgeInpainter:
    def create_edge_mask(self, furniture_rgba: np.ndarray, expand_px: int = 10) -> np.ndarray:
        alpha = furniture_rgba[:, :, 3]
        from scipy import ndimage
        dilated = ndimage.binary_dilation(alpha > 0, iterations=expand_px)
        eroded = ndimage.binary_erosion(alpha > 0, iterations=2)
        edge_mask = dilated & ~eroded
        return edge_mask.astype(np.uint8) * 255

    async def inpaint_edges(self, composited: np.ndarray, edge_mask: np.ndarray) -> np.ndarray:
        return await self._run_inpainting(composited, edge_mask)

    async def _run_inpainting(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Wire up SDXL Inpainting / Flux Fill")
