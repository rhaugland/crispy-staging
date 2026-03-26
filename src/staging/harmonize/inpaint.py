from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

# Diffusers — only available on GPU workers
try:
    import torch
    from diffusers import AutoPipelineForInpainting
except ImportError:
    torch = None
    AutoPipelineForInpainting = None


class EdgeInpainter:
    def __init__(self):
        self._pipeline = None

    def _load_pipeline(self):
        if self._pipeline is not None:
            return
        if AutoPipelineForInpainting is None:
            raise ImportError(
                "diffusers is required for inpainting. "
                "Install with: pip install diffusers transformers torch"
            )
        self._pipeline = AutoPipelineForInpainting.from_pretrained(
            "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
            torch_dtype=torch.float16,
            variant="fp16",
        )
        if torch.cuda.is_available():
            self._pipeline = self._pipeline.to("cuda")

    def create_edge_mask(self, furniture_rgba: np.ndarray, expand_px: int = 10) -> np.ndarray:
        """Create a mask of the edges around placed furniture for inpainting."""
        alpha = furniture_rgba[:, :, 3]
        from scipy import ndimage
        dilated = ndimage.binary_dilation(alpha > 0, iterations=expand_px)
        eroded = ndimage.binary_erosion(alpha > 0, iterations=2)
        edge_mask = dilated & ~eroded
        return edge_mask.astype(np.uint8) * 255

    async def inpaint_edges(self, composited: np.ndarray, edge_mask: np.ndarray) -> np.ndarray:
        """Inpaint the edges where furniture meets the room for seamless blending.

        Args:
            composited: (H, W, 3) RGB uint8 — the composited image.
            edge_mask: (H, W) uint8 — mask where 255 = region to inpaint.

        Returns:
            (H, W, 3) RGB uint8 — inpainted image.
        """
        return await self._run_inpainting(composited, edge_mask)

    async def _run_inpainting(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        self._load_pipeline()
        from PIL import Image

        pil_image = Image.fromarray(image)
        pil_mask = Image.fromarray(mask)

        # Resize to SDXL-friendly dimensions (must be multiple of 8)
        orig_size = pil_image.size
        w = (orig_size[0] // 8) * 8
        h = (orig_size[1] // 8) * 8
        pil_image = pil_image.resize((w, h))
        pil_mask = pil_mask.resize((w, h))

        result = self._pipeline(
            prompt="seamless photorealistic interior, high quality",
            negative_prompt="artifacts, blurry, distorted, unrealistic",
            image=pil_image,
            mask_image=pil_mask,
            num_inference_steps=25,
            strength=0.5,
            guidance_scale=7.5,
        ).images[0]

        # Resize back to original dimensions
        result = result.resize(orig_size)
        return np.array(result)
