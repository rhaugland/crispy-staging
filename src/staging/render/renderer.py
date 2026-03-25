from __future__ import annotations
from pathlib import Path

try:
    import bpy
except ImportError:
    bpy = None


class CyclesRenderer:
    """Configure and invoke Blender's Cycles renderer."""

    DEFAULT_SAMPLES = 128
    DEFAULT_DENOISE = True

    def configure(
        self,
        samples: int = DEFAULT_SAMPLES,
        denoise: bool = DEFAULT_DENOISE,
        use_gpu: bool = True,
    ) -> None:
        """Set Cycles render settings on the active scene."""
        scene = bpy.context.scene
        scene.render.engine = "CYCLES"
        scene.cycles.samples = samples
        scene.cycles.use_denoising = denoise

        prefs = bpy.context.preferences.addons["cycles"].preferences
        if use_gpu:
            try:
                prefs.compute_device_type = "CUDA"
                prefs.get_devices()
                for device in prefs.devices:
                    device.use = True
                scene.cycles.device = "GPU"
            except Exception:
                # Fall back to CPU silently if GPU setup fails
                scene.cycles.device = "CPU"
        else:
            scene.cycles.device = "CPU"

    def render(self, output_path: str) -> str:
        """Render the current scene to *output_path* (PNG) and return the path."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        bpy.context.scene.render.filepath = str(path)
        bpy.context.scene.render.image_settings.file_format = "PNG"
        bpy.ops.render.render(write_still=True)
        return str(path)

    def render_to_bytes(self) -> bytes:
        """Render to an in-memory buffer and return raw PNG bytes."""
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            self.render(tmp_path)
            with open(tmp_path, "rb") as f:
                return f.read()
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
