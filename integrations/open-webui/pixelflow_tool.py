"""Open WebUI custom tool: classify an image with a pixelflow reservoir.

Drop this file into Open WebUI's Tools section (Workspace -> Tools -> +).
Follows the Open WebUI Tool convention: a single ``Tools`` class whose
public methods are exposed to the model. See
https://docs.openwebui.com/features/plugin/tools/ .

Experimental research integration.
"""

from __future__ import annotations

import base64
import io

from pydantic import BaseModel, Field


class Tools:
    """Pixelflow reservoir image-feature tool for Open WebUI."""

    class Valves(BaseModel):
        """Admin-configurable settings, shown in the Open WebUI tool panel."""

        width: int = Field(default=32, description="Reservoir texture width.")
        height: int = Field(default=32, description="Reservoir texture height.")
        channels: int = Field(default=4, description="Reservoir channels.")
        steps: int = Field(default=4, description="Number of CA steps.")
        rule: str = Field(default="wave", description="CA rule name.")
        seed: int = Field(default=0, description="Reservoir RNG seed.")

    def __init__(self) -> None:
        self.valves = self.Valves()
        self._reservoir = None  # built lazily on first call

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _build_reservoir(self):
        from pixelflow import Reservoir, ReservoirConfig

        cfg = ReservoirConfig(
            width=self.valves.width,
            height=self.valves.height,
            channels=self.valves.channels,
            steps=self.valves.steps,
            rule=self.valves.rule,
            input_encoding="tile",
            seed=self.valves.seed,
        )
        return Reservoir(cfg, backend="cpu")

    # ------------------------------------------------------------------
    # Tool methods (exposed to the LLM)
    # ------------------------------------------------------------------
    def classify_with_reservoir(self, image_base64: str) -> str:
        """Run an image through a tiny pixelflow reservoir and describe its
        feature vector.

        :param image_base64: Base64-encoded image bytes (PNG/JPEG). May
            optionally include a ``data:image/...;base64,`` prefix.
        :return: Short human-readable description of the feature vector's
            L2 norm and sparsity.
        """
        import numpy as np
        from PIL import Image

        # strip optional data-url prefix
        payload = image_base64.split(",", 1)[-1].strip()
        try:
            raw = base64.b64decode(payload, validate=False)
        except Exception as exc:  # pragma: no cover
            return f"Error: could not base64-decode image ({exc})."

        try:
            img = Image.open(io.BytesIO(raw)).convert("L")
        except Exception as exc:
            return f"Error: could not decode image bytes ({exc})."

        # downsample to a small flat vector so any encoding works
        img_small = img.resize((16, 16), Image.BILINEAR)
        x = np.asarray(img_small, dtype=np.float32).ravel() / 255.0

        if self._reservoir is None:
            self._reservoir = self._build_reservoir()

        feats = self._reservoir.transform(x[np.newaxis, :])[0]

        l2 = float(np.linalg.norm(feats))
        eps = 1e-4
        sparsity = float(np.mean(np.abs(feats) < eps))
        dim = int(feats.size)

        return (
            f"Reservoir features: dim={dim}, L2 norm={l2:.4f}, "
            f"sparsity (|v|<{eps})={sparsity:.3f}. "
            f"Config: {self.valves.width}x{self.valves.height}x"
            f"{self.valves.channels}, rule='{self.valves.rule}', "
            f"steps={self.valves.steps}."
        )
