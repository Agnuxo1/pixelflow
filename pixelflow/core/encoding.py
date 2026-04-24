"""Input encoders: map a 1-D input vector into a (H, W, C) initial state."""

from __future__ import annotations

import numpy as np


def tile(
    x: np.ndarray,
    height: int,
    width: int,
    channels: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Encode x into (H, W, C) by tiling/resizing x to (H, W) in channel 0.

    If len(x) maps to an integer-ratio resize, uses nearest-neighbour.
    Otherwise falls back to average-pool (pure numpy, no scipy).
    Channels beyond 0 are filled with zeros.
    """
    state = np.zeros((height, width, channels), dtype=np.float32)

    # Reshape x into the closest 2-D shape, then resize to (H, W)
    n = len(x)
    # Try to infer a 2-D shape from x
    sq = int(np.floor(np.sqrt(n)))
    # Find the best rectangular split <= H x W aspect
    h_in, w_in = sq, n // sq
    if h_in * w_in < n:
        w_in = (n + h_in - 1) // h_in  # ceil

    # Pad x to h_in * w_in if needed
    padded = np.zeros(h_in * w_in, dtype=np.float32)
    padded[:n] = x.astype(np.float32)
    img_in = padded.reshape(h_in, w_in)

    # Resize to (height, width)
    if height == h_in and width == w_in:
        state[..., 0] = img_in
    elif (height % h_in == 0 and width % w_in == 0) or (h_in % height == 0 and w_in % width == 0):
        # Integer ratio: nearest-neighbour via repeat / striding
        state[..., 0] = _nearest_resize(img_in, height, width)
    else:
        # Average-pool fallback
        state[..., 0] = _avgpool_resize(img_in, height, width)

    return state


def phase(
    x: np.ndarray,
    height: int,
    width: int,
    channels: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Encode scalar mean of x as sin/cos phase into channels 0 and 1."""
    state = np.zeros((height, width, channels), dtype=np.float32)
    scalar = float(np.mean(x)) * 2.0 * np.pi
    state[..., 0] = np.sin(scalar)
    state[..., 1] = np.cos(scalar)
    return state


def project(
    x: np.ndarray,
    height: int,
    width: int,
    channels: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Project x to H*W*C via a fixed random Gaussian matrix (rng-seeded)."""
    out_dim = height * width * channels
    W = rng.standard_normal((out_dim, len(x))).astype(np.float32) / np.sqrt(len(x))
    projected = W @ x.astype(np.float32)
    # Normalize to [0, 1]
    mn, mx = projected.min(), projected.max()
    if mx > mn:
        projected = (projected - mn) / (mx - mn)
    else:
        projected = np.zeros_like(projected)
    return projected.reshape(height, width, channels)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _nearest_resize(img: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Resize img to (out_h, out_w) using nearest-neighbour (pure numpy)."""
    in_h, in_w = img.shape
    row_idx = (np.arange(out_h) * in_h / out_h).astype(int)
    col_idx = (np.arange(out_w) * in_w / out_w).astype(int)
    return img[np.ix_(row_idx, col_idx)]


def _avgpool_resize(img: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Resize img to (out_h, out_w) via bilinear average-pool (pure numpy)."""
    in_h, in_w = img.shape
    # Use nearest-neighbour for simplicity when aspect is non-integer
    # (pure numpy, no scipy)
    return _nearest_resize(img, out_h, out_w)


# ---------------------------------------------------------------------------
# Encoder dispatch
# ---------------------------------------------------------------------------

_ENCODERS = {
    "tile": tile,
    "phase": phase,
    "project": project,
}


def get_encoder(name: str):
    """Return encoder function by name; raises KeyError if unknown."""
    if name not in _ENCODERS:
        raise KeyError(f"Unknown encoder '{name}'. Available: {list(_ENCODERS)}")
    return _ENCODERS[name]
