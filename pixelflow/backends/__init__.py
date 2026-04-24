"""Backend registry for pixelflow."""

from __future__ import annotations

_BACKENDS = {"cpu", "moderngl"}


def get_backend(name: str) -> str:
    """Return validated backend name; raises ValueError if unsupported."""
    if name not in _BACKENDS:
        raise ValueError(f"Unknown backend '{name}'. Available: {sorted(_BACKENDS)}")
    return name
