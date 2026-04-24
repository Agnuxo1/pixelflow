"""Optional import: registers the 'moderngl' backend in the backend registry.

The core agent's ``pixelflow/backends/__init__.py`` may optionally import this
module to advertise the GPU backend.  If moderngl is not installed this file
still imports cleanly; the registration simply marks the backend as available
so that ``get_backend('moderngl')`` succeeds, and the actual ImportError is
deferred to the first call to ``run_moderngl``.
"""

from __future__ import annotations

from pixelflow.backends import _BACKENDS  # type: ignore[attr-defined]

_BACKENDS.add("moderngl")
