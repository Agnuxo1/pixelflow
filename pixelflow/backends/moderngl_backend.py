"""moderngl GPU backend for pixelflow.

Public API::

    run_moderngl(initial_state, rule, steps, rng) -> np.ndarray

Matches the CPU backend signature exactly.
"""

from __future__ import annotations

import numpy as np

from pixelflow.core.rules import RuleSpec

# ---------------------------------------------------------------------------
# Uniform name maps per rule (rule.name -> dict of param_key -> uniform_name)
# These must match the uniform names in the GLSL shaders in core/rules.py.
# ---------------------------------------------------------------------------

_UNIFORM_MAP: dict[str, dict[str, str]] = {
    "diffusion_reaction": {
        "feed": "u_feed",
        "kill": "u_kill",
        "dt":   "u_dt",
        "du":   "u_du",
        "dv":   "u_dv",
    },
    "life_like": {
        "threshold": "u_threshold",
        "noise":     "u_noise",
    },
    "wave": {
        "c":       "u_c",
        "damping": "u_damping",
        "dt":      "u_dt",
    },
}


def _set_uniforms(prog, rule: RuleSpec, params: dict, width: int, height: int) -> None:
    """Write per-rule parameter uniforms and the texel-size helper."""
    umap = _UNIFORM_MAP.get(rule.name, {})
    merged = {**rule.default_params, **params}

    for param_key, uniform_name in umap.items():
        if uniform_name in prog:
            prog[uniform_name].value = float(merged[param_key])

    # All shaders use u_texel_size for periodic wrapping.
    if "u_texel_size" in prog:
        prog["u_texel_size"].value = (1.0 / width, 1.0 / height)


def run_moderngl(
    initial_state: np.ndarray,
    rule: RuleSpec,
    steps: int,
    rng: np.random.Generator,
    rule_params: dict | None = None,
) -> np.ndarray:
    """Run *rule* for *steps* on *initial_state* using the moderngl GPU backend.

    Parameters
    ----------
    initial_state:
        Float32 array of shape (H, W, 4).
    rule:
        A RuleSpec from ``pixelflow.core.rules``.
    steps:
        Number of CA evolution steps.
    rng:
        A numpy random Generator (unused on GPU; present for API parity with
        the CPU backend — determinism is provided by the fixed shader).
    rule_params:
        Optional parameter overrides for the rule (merged with defaults).

    Returns
    -------
    np.ndarray
        Float32 array of shape (H, W, 4) — the final CA state.

    Raises
    ------
    ImportError
        If moderngl is not installed.
    RuntimeError
        If a standalone OpenGL context cannot be created.
    """
    try:
        import moderngl
    except ImportError as exc:
        raise ImportError(
            "moderngl backend requires moderngl to be installed. "
            "Run: pip install pixelflow[gpu]"
        ) from exc

    if rule_params is None:
        rule_params = {}

    state = np.ascontiguousarray(initial_state, dtype=np.float32)
    if state.ndim != 3 or state.shape[2] != 4:
        raise ValueError(
            f"initial_state must have shape (H, W, 4), got {state.shape}"
        )
    H, W = state.shape[:2]

    # ------------------------------------------------------------------
    # Create standalone (headless) context.
    # ------------------------------------------------------------------
    try:
        ctx = moderngl.create_standalone_context()
    except Exception as exc:
        raise RuntimeError(
            f"Failed to create a moderngl standalone context: {exc}. "
            "Ensure your system has OpenGL drivers installed (or a software "
            "renderer such as Mesa). On headless servers, set DISPLAY or use "
            "EGL. Callers may fall back to the CPU backend."
        ) from exc

    from pixelflow.backends.moderngl_shaders import build_program

    ctx_ok = False
    try:
        # ------------------------------------------------------------------
        # Compile shader program.
        # ------------------------------------------------------------------
        prog = build_program(ctx, rule.glsl)

        # Bind the texture sampler to unit 0.
        if "u_state" in prog:
            prog["u_state"].value = 0

        # ------------------------------------------------------------------
        # Allocate two RGBA32F textures for ping-pong.
        # moderngl texture data: numpy array flattened to bytes, row-major.
        # ------------------------------------------------------------------
        tex_a = ctx.texture((W, H), 4, dtype="f4", data=state.tobytes())
        tex_b = ctx.texture((W, H), 4, dtype="f4")

        tex_a.filter = (moderngl.NEAREST, moderngl.NEAREST)
        tex_b.filter = (moderngl.NEAREST, moderngl.NEAREST)

        # Two framebuffers: one renders into tex_a, one into tex_b.
        fbo_a = ctx.framebuffer(color_attachments=[tex_a])
        fbo_b = ctx.framebuffer(color_attachments=[tex_b])

        # A dummy VAO for the attribute-less full-screen draw.
        vao = ctx.vertex_array(prog, [])

        # Set rule uniforms (static across steps — params don't change).
        _set_uniforms(prog, rule, rule_params, W, H)

        # ------------------------------------------------------------------
        # Ping-pong rendering loop.
        # src_tex -> read  |  dst_fbo -> write, then swap.
        # ------------------------------------------------------------------
        src_tex = tex_a
        dst_fbo = fbo_b
        alt_tex = tex_b
        alt_fbo = fbo_a

        for _ in range(steps):
            dst_fbo.use()
            ctx.viewport = (0, 0, W, H)
            src_tex.use(location=0)
            vao.render(moderngl.TRIANGLES, vertices=3)

            # Swap
            src_tex, alt_tex = alt_tex, src_tex
            dst_fbo, alt_fbo = alt_fbo, dst_fbo

        # After the loop, src_tex holds the latest result (the last write went
        # into dst_fbo whose attachment is now aliased via the swap above; we
        # need the texture that was last written, which is now alt_tex before
        # the last swap).  Let's re-derive cleanly:
        #
        # After loop body, we swapped, so the texture last written is src_tex.
        result_tex = src_tex

        # ------------------------------------------------------------------
        # Read back pixels.
        # ------------------------------------------------------------------
        raw = result_tex.read()
        result = np.frombuffer(raw, dtype=np.float32).reshape(H, W, 4)

        ctx_ok = True
        return result.copy()

    finally:
        if not ctx_ok:
            # Attempt to clean up even on error.
            pass
        # Always release GL resources.
        try:
            ctx.release()
        except Exception:
            pass
