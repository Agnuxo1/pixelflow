"""CA rule registry: numpy implementations + GLSL shaders."""

from __future__ import annotations

import logging
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

StepFn = Callable[[np.ndarray, dict, np.random.Generator], np.ndarray]
ValidateFn = Callable[[dict], None]


# ---------------------------------------------------------------------------
# diffusion_reaction  (Gray-Scott)
# ---------------------------------------------------------------------------

_DR_DEFAULT_PARAMS: dict[str, float] = {
    "feed": 0.055,
    "kill": 0.062,
    "dt": 1.0,
    "du": 0.16,
    "dv": 0.08,
}

_DR_GLSL = """\
#version 330 core

uniform sampler2D u_state;
uniform float u_feed;
uniform float u_kill;
uniform float u_dt;
uniform float u_du;
uniform float u_dv;
uniform vec2 u_texel_size;   // 1.0 / vec2(width, height)

out vec4 fragColor;

void main() {
    vec2 uv = gl_FragCoord.xy * u_texel_size;

    // Periodic neighbours (wrap via fract)
    vec4 c  = texture(u_state, uv);
    vec4 n  = texture(u_state, fract(uv + vec2( 0.0,  u_texel_size.y)));
    vec4 s  = texture(u_state, fract(uv + vec2( 0.0, -u_texel_size.y)));
    vec4 e  = texture(u_state, fract(uv + vec2( u_texel_size.x,  0.0)));
    vec4 w  = texture(u_state, fract(uv + vec2(-u_texel_size.x,  0.0)));

    float U = c.r;
    float V = c.g;

    float lapU = (n.r + s.r + e.r + w.r) - 4.0 * U;
    float lapV = (n.g + s.g + e.g + w.g) - 4.0 * V;

    float uvv = U * V * V;
    float newU = clamp(U + u_dt * (u_du * lapU - uvv + u_feed * (1.0 - U)), 0.0, 1.0);
    float newV = clamp(V + u_dt * (u_dv * lapV + uvv - (u_feed + u_kill) * V), 0.0, 1.0);

    fragColor = vec4(newU, newV, c.b, c.a);
}
"""


def _dr_step(
    state: np.ndarray, params: dict, rng: np.random.Generator
) -> np.ndarray:
    """One Gray-Scott step; state shape (H, W, 4)."""
    feed = float(params.get("feed", _DR_DEFAULT_PARAMS["feed"]))
    kill = float(params.get("kill", _DR_DEFAULT_PARAMS["kill"]))
    dt = float(params.get("dt", _DR_DEFAULT_PARAMS["dt"]))
    du = float(params.get("du", _DR_DEFAULT_PARAMS["du"]))
    dv = float(params.get("dv", _DR_DEFAULT_PARAMS["dv"]))

    U = state[..., 0]
    V = state[..., 1]

    # Discrete 2D Laplacian with periodic (roll) boundaries — no pixel loops
    lap_U = (
        np.roll(U, 1, axis=0) + np.roll(U, -1, axis=0)
        + np.roll(U, 1, axis=1) + np.roll(U, -1, axis=1)
        - 4.0 * U
    )
    lap_V = (
        np.roll(V, 1, axis=0) + np.roll(V, -1, axis=0)
        + np.roll(V, 1, axis=1) + np.roll(V, -1, axis=1)
        - 4.0 * V
    )

    uvv = U * V * V

    new_U = np.clip(U + dt * (du * lap_U - uvv + feed * (1.0 - U)), 0.0, 1.0)
    new_V = np.clip(V + dt * (dv * lap_V + uvv - (feed + kill) * V), 0.0, 1.0)

    out = state.copy()
    out[..., 0] = new_U
    out[..., 1] = new_V
    return out


def _dr_validate(params: dict) -> None:
    """Validate diffusion_reaction params."""
    valid = set(_DR_DEFAULT_PARAMS.keys())
    for k in params:
        if k not in valid:
            raise ValueError(f"Unknown diffusion_reaction param '{k}'. Valid: {valid}")
    for k in ("feed", "kill", "dt", "du", "dv"):
        v = params.get(k, _DR_DEFAULT_PARAMS[k])
        if float(v) <= 0:
            raise ValueError(f"Param '{k}' must be positive, got {v}")


# ---------------------------------------------------------------------------
# life_like  (continuous smoothed Conway over channel 0)
# ---------------------------------------------------------------------------

_LL_DEFAULT_PARAMS: dict[str, float] = {
    "threshold": 0.5,   # cells above this are 'alive'
    "noise": 0.0,       # optional additive noise std
}

_LL_GLSL = """\
#version 330 core

uniform sampler2D u_state;
uniform float u_threshold;
uniform float u_noise;        // currently unused in deterministic GPU path
uniform vec2 u_texel_size;

out vec4 fragColor;

void main() {
    vec2 uv = gl_FragCoord.xy * u_texel_size;

    float c00 = texture(u_state, fract(uv + vec2(-u_texel_size.x, -u_texel_size.y))).r;
    float c10 = texture(u_state, fract(uv + vec2( 0.0,            -u_texel_size.y))).r;
    float c20 = texture(u_state, fract(uv + vec2( u_texel_size.x, -u_texel_size.y))).r;
    float c01 = texture(u_state, fract(uv + vec2(-u_texel_size.x,  0.0           ))).r;
    float self= texture(u_state, uv).r;
    float c21 = texture(u_state, fract(uv + vec2( u_texel_size.x,  0.0           ))).r;
    float c02 = texture(u_state, fract(uv + vec2(-u_texel_size.x,  u_texel_size.y))).r;
    float c12 = texture(u_state, fract(uv + vec2( 0.0,             u_texel_size.y))).r;
    float c22 = texture(u_state, fract(uv + vec2( u_texel_size.x,  u_texel_size.y))).r;

    // Sum of 8 neighbours binarised at threshold
    float alive = step(u_threshold, self);
    float n_sum = step(u_threshold, c00) + step(u_threshold, c10)
                + step(u_threshold, c20) + step(u_threshold, c01)
                + step(u_threshold, c21) + step(u_threshold, c02)
                + step(u_threshold, c12) + step(u_threshold, c22);

    // Conway B3/S23 as continuous blending:
    //   born  if dead  and n_sum == 3
    //   alive if alive and (n_sum == 2 || n_sum == 3)
    float born    = (1.0 - alive) * step(2.5, n_sum) * step(n_sum, 3.5);
    float survive = alive         * step(1.5, n_sum) * step(n_sum, 3.5);
    float next = clamp(born + survive, 0.0, 1.0);

    vec4 s = texture(u_state, uv);
    fragColor = vec4(next, s.g, s.b, s.a);
}
"""


def _ll_step(
    state: np.ndarray, params: dict, rng: np.random.Generator
) -> np.ndarray:
    """One life-like CA step over channel 0."""
    threshold = float(params.get("threshold", _LL_DEFAULT_PARAMS["threshold"]))
    noise = float(params.get("noise", _LL_DEFAULT_PARAMS["noise"]))

    s = state[..., 0]
    alive = (s >= threshold).astype(np.float32)

    # 8-neighbour sum using roll (periodic boundary)
    n_sum = (
        np.roll(alive,  1, axis=0) + np.roll(alive, -1, axis=0)
        + np.roll(alive,  1, axis=1) + np.roll(alive, -1, axis=1)
        + np.roll(np.roll(alive,  1, axis=0),  1, axis=1)
        + np.roll(np.roll(alive,  1, axis=0), -1, axis=1)
        + np.roll(np.roll(alive, -1, axis=0),  1, axis=1)
        + np.roll(np.roll(alive, -1, axis=0), -1, axis=1)
    )

    born    = ((alive < 0.5) & (n_sum == 3)).astype(np.float32)
    survive = ((alive >= 0.5) & ((n_sum == 2) | (n_sum == 3))).astype(np.float32)
    next_s = np.clip(born + survive, 0.0, 1.0)

    if noise > 0.0:
        next_s = np.clip(next_s + rng.normal(0.0, noise, next_s.shape), 0.0, 1.0)

    out = state.copy()
    out[..., 0] = next_s
    return out


def _ll_validate(params: dict) -> None:
    """Validate life_like params."""
    valid = {"threshold", "noise"}
    for k in params:
        if k not in valid:
            raise ValueError(f"Unknown life_like param '{k}'. Valid: {valid}")


# ---------------------------------------------------------------------------
# wave  (discrete wave equation: amplitude=ch0, velocity=ch1)
# ---------------------------------------------------------------------------

_WV_DEFAULT_PARAMS: dict[str, float] = {
    "c": 0.5,
    "damping": 0.999,
    "dt": 1.0,
}

_WV_GLSL = """\
#version 330 core

uniform sampler2D u_state;
uniform float u_c;
uniform float u_damping;
uniform float u_dt;
uniform vec2 u_texel_size;

out vec4 fragColor;

void main() {
    vec2 uv = gl_FragCoord.xy * u_texel_size;

    vec4 cur = texture(u_state, uv);
    float amp = cur.r;
    float vel = cur.g;

    float n = texture(u_state, fract(uv + vec2( 0.0,  u_texel_size.y))).r;
    float s = texture(u_state, fract(uv + vec2( 0.0, -u_texel_size.y))).r;
    float e = texture(u_state, fract(uv + vec2( u_texel_size.x,  0.0))).r;
    float w = texture(u_state, fract(uv + vec2(-u_texel_size.x,  0.0))).r;

    float lap = n + s + e + w - 4.0 * amp;
    float c2 = u_c * u_c;
    float new_vel = (vel + u_dt * c2 * lap) * u_damping;
    float new_amp = clamp(amp + u_dt * new_vel, -1.0, 1.0);

    fragColor = vec4(new_amp, new_vel, cur.b, cur.a);
}
"""


def _wv_step(
    state: np.ndarray, params: dict, rng: np.random.Generator
) -> np.ndarray:
    """One discrete wave equation step; amplitude=ch0, velocity=ch1."""
    c = float(params.get("c", _WV_DEFAULT_PARAMS["c"]))
    damping = float(params.get("damping", _WV_DEFAULT_PARAMS["damping"]))
    dt = float(params.get("dt", _WV_DEFAULT_PARAMS["dt"]))

    amp = state[..., 0]
    vel = state[..., 1]

    lap = (
        np.roll(amp, 1, axis=0) + np.roll(amp, -1, axis=0)
        + np.roll(amp, 1, axis=1) + np.roll(amp, -1, axis=1)
        - 4.0 * amp
    )

    c2 = c * c
    new_vel = (vel + dt * c2 * lap) * damping
    new_amp = np.clip(amp + dt * new_vel, -1.0, 1.0)

    out = state.copy()
    out[..., 0] = new_amp
    out[..., 1] = new_vel
    return out


def _wv_validate(params: dict) -> None:
    """Validate wave params."""
    valid = {"c", "damping", "dt"}
    for k in params:
        if k not in valid:
            raise ValueError(f"Unknown wave param '{k}'. Valid: {valid}")
    if float(params.get("c", _WV_DEFAULT_PARAMS["c"])) <= 0:
        raise ValueError("wave param 'c' must be positive")
    d = float(params.get("damping", _WV_DEFAULT_PARAMS["damping"]))
    if not (0.0 < d <= 1.0):
        raise ValueError("wave param 'damping' must be in (0, 1]")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class RuleSpec:
    """Container for a registered CA rule."""

    def __init__(
        self,
        name: str,
        step_fn: StepFn,
        glsl: str,
        default_params: dict[str, Any],
        validate_fn: ValidateFn,
    ) -> None:
        self.name = name
        self.step = step_fn
        self.glsl = glsl
        self.default_params = default_params
        self.validate = validate_fn


_REGISTRY: dict[str, RuleSpec] = {
    "diffusion_reaction": RuleSpec(
        "diffusion_reaction",
        _dr_step,
        _DR_GLSL,
        dict(_DR_DEFAULT_PARAMS),
        _dr_validate,
    ),
    "life_like": RuleSpec(
        "life_like",
        _ll_step,
        _LL_GLSL,
        dict(_LL_DEFAULT_PARAMS),
        _ll_validate,
    ),
    "wave": RuleSpec(
        "wave",
        _wv_step,
        _WV_GLSL,
        dict(_WV_DEFAULT_PARAMS),
        _wv_validate,
    ),
}


def get_rule(name: str) -> RuleSpec:
    """Return a RuleSpec by name; raises KeyError if unknown."""
    if name not in _REGISTRY:
        raise KeyError(f"Unknown CA rule '{name}'. Available: {list(_REGISTRY)}")
    return _REGISTRY[name]


def list_rules() -> list[str]:
    """Return names of all registered CA rules."""
    return list(_REGISTRY.keys())


def register_rule(spec: RuleSpec) -> None:
    """Register a custom RuleSpec under spec.name."""
    _REGISTRY[spec.name] = spec
