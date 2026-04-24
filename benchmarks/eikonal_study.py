"""Quantitative Eikonal study.

Compares the fast-marching reference solver (scikit-fmm-style, pure python)
against a simple wave-reservoir arrival-time estimate across a grid of speeds,
sources, and step counts. Writes a summary JSON with relative errors.

This is NOT a claim that the wave reservoir solves the Eikonal equation
exactly (it does not — the wave equation is second-order in time, Eikonal is
first-order). The point is to measure how close a simple-to-set-up reservoir
can get and where it breaks down.
"""
from __future__ import annotations

import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from pixelflow import Reservoir, ReservoirConfig
from pixelflow.tasks.eikonal import solve_reference


def wave_reservoir_arrival_time(
    grid: np.ndarray, source: tuple[int, int], steps: int, dt: float = 1.0
) -> np.ndarray:
    """Simulate a wave pulse from `source` through a speed field, return first-arrival times.

    This is the naive reservoir analogue of the Eikonal solver: place an
    impulse at the source, run the wave rule with per-cell c = speed_field,
    and record the first time each cell's amplitude exceeds a threshold.
    """
    h, w = grid.shape
    state = np.zeros((h, w, 4), dtype=np.float32)
    state[source[0], source[1], 0] = 1.0  # initial amplitude

    cfg = ReservoirConfig(
        width=w, height=h, channels=4, steps=1,
        rule="wave",
        rule_params={"c": float(np.mean(grid)), "damping": 1.0, "dt": dt},
        input_encoding="tile", seed=0,
    )
    res = Reservoir(cfg, backend="cpu")

    arrival = np.full((h, w), np.inf, dtype=np.float32)
    arrival[source[0], source[1]] = 0.0
    thresh = 0.02
    for t in range(1, steps + 1):
        state = res.step(state)
        reached = (np.abs(state[..., 0]) > thresh) & np.isinf(arrival)
        arrival[reached] = t * dt
    arrival[np.isinf(arrival)] = steps * dt
    return arrival


def main() -> None:
    results: dict = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "cases": [],
    }
    cases = [
        {"size": 32, "source": (16, 16), "steps": 40, "speed": "constant"},
        {"size": 64, "source": (32, 32), "steps": 80, "speed": "constant"},
        {"size": 64, "source": (10, 10), "steps": 100, "speed": "constant"},
        {"size": 64, "source": (32, 32), "steps": 80, "speed": "radial"},
    ]
    for case in cases:
        s = case["size"]
        if case["speed"] == "constant":
            grid = np.ones((s, s), dtype=np.float32)
        else:  # radial: slower near edges
            ys, xs = np.mgrid[0:s, 0:s]
            r = np.sqrt((ys - s / 2) ** 2 + (xs - s / 2) ** 2)
            grid = (1.0 - 0.5 * r / r.max()).astype(np.float32)

        src = case["source"]
        t0 = time.perf_counter()
        ref = solve_reference(grid, src)
        t_ref = time.perf_counter() - t0

        t0 = time.perf_counter()
        wave = wave_reservoir_arrival_time(grid, src, steps=case["steps"])
        t_wave = time.perf_counter() - t0

        # Normalise both to same scale (the wave reservoir has no notion of
        # physical time units; we fit a scalar best-fit factor).
        mask = np.isfinite(ref) & (wave < case["steps"])
        if mask.sum() < 10:
            rel_err = float("nan")
        else:
            scale = np.dot(ref[mask], wave[mask]) / (np.dot(wave[mask], wave[mask]) + 1e-9)
            rel_err = float(np.mean(np.abs(scale * wave[mask] - ref[mask]) / (ref[mask] + 1e-6)))

        case_out = {**case, "ref_time_s": t_ref, "wave_time_s": t_wave,
                    "mean_relative_error": rel_err, "valid_cells": int(mask.sum())}
        results["cases"].append(case_out)
        print(f"[eikonal] {case}  rel_err={rel_err:.3f}  valid={mask.sum()}")

    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"eikonal_study_{stamp}.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"[done] wrote {out_path}")


if __name__ == "__main__":
    main()
