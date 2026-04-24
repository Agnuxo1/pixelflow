"""CIFAR-10 benchmark for pixelflow.

Reservoir + LogisticReadout vs raw-pixel LogisticRegression baseline.
Results written to benchmarks/results/cifar10_reservoir_<ts>.json.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

from sklearn.linear_model import LogisticRegression

from pixelflow import Reservoir, ReservoirConfig, LogisticReadout
from pixelflow.tasks.cifar10 import load as load_cifar10


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--subset", type=int, default=10000)
    p.add_argument("--backend", choices=["cpu", "moderngl", "cuda"], default="cpu")
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--height", type=int, default=32)
    p.add_argument("--steps", type=int, default=6)
    p.add_argument("--rule", default="wave")
    p.add_argument("--grayscale", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    results: dict = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "config": vars(args),
    }

    print(f"[cifar10] loading subset={args.subset} grayscale={args.grayscale} ...")
    t0 = time.perf_counter()
    X_train, y_train, X_test, y_test = load_cifar10(
        subset=args.subset, seed=args.seed, grayscale=args.grayscale
    )
    results["load_time_s"] = time.perf_counter() - t0
    print(f"[cifar10]  loaded in {results['load_time_s']:.2f}s  "
          f"train={X_train.shape} test={X_test.shape}")

    print("[baseline] fitting logistic regression on raw pixels ...")
    t0 = time.perf_counter()
    baseline = LogisticRegression(max_iter=500, n_jobs=-1, solver="lbfgs").fit(X_train, y_train)
    baseline_acc = float(baseline.score(X_test, y_test))
    results["baseline_acc"] = baseline_acc
    results["baseline_fit_time_s"] = time.perf_counter() - t0
    print(f"[baseline]  acc={baseline_acc:.4f}  ({results['baseline_fit_time_s']:.1f}s)")

    cfg = ReservoirConfig(
        width=args.width, height=args.height, channels=4,
        steps=args.steps, rule=args.rule,
        input_encoding="tile" if args.grayscale else "project",
        seed=args.seed,
    )
    print(f"[reservoir] {cfg}  backend={args.backend}")
    res = Reservoir(cfg, backend=args.backend)

    t0 = time.perf_counter()
    F_train = res.transform(X_train)
    results["reservoir_transform_train_s"] = time.perf_counter() - t0
    print(f"[reservoir]  train features {F_train.shape} in "
          f"{results['reservoir_transform_train_s']:.2f}s")

    t0 = time.perf_counter()
    F_test = res.transform(X_test)
    results["reservoir_transform_test_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    readout = LogisticReadout(C=1.0, max_iter=500).fit(F_train, y_train)
    results["readout_fit_s"] = time.perf_counter() - t0
    reservoir_acc = float(readout.score(F_test, y_test))
    results["reservoir_acc"] = reservoir_acc
    print(f"[reservoir]  acc={reservoir_acc:.4f}  (readout fit "
          f"{results['readout_fit_s']:.1f}s)")

    results["verdict"] = (
        "reservoir_beats_baseline" if reservoir_acc > baseline_acc
        else "baseline_beats_reservoir"
    )

    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"cifar10_reservoir_{stamp}.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"[done] wrote {out_path}")


if __name__ == "__main__":
    main()
