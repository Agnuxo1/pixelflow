# Benchmarks

All benchmarks write measured results as JSON to `benchmarks/results/` with
hardware + config metadata. Numbers are honest — failures and regressions are
reported as-is, not hidden.

## Running

```bash
python benchmarks/mnist_reservoir.py --subset 10000 --backend cpu
python benchmarks/mnist_reservoir.py --subset 10000 --backend moderngl  # requires pixelflow[gpu]
```

## Interpreting results

Each JSON contains:

- `config` — full argparse config
- `baseline_acc` — plain LogisticRegression on raw pixels
- `reservoir_acc` — Reservoir → LogisticReadout
- `verdict` — which beat the other on held-out test set
- timings for load, baseline fit, reservoir transform, readout fit

A reservoir that loses to the raw-pixel baseline is a *useful* negative result —
it tells us the chosen rule or encoding is not adding signal on this task.
Do not edit JSON artifacts.
