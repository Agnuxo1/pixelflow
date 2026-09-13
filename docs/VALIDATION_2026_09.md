# Reproducible validation after the batch-invariance correction

## What was tested

The corrected projection encoder and the scikit-learn adapter were evaluated
on the bundled [scikit-learn digits dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html).
These are 8x8 digits, **not MNIST**. No dataset download, API key, GPU or paid
service is required to reproduce this evaluation.

The protocol was fixed before this run: five stratified 75/25 splits with seeds
0 through 4, a raw-input baseline, projection-only ablation, and two reservoir
steps. All models use the same train/test indices, training-only StandardScaler,
and LogisticRegression(C=1, max_iter=2000). No hyperparameter search was used.
The reservoir is 8x8x4, diffusion_reaction, project encoding, CPU backend.

## Results

| Feature map | Mean accuracy | Standard deviation across splits |
|---|---:|---:|
| Raw inputs | 96.933% | 1.137 percentage points |
| Projection only, zero evolution steps | 96.800% | 0.412 percentage points |
| Reservoir, two evolution steps | 96.978% | 0.518 percentage points |

The reservoir-minus-raw differences, in percentage points, were
+0.667, -0.667, +1.778, -1.111 and -0.444.
The mean difference is +0.044 percentage points: two wins and three losses.
**This does not establish superiority over the conventional baseline.**
The partitions overlap, so their standard deviation is descriptive, not a
confidence interval from independent experimental replications. This evaluation
also does not establish equivalence, generalization to other datasets, energy
savings, or speed advantages.

## Reproduce

From a checkout containing this benchmark:

```bash
python -m pip install -e .
python -m benchmarks.reproducible_digits --output benchmarks/results/my_digits_run.json
```

Original environment: Python 3.13.7, NumPy 2.2.6, scikit-learn 1.4.0; all
dependency versions and the platform are recorded in the JSON. Dependency and
solver changes may change the numeric results. The script records its SHA-256,
base Git revision, dataset hashes, split hashes and prediction hashes. The
classifier's iteration count is included for convergence inspection.

Raw record: [digits_validation_20260913.json](../benchmarks/results/digits_validation_20260913.json).
Executable protocol: [reproducible_digits.py](../benchmarks/reproducible_digits.py).
Protocol tests: [test_benchmark_protocol.py](../tests/test_benchmark_protocol.py).

Each run asserts prediction-order invariance and, for the reservoir models,
single-sample versus batch feature invariance. Recorded timings include those
checks and must not be advertised as a throughput benchmark.

## Integration status

`pixelflow.sklearn.ReservoirTransformer` is maintained by this project and has
tests for real Pipeline, GridSearchCV, cloning and serialization. It is not
an upstream scikit-learn feature or an endorsement by its maintainers.

The full non-slow suite passed locally: 76 tests, including CUDA and OpenGL
checks. GPU correctness tests are separate from this CPU accuracy evaluation.
The slow external-dataset test and historical full MNIST/CIFAR experiments
were not rerun in this evaluation.
