"""Unit tests for pixelflow.tasks."""

import numpy as np
import pytest

from pixelflow.tasks.synthetic import two_moons, checkerboard
from pixelflow.tasks.eikonal import solve_reference


class TestTwoMoons:
    def test_shape(self):
        X, y = two_moons(n=100, noise=0.1, seed=0)
        assert X.shape == (100, 2)
        assert y.shape == (100,)

    def test_labels(self):
        _, y = two_moons(n=200, noise=0.1, seed=0)
        assert set(y).issubset({0, 1})
        # both classes present
        assert 0 in y and 1 in y

    def test_dtype(self):
        X, y = two_moons(n=50, noise=0.05, seed=1)
        assert X.dtype == np.float32
        assert np.issubdtype(y.dtype, np.integer)

    def test_reproducible(self):
        X1, y1 = two_moons(n=100, noise=0.1, seed=7)
        X2, y2 = two_moons(n=100, noise=0.1, seed=7)
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)


class TestCheckerboard:
    def test_shape(self):
        X, y = checkerboard(n_per_side=4, seed=0)
        n_cells = 4 * 4
        assert X.shape == (n_cells * 100, 2)
        assert y.shape == (n_cells * 100,)

    def test_labels_binary(self):
        _, y = checkerboard(n_per_side=3, seed=0)
        assert set(y).issubset({0, 1})


class TestEikonalSolver:
    def test_constant_speed_vs_euclidean(self):
        """On a constant speed=1 grid the arrival time must equal Euclidean distance."""
        H, W = 51, 51
        grid = np.ones((H, W), dtype=np.float64)
        src = (H // 2, W // 2)
        T = solve_reference(grid, src)

        rows = np.arange(H)
        cols = np.arange(W)
        rr, cc = np.meshgrid(rows, cols, indexing="ij")
        euclidean = np.sqrt((rr - src[0]) ** 2 + (cc - src[1]) ** 2)

        # First-order FMM accuracy: the mean relative error across all grid
        # points (excluding the source) should be < 5% on a constant-speed grid.
        # The worst-case max error (~20%) occurs only at immediate diagonal
        # neighbors of the source due to the 1st-order upwind stencil; this is
        # a known property of 1st-order FMM and does not affect global accuracy.
        mask = euclidean > 0
        rel_err = np.abs(T[mask] - euclidean[mask]) / euclidean[mask]
        mean_rel = rel_err.mean()
        assert mean_rel < 0.05, (
            f"mean relative error {mean_rel:.4f} exceeds 5% "
            f"(tested {mask.sum()} grid points)"
        )

    def test_source_is_zero(self):
        grid = np.ones((20, 20))
        T = solve_reference(grid, (5, 10))
        assert T[5, 10] == 0.0

    def test_higher_speed_shorter_time(self):
        """Doubling speed should halve arrival times (far from source)."""
        H, W = 31, 31
        src = (0, 0)
        T1 = solve_reference(np.ones((H, W)), src)
        T2 = solve_reference(2.0 * np.ones((H, W)), src)
        # T2 should be approximately T1 / 2
        interior = T1 > 1.0
        ratio = T1[interior] / T2[interior]
        assert np.allclose(ratio, 2.0, atol=0.15), f"speed ratio off: {ratio.mean():.3f}"

    def test_invalid_grid_raises(self):
        with pytest.raises(ValueError):
            solve_reference(np.zeros((10, 10)), (5, 5))

    def test_out_of_bounds_source_raises(self):
        with pytest.raises(ValueError):
            solve_reference(np.ones((10, 10)), (15, 5))

    def test_non_2d_grid_raises(self):
        with pytest.raises(ValueError):
            solve_reference(np.ones((10, 10, 3)), (5, 5))


@pytest.mark.slow
def test_mnist_load_shapes():
    """MNIST loader: check shapes and value range. Marked slow — skipped by default."""
    pytest.importorskip("sklearn.datasets")
    from pixelflow.tasks.mnist import load

    X_train, y_train, X_test, y_test = load(subset=500, seed=0)
    assert X_train.shape == (500, 784)
    assert y_train.shape == (500,)
    assert X_test.shape[1] == 784
    assert X_train.dtype == np.float32
    assert X_train.min() >= 0.0 and X_train.max() <= 1.0
    assert set(y_train).issubset(set(range(10)))
