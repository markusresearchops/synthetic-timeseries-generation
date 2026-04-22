"""Tests for GP kernels."""

import numpy as np
import pytest

from synthetic_timeseries_generation.gp_kernels import (
    RBFKernel, MaternKernel, ExpSineSquaredKernel, ScaledKernel, ComposedKernel,
)


class TestRBFKernel:
    def test_rbf_properties(self):
        kernel = RBFKernel(variance=2.0, lengthscale=1.0)

        X = np.array([[0], [1], [2]])
        K = kernel(X, X)

        # Check symmetry
        assert np.allclose(K, K.T)

        # Check positive semi-definite
        eigvals = np.linalg.eigvals(K)
        assert np.all(eigvals >= -1e-10)

        # Check diagonal is variance
        assert np.allclose(np.diag(K), 2.0)

    def test_rbf_distance_decay(self):
        """RBF should decay with distance."""
        kernel = RBFKernel(variance=1.0, lengthscale=1.0)

        X_close = np.array([[0], [0.1]])
        X_far = np.array([[0], [10]])

        K_close = kernel(X_close[0:1], X_close[1:2])
        K_far = kernel(X_far[0:1], X_far[1:2])

        assert K_close[0, 0] > K_far[0, 0]


class TestMaternKernel:
    def test_matern_properties(self):
        kernel = MaternKernel(variance=1.0, lengthscale=1.0, nu=2.5)

        X = np.array([[0], [1], [2]])
        K = kernel(X, X)

        # Check symmetry
        assert np.allclose(K, K.T)

        # Check positive semi-definite
        eigvals = np.linalg.eigvals(K)
        assert np.all(eigvals >= -1e-10)


class TestPeriodicKernel:
    def test_periodic_properties(self):
        kernel = ExpSineSquaredKernel(variance=1.0, lengthscale=1.0, period=10.0)

        X = np.array([[0], [10], [20]])  # One period apart
        K = kernel(X, X)

        # Check symmetry
        assert np.allclose(K, K.T)

        # Points one period apart should be similar
        assert K[0, 1] > 0.5


class TestKernelComposition:
    def test_kernel_addition(self):
        kernel1 = RBFKernel(variance=1.0)
        kernel2 = RBFKernel(variance=1.0)
        kernel_sum = kernel1 + kernel2

        X = np.array([[0], [1]])
        K_sum = kernel_sum(X, X)
        K1 = kernel1(X, X)
        K2 = kernel2(X, X)

        assert np.allclose(K_sum, K1 + K2)

    def test_scaled_kernel(self):
        kernel = RBFKernel(variance=1.0)
        scaled = 2.0 * kernel

        X = np.array([[0], [1]])
        K_scaled = scaled(X, X)
        K_orig = kernel(X, X)

        assert np.allclose(K_scaled, 2.0 * K_orig)
