"""Tests for the paper-faithful kernel bank (Chronos Table 2)."""

from __future__ import annotations

import numpy as np
import pytest

from synthetic_timeseries_generation.chronos_kernels import (
    add_kernels,
    build_kernel_bank,
    constant_kernel,
    linear_kernel,
    mul_kernels,
    periodic_kernel,
    rational_quadratic_kernel,
    rbf_kernel,
    white_noise_kernel,
)


def _grid(n=20):
    return np.arange(n, dtype=np.float64).reshape(-1, 1)


def test_kernel_bank_has_expected_count_and_types():
    K = build_kernel_bank()
    # 1 (Const) + 2 (WN) + 3 (Linear) + 3 (RBF) + 3 (RQ) + 19 (Periodic) = 31
    assert len(K) == 31
    counts = {}
    for spec in K:
        counts[spec.name] = counts.get(spec.name, 0) + 1
    assert counts["Const"] == 1
    assert counts["WhiteNoise"] == 2
    assert counts["Linear"] == 3
    assert counts["RBF"] == 3
    assert counts["RationalQuadratic"] == 3
    assert counts["Periodic"] == 19


def test_constant_kernel_is_constant():
    k = constant_kernel(C=2.5)
    K = k(_grid(), _grid())
    assert K.shape == (20, 20)
    assert np.allclose(K, 2.5)


def test_white_noise_is_diagonal_when_self():
    k = white_noise_kernel(sigma_n=0.7)
    K = k(_grid(), _grid())
    assert np.allclose(np.diag(K), 0.7)
    # Off-diagonal must be zero
    assert np.allclose(K - np.diag(np.diag(K)), 0.0)


def test_linear_kernel_is_correct_inner_product():
    k = linear_kernel(sigma=2.0)
    x = _grid(5)
    K = k(x, x)
    # K[i, j] = σ² + i*j
    expected = 4.0 + x @ x.T
    assert np.allclose(K, expected)


def test_rbf_kernel_correct_diagonal_and_decay():
    k = rbf_kernel(lengthscale=1.0)
    K = k(_grid(5), _grid(5))
    # Diagonal: ||x-x||² = 0 → 1
    assert np.allclose(np.diag(K), 1.0)
    # Strictly positive, monotone decay with distance
    assert (K > 0).all()


def test_rbf_lengthscale_controls_smoothness():
    K_short = rbf_kernel(lengthscale=0.1)(_grid(5), _grid(5))
    K_long = rbf_kernel(lengthscale=10.0)(_grid(5), _grid(5))
    # Long lengthscale → less decay between adjacent points
    assert K_long[0, 1] > K_short[0, 1]


def test_rational_quadratic_diagonal_unit():
    k = rational_quadratic_kernel(alpha=1.0)
    K = k(_grid(5), _grid(5))
    assert np.allclose(np.diag(K), 1.0)


def test_periodic_kernel_periodicity():
    k = periodic_kernel(period=4.0)
    x = np.array([0.0, 4.0]).reshape(-1, 1)  # one full period apart
    K = k(x, x)
    # k(0, 4) should equal k(0, 0) since they're one full period apart
    assert np.isclose(K[0, 1], K[0, 0])


def test_kernel_composition_addition():
    k1 = constant_kernel(C=1.0)
    k2 = constant_kernel(C=2.0)
    k_sum = add_kernels(k1, k2)
    K = k_sum(_grid(5), _grid(5))
    assert np.allclose(K, 3.0)


def test_kernel_composition_multiplication():
    k1 = constant_kernel(C=2.0)
    k2 = constant_kernel(C=3.0)
    k_mul = mul_kernels(k1, k2)
    K = k_mul(_grid(5), _grid(5))
    assert np.allclose(K, 6.0)


def test_kernel_bank_periods_match_paper_table_2():
    K = build_kernel_bank()
    periodic_periods = sorted(spec.params["p"] for spec in K if spec.name == "Periodic")
    expected = sorted([24, 48, 96, 168, 336, 672, 7, 14, 30, 60, 365, 730,
                       4, 26, 52, 6, 12, 40, 10])
    assert periodic_periods == expected


def test_all_bank_kernels_callable_on_unit_grid():
    K = build_kernel_bank()
    x = _grid(8)
    for spec in K:
        out = spec(x, x)
        assert out.shape == (8, 8)
        assert np.isfinite(out).all(), f"non-finite output from {spec.name} {spec.params}"
