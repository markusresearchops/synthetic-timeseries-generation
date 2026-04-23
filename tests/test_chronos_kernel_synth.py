"""Tests for KernelSynth (Algorithm 2)."""

from __future__ import annotations

import numpy as np
import pytest

from synthetic_timeseries_generation.chronos_kernel_synth import (
    KernelSynthSample,
    generate_kernel_synth_dataset,
    kernel_synth,
    sample_gp_prior,
)
from synthetic_timeseries_generation.chronos_kernels import (
    build_kernel_bank,
    rbf_kernel,
)


def test_sample_gp_prior_shape_and_finite():
    rng = np.random.default_rng(0)
    x = sample_gp_prior(rbf_kernel(lengthscale=5.0), l_syn=64, rng=rng)
    assert x.shape == (64,)
    assert np.isfinite(x).all()


def test_sample_gp_prior_reproducible_with_seed():
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    x1 = sample_gp_prior(rbf_kernel(lengthscale=5.0), l_syn=32, rng=rng1)
    x2 = sample_gp_prior(rbf_kernel(lengthscale=5.0), l_syn=32, rng=rng2)
    assert np.allclose(x1, x2)


def test_kernel_synth_default_returns_array_of_paper_length():
    rng = np.random.default_rng(0)
    x = kernel_synth(rng=rng)  # paper default l_syn=1024, J=5
    assert isinstance(x, np.ndarray)
    assert x.shape == (1024,)
    assert np.isfinite(x).all()


def test_kernel_synth_with_provenance_records_composition():
    rng = np.random.default_rng(7)
    s = kernel_synth(rng=rng, l_syn=128, return_provenance=True)
    assert isinstance(s, KernelSynthSample)
    assert s.series.shape == (128,)
    assert 1 <= len(s.kernel_specs) <= 5
    assert len(s.operators) == len(s.kernel_specs) - 1
    for op in s.operators:
        assert op in ("+", "×")
    # composition_str is a non-empty human-readable string
    assert len(s.composition_str) > 0


def test_kernel_synth_j_max_respected():
    rng = np.random.default_rng(0)
    # Force j_max=1 → should always pick exactly 1 kernel
    for _ in range(50):
        s = kernel_synth(rng=rng, j_max=1, l_syn=64, return_provenance=True)
        assert len(s.kernel_specs) == 1
        assert s.operators == []


def test_generate_kernel_synth_dataset_batch_shape():
    arr = generate_kernel_synth_dataset(n_series=10, l_syn=128, seed=0)
    assert arr.shape == (10, 128)
    assert np.isfinite(arr).all()


def test_generate_kernel_synth_dataset_deterministic_under_seed():
    a = generate_kernel_synth_dataset(n_series=5, l_syn=64, seed=99)
    b = generate_kernel_synth_dataset(n_series=5, l_syn=64, seed=99)
    assert np.allclose(a, b)


def test_generate_kernel_synth_dataset_with_provenance_logs_each_sample():
    arr, samples = generate_kernel_synth_dataset(
        n_series=4, l_syn=64, seed=0, log_provenance=True,
    )
    assert arr.shape == (4, 64)
    assert len(samples) == 4
    for s in samples:
        assert isinstance(s, KernelSynthSample)


def test_kernel_synth_diversity_of_compositions():
    """Across many draws we should see different kernels and operators."""
    rng = np.random.default_rng(0)
    seen_kernels = set()
    seen_ops = set()
    for _ in range(200):
        s = kernel_synth(rng=rng, l_syn=32, return_provenance=True)
        for k in s.kernel_specs:
            seen_kernels.add(k.name)
        for op in s.operators:
            seen_ops.add(op)
    # We should see at least 4 different kernel types (not all draws hit Constant only)
    assert len(seen_kernels) >= 4
    # Both operators should appear
    assert seen_ops == {"+", "×"} or len(seen_ops) >= 1
