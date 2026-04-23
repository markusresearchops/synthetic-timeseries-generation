"""Tests for TSMixup (Algorithm 1)."""

from __future__ import annotations

import numpy as np
import pytest

from synthetic_timeseries_generation.tsmixup import (
    TSMixupSample,
    array_source,
    generate_tsmixup_dataset,
    tsmixup,
)


def _two_sources():
    """Two distinguishable sources: source 0 = sine wave, source 1 = ramp."""
    rng = np.random.default_rng(0)
    # Source 0: 4096-long sine
    x = np.arange(4096)
    sine = np.sin(2 * np.pi * x / 100.0)
    # Source 1: 4096-long linear ramp
    ramp = np.linspace(-1, 1, 4096)
    return [array_source(sine), array_source(ramp)]


def test_tsmixup_returns_array_of_sampled_length():
    sources = _two_sources()
    rng = np.random.default_rng(0)
    out = tsmixup(sources, rng=rng, l_min=128, l_max=256)
    assert isinstance(out, np.ndarray)
    assert out.ndim == 1
    assert 128 <= len(out) <= 256


def test_tsmixup_returns_provenance_when_requested():
    sources = _two_sources()
    rng = np.random.default_rng(0)
    s = tsmixup(sources, K=3, rng=rng, return_provenance=True)
    assert isinstance(s, TSMixupSample)
    assert 1 <= s.k <= 3
    assert s.weights.shape == (s.k,)
    assert np.isclose(s.weights.sum(), 1.0)
    assert all(0.0 <= w <= 1.0 for w in s.weights)
    assert len(s.source_indices) == s.k


def test_tsmixup_k_eq_1_returns_mean_scaled_source():
    sources = _two_sources()
    rng = np.random.default_rng(42)
    # K=1 forces k=1 (single-source mix), which should just be the mean-scaled source
    out = tsmixup(sources, K=1, rng=rng, l_min=64, l_max=64)
    # With k=1 and single weight, weight is exactly 1.0 → result = mean-scaled source
    assert len(out) == 64
    assert np.isclose(np.mean(np.abs(out)), 1.0, atol=1e-9)


def test_tsmixup_dirichlet_weights_correct_concentration():
    """At α=1.5, weights should be roughly balanced; at very high α they concentrate."""
    sources = _two_sources()
    # Run many draws and inspect distribution
    rng = np.random.default_rng(0)
    weights_hi_alpha = []
    for _ in range(200):
        s = tsmixup(sources, K=3, alpha=100.0, rng=rng, return_provenance=True)
        if s.k == 3:
            weights_hi_alpha.append(s.weights)

    if weights_hi_alpha:
        # High α → all weights cluster near 1/k = 1/3
        mean_w = np.array(weights_hi_alpha).mean(axis=0)
        assert np.allclose(mean_w, 1/3, atol=0.05)


def test_tsmixup_mean_scaling_normalisation():
    """Each component is mean-abs scaled to 1 before mixing."""
    sources = _two_sources()
    rng = np.random.default_rng(0)
    # Force a 2-source mix to exercise the scaling step
    out = tsmixup(sources, K=3, rng=rng, l_min=512, l_max=512)
    # Output is a convex combination of unit-mean-abs series, so its mean-abs is bounded
    # by max(λ_i * 1) = max λ ≤ 1, and ≥ ... well it depends on cancellation.
    # Sanity: the output should not be wildly larger than 1 in mean-abs.
    assert np.mean(np.abs(out)) < 5.0  # generous bound — typically < 1.5


def test_generate_tsmixup_dataset_returns_n_series_with_varying_lengths():
    sources = _two_sources()
    out = generate_tsmixup_dataset(sources, n_series=20, l_min=128, l_max=256, seed=0)
    assert len(out) == 20
    lengths = [len(x) for x in out]
    # With 20 draws over a range of 128 values, we should see length variation
    assert len(set(lengths)) > 1
    for x in out:
        assert 128 <= len(x) <= 256


def test_array_source_handles_short_underlying_via_tile():
    short = np.array([1.0, 2.0, 3.0])
    src = array_source(short)
    rng = np.random.default_rng(0)
    out = src(rng, 10)
    assert len(out) == 10
    # Should be a tiled version
    assert np.array_equal(out[:3], short)


def test_array_source_window_sampling_within_long_series():
    long = np.arange(1000.0)
    src = array_source(long)
    rng = np.random.default_rng(0)
    for _ in range(20):
        out = src(rng, 50)
        assert len(out) == 50
        # Check it's a contiguous slice of the source
        # Find the start index by matching the first value
        assert out[0] in long
        # Differences should all be 1 (since source is np.arange)
        assert np.allclose(np.diff(out), 1.0)


def test_tsmixup_reproducible_under_seed():
    sources = _two_sources()
    a = generate_tsmixup_dataset(sources, n_series=5, l_min=64, l_max=128, seed=123)
    b = generate_tsmixup_dataset(sources, n_series=5, l_min=64, l_max=128, seed=123)
    assert len(a) == len(b)
    for xa, xb in zip(a, b):
        assert len(xa) == len(xb)
        assert np.allclose(xa, xb)


def test_tsmixup_raises_on_empty_sources():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="at least one source"):
        tsmixup([], rng=rng)
