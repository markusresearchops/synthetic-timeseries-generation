"""Tests for the OHLCV wrapper around KernelSynth output."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from synthetic_timeseries_generation.chronos_kernel_synth import (
    generate_kernel_synth_dataset,
)
from synthetic_timeseries_generation.ohlcv_adapter import (
    OHLCVConfig,
    batch_to_ohlcv,
    series_to_ohlcv,
)


def test_series_to_ohlcv_columns_and_length():
    rng = np.random.default_rng(0)
    series = rng.standard_normal(100)
    df = series_to_ohlcv(series, rng=rng)
    expected = {"date", "symbol", "open", "high", "low", "close",
                "average", "volume", "barCount"}
    assert set(df.columns) == expected
    assert len(df) == 100


def test_series_to_ohlcv_high_low_invariants():
    """high ≥ open, close, low; low ≤ open, close, high — by construction."""
    rng = np.random.default_rng(0)
    series = rng.standard_normal(200)
    df = series_to_ohlcv(series, rng=rng)
    assert (df["high"] >= df["open"]).all()
    assert (df["high"] >= df["close"]).all()
    assert (df["high"] >= df["low"]).all()
    assert (df["low"] <= df["open"]).all()
    assert (df["low"] <= df["close"]).all()


def test_series_to_ohlcv_starts_at_configured_price():
    rng = np.random.default_rng(0)
    series = np.zeros(20)  # zero log-return → flat price
    df = series_to_ohlcv(series, config=OHLCVConfig(start_price=250.0), rng=rng)
    # First open is exactly start_price; close path stays near it (zero log-return)
    assert df["open"].iloc[0] == 250.0
    assert (df["close"] - 250.0).abs().max() < 1e-9


def test_series_to_ohlcv_uses_log_return_scaling():
    rng = np.random.default_rng(0)
    # All-ones series: every bar's log-return is `return_scale * 1`
    series = np.ones(50)
    df = series_to_ohlcv(
        series, config=OHLCVConfig(start_price=100.0, return_scale=0.001),
        rng=rng,
    )
    # Close after n steps = 100 * exp(n * 0.001)
    expected_final = 100.0 * np.exp(50 * 0.001)
    assert abs(df["close"].iloc[-1] - expected_final) < 1e-9


def test_series_to_ohlcv_volume_is_positive_int():
    rng = np.random.default_rng(0)
    df = series_to_ohlcv(np.zeros(30), rng=rng)
    assert (df["volume"] >= 1).all()
    assert (df["barCount"] >= 1).all()
    # barCount is integer
    assert pd.api.types.is_integer_dtype(df["barCount"])


def test_series_to_ohlcv_dates_are_tz_aware_utc():
    rng = np.random.default_rng(0)
    df = series_to_ohlcv(np.zeros(5), start_date="2024-06-01 14:30",
                         freq="1min", rng=rng)
    assert df["date"].dt.tz is not None
    assert str(df["date"].dt.tz) == "UTC"
    # 1-min spacing
    deltas = df["date"].diff().dropna()
    assert (deltas == pd.Timedelta("1min")).all()


def test_batch_to_ohlcv_one_symbol_per_row():
    series_batch = np.random.default_rng(0).standard_normal((5, 30))
    df = batch_to_ohlcv(series_batch, symbol_prefix="X", seed=42)
    assert df["symbol"].nunique() == 5
    assert sorted(df["symbol"].unique()) == [f"X{i:04d}" for i in range(5)]
    assert len(df) == 5 * 30


def test_series_to_ohlcv_handles_extreme_gp_sample_without_zero_collapse():
    """A KernelSynth sample with huge variance must not drive prices to 0."""
    rng = np.random.default_rng(0)
    # Synthetic GP sample with extreme scale (e.g. Linear×Periodic kernel composition)
    huge = rng.standard_normal(2000) * 500.0
    df = series_to_ohlcv(huge, rng=rng, config=OHLCVConfig(start_price=100.0))
    # No price field should hit zero or go negative
    for col in ("open", "high", "low", "close", "average"):
        assert (df[col] > 0).all(), f"{col} hit zero/negative under extreme GP sample"


def test_series_to_ohlcv_handles_constant_input():
    """A degenerate (constant) input produces a flat price path, not an error."""
    rng = np.random.default_rng(0)
    df = series_to_ohlcv(np.zeros(100), rng=rng)
    # All zeros → flat close at start_price (default 100)
    assert (df["close"] > 0).all()
    assert (df["close"] - 100.0).abs().max() < 1e-9


def test_full_pipeline_kernelsynth_to_ohlcv():
    """End-to-end: KernelSynth → OHLCV → tokenized-forecaster-compatible parquet schema."""
    series = generate_kernel_synth_dataset(n_series=3, l_syn=64, seed=0)
    df = batch_to_ohlcv(series, symbol_prefix="SYN", seed=0)
    # Sanity: same expected columns the tokenized-forecaster pipeline reads
    required_cols = {"date", "symbol", "open", "high", "low", "close",
                     "average", "volume", "barCount"}
    assert required_cols.issubset(df.columns)
    # 3 symbols × 64 bars = 192 rows
    assert len(df) == 3 * 64
    # All bars finite
    for col in ("open", "high", "low", "close", "average", "volume", "barCount"):
        assert df[col].notna().all()
        assert np.isfinite(df[col]).all()
