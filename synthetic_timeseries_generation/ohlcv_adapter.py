"""Convert KernelSynth-style univariate series → OHLCV bars for our pipeline.

The Chronos paper produces *univariate* synthetic series. Our downstream
tokeniser (`tokenized-forecaster`) expects OHLCV with columns
`open, high, low, close, average, volume, barCount` plus `date` and `symbol`.

This module provides a minimal, transparent adapter that:
  1. Treats a KernelSynth length-l_syn sample as a (zero-mean-ish) log-return
     series after a bounded scaling step.
  2. Produces a price path via `exp(cumsum(.))`, anchored at a configurable
     starting price.
  3. Synthesises OHL/V/barCount around the close path with simple bar-local
     noise (open ≈ previous close, high ≥ max(open, close), low ≤ min(...),
     volume drawn from a lognormal).

This is **our extension** to the paper, not part of Algorithm 2. The Chronos
paper itself stops at the univariate series; the OHLCV synthesis here is a
practical wrapper for our equity-bar workflow.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class OHLCVConfig:
    """Hyperparameters for the OHLCV wrapper."""
    start_price: float = 100.0
    return_scale: float = 0.001       # multiply raw GP sample by this before cumsum (≈ 1m std on equity)
    intrabar_noise: float = 0.0005    # std of OHL fluctuation around the close path (relative)
    volume_log_mean: float = 13.5     # ≈ ln(700_000); typical 1-min equity volume
    volume_log_std: float = 1.0       # cross-bar volume variability
    bars_per_minute_mean: int = 250   # mean trade count per bar
    bars_per_minute_std: int = 80


def series_to_ohlcv(
    series: np.ndarray,
    *,
    symbol: str = "SYN0000",
    start_date: str = "2024-01-02 14:30",
    freq: str = "1min",
    config: OHLCVConfig = OHLCVConfig(),
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Wrap a 1-D KernelSynth output as a synthetic OHLCV bar dataframe.

    Parameters
    ----------
    series : np.ndarray of shape (n,)
        A KernelSynth sample (zero-mean-ish, unit-stdev-ish on average).
    symbol : str
        Symbol name to assign to the synthetic series.
    start_date : str
        Wall-clock timestamp of the first bar (UTC).
    freq : str
        Pandas freq string (default "1min").
    config : OHLCVConfig
        Bar-local noise / volume / trade-count parameters.
    rng : numpy.random.Generator
        For reproducibility.

    Returns
    -------
    pandas.DataFrame with columns:
        date, symbol, open, high, low, close, average, volume, barCount
    Compatible with `tokenized-forecaster` consolidate-bars input format.
    """
    if rng is None:
        rng = np.random.default_rng()

    n = len(series)
    log_returns = config.return_scale * series
    # Cumulate to a price path, anchored at start_price
    close = config.start_price * np.exp(np.cumsum(log_returns))

    # Open: previous close + small noise; first bar opens at the start price
    open_ = np.empty(n)
    open_[0] = config.start_price
    open_[1:] = close[:-1] * (1.0 + rng.normal(0, config.intrabar_noise, n - 1))

    # Intra-bar high / low: extend beyond max(open, close) by a positive noise
    upper = np.maximum(open_, close)
    lower = np.minimum(open_, close)
    high_extension = np.abs(rng.normal(0, config.intrabar_noise, n)) * close
    low_extension = np.abs(rng.normal(0, config.intrabar_noise, n)) * close
    high = upper + high_extension
    low = lower - low_extension

    # Average: VWAP-like; sit between low and high biased toward close
    weights = rng.uniform(0.3, 0.7, n)
    average = weights * close + (1 - weights) * (high + low) * 0.5

    # Volume: lognormal cross-bar; correlate softly with absolute return magnitude
    base_volume = rng.lognormal(config.volume_log_mean, config.volume_log_std, n)
    vol_modulator = 1.0 + 2.0 * np.abs(log_returns) / max(np.std(log_returns), 1e-9)
    volume = (base_volume * vol_modulator).astype(np.int64)
    volume = np.maximum(volume, 1)

    # barCount: Gaussian-ish around mean
    bar_count = rng.normal(config.bars_per_minute_mean,
                           config.bars_per_minute_std, n).clip(1).astype(np.int64)

    dates = pd.date_range(start_date, periods=n, freq=freq, tz="UTC")
    return pd.DataFrame({
        "date": dates,
        "symbol": symbol,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "average": average,
        "volume": volume.astype(float),
        "barCount": bar_count,
    })


def batch_to_ohlcv(
    series_batch: np.ndarray,
    *,
    symbol_prefix: str = "SYN",
    start_date: str = "2024-01-02 14:30",
    freq: str = "1min",
    config: OHLCVConfig = OHLCVConfig(),
    seed: int | None = 0,
) -> pd.DataFrame:
    """Wrap each row of a (n_paths, l_syn) batch as one synthetic symbol.

    Returns a single concatenated DataFrame (long form), one symbol per
    KernelSynth draw. Symbols are named `{prefix}{i:04d}`.
    """
    rng = np.random.default_rng(seed)
    parts = []
    for i, ser in enumerate(series_batch):
        sub = series_to_ohlcv(
            ser,
            symbol=f"{symbol_prefix}{i:04d}",
            start_date=start_date,
            freq=freq,
            config=config,
            rng=rng,
        )
        parts.append(sub)
    return pd.concat(parts, ignore_index=True)
