"""TSMixup — paper-faithful implementation of Chronos Algorithm 1.

Algorithm 1 (Ansari et al. 2024, Appendix A):

    Input: time series datasets {X_1, ..., X_{N_d}},
           K = 3                  max time series to mix
           α = 1.5                symmetric Dirichlet concentration
           l_min = 128, l_max = 2048

    1: k ~ U{1, K}                                   number of series to mix
    2: l ~ U{l_min, l_max}                           length of augmented series
    3: for i ∈ 1..k do
    4:     n ~ U{1, N_d}                             sample dataset index
    5:     x^(i)_{1:l} ~ X_n                          sample series of length l from dataset n
    6:     x_tilde^(i) ← x^(i) / (1/l Σ_j |x^(i)_j|)  mean-scale (divide by mean abs value)
    7: end for
    8: [λ_1,...,λ_k] ~ Dir(α,...,α)                   sample mixing weights
    9: return Σ_i λ_i · x_tilde^(i)

10M TSMixup augmentations were generated for the original Chronos training.
With K=3 and a single source dataset, an "original" series ends up sampled
with probability 1/3 (k=1 case).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np


# A "source" is a callable: rng, length -> 1-D series of that length.
# This generic interface lets the same TSMixup implementation work over:
#   - in-memory arrays
#   - parquet-on-disk datasets (lazy)
#   - mixed real + synthetic pools
SeriesSource = Callable[[np.random.Generator, int], np.ndarray]


@dataclass
class TSMixupSample:
    """One TSMixup draw with provenance."""
    series: np.ndarray            # shape (l,)
    k: int
    weights: np.ndarray           # shape (k,)
    source_indices: list[int]     # which dataset index was picked at each step


def _mean_scale(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Divide by the mean absolute value (paper's "mean scaling")."""
    s = np.mean(np.abs(x))
    return x / max(s, eps)


def tsmixup(
    sources: Sequence[SeriesSource],
    *,
    K: int = 3,
    alpha: float = 1.5,
    l_min: int = 128,
    l_max: int = 2048,
    rng: np.random.Generator | None = None,
    return_provenance: bool = False,
) -> np.ndarray | TSMixupSample:
    """Generate one TSMixup-augmented series.

    Parameters
    ----------
    sources : sequence of SeriesSource callables
        Each callable takes (rng, length) and returns a 1-D array of that length.
        Typically you'd pass one source per real dataset; the algorithm samples
        N_d = len(sources) and picks one uniformly per mix step.
    K : max number of series to mix per augmentation (paper K=3)
    alpha : Dirichlet concentration (paper α=1.5)
    l_min, l_max : sampled length range (paper [128, 2048])
    rng : numpy.random.Generator
    return_provenance : bool
        If True, return a TSMixupSample wrapping the array + provenance.

    Returns
    -------
    np.ndarray of shape (l,) for some l ∈ [l_min, l_max]
        OR a TSMixupSample wrapping the same.
    """
    if rng is None:
        rng = np.random.default_rng()
    if not sources:
        raise ValueError("at least one source required")

    # Step 1: number of series to mix
    k = int(rng.integers(1, K + 1))

    # Step 2: augmented length
    l = int(rng.integers(l_min, l_max + 1))

    # Steps 3–7: sample k series, each from a uniformly-chosen source, and mean-scale
    parts: list[np.ndarray] = []
    src_idx_log: list[int] = []
    for _ in range(k):
        n = int(rng.integers(0, len(sources)))
        src_idx_log.append(n)
        x = sources[n](rng, l)
        if len(x) != l:
            raise ValueError(f"source {n} returned length {len(x)}, expected {l}")
        parts.append(_mean_scale(x))

    # Step 8: weights ~ Dir(α, ..., α)
    weights = rng.dirichlet([alpha] * k)

    # Step 9: convex combination
    out = np.zeros(l, dtype=np.float64)
    for w, p in zip(weights, parts):
        out += w * p

    if return_provenance:
        return TSMixupSample(series=out, k=k, weights=weights,
                             source_indices=src_idx_log)
    return out


def generate_tsmixup_dataset(
    sources: Sequence[SeriesSource],
    n_series: int,
    *,
    K: int = 3,
    alpha: float = 1.5,
    l_min: int = 128,
    l_max: int = 2048,
    seed: int | None = 0,
) -> list[np.ndarray]:
    """Generate a batch of TSMixup-augmented series.

    Returned series have variable length in [l_min, l_max] — they are NOT
    stacked into a 2-D array because lengths differ. Caller chooses storage.
    """
    rng = np.random.default_rng(seed)
    return [
        tsmixup(sources, K=K, alpha=alpha, l_min=l_min, l_max=l_max, rng=rng)
        for _ in range(n_series)
    ]


# --- helpers to wrap common data shapes as SeriesSource callables -----------

def array_source(arr: np.ndarray) -> SeriesSource:
    """Wrap a 2-D (n_series, length) array OR a 1-D series as a SeriesSource.

    The source samples a random length-l contiguous window from a uniformly-
    chosen series in the array. If the underlying series is shorter than l,
    it is tiled to length l (matches the spirit of "sample of length l from
    dataset n" when the underlying series is short).
    """
    arr = np.asarray(arr)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    n, length = arr.shape

    def src(rng: np.random.Generator, l: int) -> np.ndarray:
        i = int(rng.integers(0, n))
        x = arr[i]
        if length >= l:
            start = int(rng.integers(0, length - l + 1))
            return x[start:start + l].astype(np.float64)
        # too short — tile
        reps = (l + length - 1) // length
        return np.tile(x, reps)[:l].astype(np.float64)
    return src


def parquet_column_source(parquet_path, column: str) -> SeriesSource:
    """Wrap a single column of a parquet file as a SeriesSource.

    The full column is loaded once into memory on first call; subsequent calls
    sample windows from the cached array. Suitable for parquet files of up to
    a few hundred MB; use a sharded loader for larger.
    """
    import pandas as pd
    cache: dict[str, np.ndarray] = {}

    def _load() -> np.ndarray:
        if "arr" not in cache:
            cache["arr"] = pd.read_parquet(parquet_path, columns=[column])[column].to_numpy(np.float64)
        return cache["arr"]

    def src(rng: np.random.Generator, l: int) -> np.ndarray:
        x = _load()
        if len(x) >= l:
            start = int(rng.integers(0, len(x) - l + 1))
            return x[start:start + l]
        reps = (l + len(x) - 1) // len(x)
        return np.tile(x, reps)[:l]
    return src
