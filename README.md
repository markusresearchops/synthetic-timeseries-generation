# synthetic-timeseries-generation

Paper-faithful implementation of **KernelSynth** (Algorithm 2) and **TSMixup** (Algorithm 1) from:

> Ansari et al. 2024. *Chronos: Learning the Language of Time Series.* TMLR.
> https://openreview.net/forum?id=gerNCVqqtR

Companion to [`tokenized-forecaster`](https://github.com/markusresearchops/tokenized-forecaster) — the OHLCV adapter wraps KernelSynth output so synthetic bars can be fed directly into that pipeline's `consolidate-bars` / `pipeline-mh` stages.

## What's implemented

### KernelSynth (Algorithm 2)
Synthetic time series via Gaussian-process priors composed from a discrete kernel bank:

- 31-entry kernel bank (paper Table 2): Constant, WhiteNoise (×2), Linear (×3), RBF (×3), RationalQuadratic (×3), Periodic (×19)
- Algorithm 2 verbatim: sample `j ~ U{1,5}` kernels iid with replacement, compose via random `{+, ×}`, sample from the GP prior at length `l_syn = 1024`
- Optional provenance logging (kernel names, hyperparameters, operators) for every draw

### TSMixup (Algorithm 1)
Convex combinations of real time series for diverse augmentation:

- `k ~ U{1, K=3}` series mixed per augmentation
- Length `l ~ U{l_min=128, l_max=2048}`
- Mean-scaled before mixing
- Mixing weights `~ Dir(α=1.5)`
- Pluggable source interface (`SeriesSource = (rng, length) → array`); helpers for in-memory arrays and parquet columns

### OHLCV adapter (our extension)
The Chronos paper produces univariate series. For our equity-bar pipeline, we wrap the output as OHLCV bars:

- `series_to_ohlcv(series)` — exp-cumsum to get a price path, build OHL/V/barCount around the close path with bounded per-bar noise
- Output schema matches what `tokenized-forecaster` expects: `date, symbol, open, high, low, close, average, volume, barCount`

Not part of the paper — clearly marked as our extension.

## Install

```bash
pip install -e ".[test]"
```

## Use as a library

```python
from synthetic_timeseries_generation import (
    kernel_synth, generate_kernel_synth_dataset,
    tsmixup, generate_tsmixup_dataset, parquet_column_source,
    series_to_ohlcv, batch_to_ohlcv,
)
import numpy as np

# 1. KernelSynth — single series with provenance
rng = np.random.default_rng(0)
sample = kernel_synth(rng=rng, return_provenance=True)
print(sample.composition_str)   # e.g. "RBF(l=1) × Periodic(p=24) + Linear(σ=10)"
print(sample.series.shape)      # (1024,)

# 2. KernelSynth — batch of 100 univariate series
arr = generate_kernel_synth_dataset(n_series=100, l_syn=1024, seed=0)
# arr.shape == (100, 1024)

# 3. Wrap as OHLCV bars (our adapter, not in paper)
ohlcv_df = batch_to_ohlcv(arr, symbol_prefix="SYN", seed=0)
ohlcv_df.to_parquet("synthetic_year=2024.parquet", index=False)

# 4. TSMixup over real data
sources = [parquet_column_source("aapl_year=2024.parquet", column="close")]
augmented = generate_tsmixup_dataset(sources, n_series=10_000, seed=0)
# list of variable-length numpy arrays in [128, 2048]
```

## Use the CLI

```bash
# Generate 1000 KernelSynth series as OHLCV bars (drop into tokenized-forecaster)
chronos-synth kernelsynth --n 1000 --output-dir ./synth_data --year 2024

# Generate 10000 TSMixup augmentations from a real parquet column
chronos-synth tsmixup --source data/AAPL.parquet --column close --n 10000 \
                     --output ./tsmixup.parquet
```

## Layout

```
synthetic_timeseries_generation/
  chronos_kernels.py        # paper Table 2 kernel bank
  chronos_kernel_synth.py   # Algorithm 2 (KernelSynth)
  tsmixup.py                # Algorithm 1 (TSMixup)
  ohlcv_adapter.py          # univariate → OHLCV wrapper (our extension)
  cli.py                    # CLI: chronos-synth subcommands
tests/
  test_chronos_kernels.py
  test_chronos_kernel_synth.py
  test_tsmixup.py
  test_ohlcv_adapter.py
docs/
  paper_audit.md            # comparison vs the paper, including history of cleanup
```

## Tests

```bash
pytest                       # all 38 tests
```

## Why no PyTorch?

KernelSynth and TSMixup are **data-generation scripts**. They produce numpy arrays / parquet files that a downstream model trainer consumes. The paper's own implementation is plain NumPy/SciPy. PyTorch belongs in the model code — see `tokenized-forecaster` for that side of the work.

## History

The initial commit (`8765092`) was a partial implementation built from an abstract-only summary of the Chronos paper — it was missing 4 of 6 paper kernels (notably Linear, which is what generates trend), used GP regression instead of GP prior sampling, and lacked TSMixup entirely. Commit `86ff4d3` added the paper-faithful implementation alongside the legacy code; this commit removes the superseded legacy modules, leaving only the paper-faithful path.

See `docs/paper_audit.md` for the original line-by-line audit.
