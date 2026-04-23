"""Paper-faithful KernelSynth + TSMixup for synthetic time series generation.

Implements Algorithms 1 (TSMixup) and 2 (KernelSynth) plus Table 2 (kernel bank)
from the Chronos paper (Ansari et al. 2024, TMLR — https://openreview.net/forum?id=gerNCVqqtR).

Public surface:
  • build_kernel_bank()                 — paper Table 2 (31 entries)
  • kernel_synth(rng=...)               — Algorithm 2 (single series)
  • generate_kernel_synth_dataset(n)    — batched Algorithm 2
  • tsmixup(sources, rng=...)           — Algorithm 1 (single augmentation)
  • generate_tsmixup_dataset(srcs, n)   — batched Algorithm 1
  • series_to_ohlcv(series)             — wrap univariate output as OHLCV bars
                                          (our extension; not from the paper)
  • CLI entry point: chronos-synth      — kernelsynth | tsmixup subcommands
"""

__version__ = "0.2.0"

from .chronos_kernels import (
    KernelSpec,
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
from .chronos_kernel_synth import (
    KernelSynthSample,
    generate_kernel_synth_dataset,
    kernel_synth,
    sample_gp_prior,
)
from .ohlcv_adapter import (
    OHLCVConfig,
    batch_to_ohlcv,
    series_to_ohlcv,
)
from .tsmixup import (
    TSMixupSample,
    array_source,
    generate_tsmixup_dataset,
    parquet_column_source,
    tsmixup,
)

__all__ = [
    # Kernel bank (paper Table 2)
    "KernelSpec",
    "build_kernel_bank",
    "constant_kernel", "white_noise_kernel", "linear_kernel",
    "rbf_kernel", "rational_quadratic_kernel", "periodic_kernel",
    "add_kernels", "mul_kernels",
    # KernelSynth (Algorithm 2)
    "KernelSynthSample",
    "kernel_synth",
    "generate_kernel_synth_dataset",
    "sample_gp_prior",
    # TSMixup (Algorithm 1)
    "TSMixupSample",
    "tsmixup",
    "generate_tsmixup_dataset",
    "array_source",
    "parquet_column_source",
    # OHLCV adapter (our extension for the tokenized-forecaster pipeline)
    "OHLCVConfig",
    "series_to_ohlcv",
    "batch_to_ohlcv",
]
