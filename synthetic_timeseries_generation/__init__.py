"""
Synthetic Time Series Generation via Gaussian Processes.

Inspired by Chronos (Ansari et al., 2024) synthetic data generation approach.
Generates realistic synthetic price paths using GPs to improve forecasting model generalization.
"""

__version__ = "0.1.0"

from .gp_kernels import RBFKernel, MaternKernel, ExpSineSquaredKernel, compose_kernel
from .gp_processes import GaussianProcess, generate_synthetic_paths
from .price_models import generate_geometric_brownian_motion, generate_mean_reverting_process
from .synthetic_dataset import SyntheticDatasetGenerator

# Paper-faithful Chronos modules (Ansari et al. 2024 Algorithms 1+2)
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
from .tsmixup import (
    TSMixupSample,
    array_source,
    generate_tsmixup_dataset,
    parquet_column_source,
    tsmixup,
)

__all__ = [
    # Legacy (overnight implementation; kept for back-compat)
    "RBFKernel",
    "MaternKernel",
    "ExpSineSquaredKernel",
    "compose_kernel",
    "GaussianProcess",
    "generate_synthetic_paths",
    "generate_geometric_brownian_motion",
    "generate_mean_reverting_process",
    "SyntheticDatasetGenerator",
    # Chronos paper-faithful
    "KernelSpec",
    "build_kernel_bank",
    "constant_kernel", "white_noise_kernel", "linear_kernel",
    "rbf_kernel", "rational_quadratic_kernel", "periodic_kernel",
    "add_kernels", "mul_kernels",
    "KernelSynthSample",
    "kernel_synth",
    "generate_kernel_synth_dataset",
    "sample_gp_prior",
    "TSMixupSample",
    "tsmixup",
    "generate_tsmixup_dataset",
    "array_source",
    "parquet_column_source",
]
