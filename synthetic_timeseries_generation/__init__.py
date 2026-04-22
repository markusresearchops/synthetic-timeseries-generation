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

__all__ = [
    "RBFKernel",
    "MaternKernel",
    "ExpSineSquaredKernel",
    "compose_kernel",
    "GaussianProcess",
    "generate_synthetic_paths",
    "generate_geometric_brownian_motion",
    "generate_mean_reverting_process",
    "SyntheticDatasetGenerator",
]
