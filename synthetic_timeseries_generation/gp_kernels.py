"""
Gaussian Process kernels for time series generation.

Provides RBF, Matern, and periodic kernels suitable for financial time series.
"""

import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import gamma


class Kernel:
    """Base kernel class."""

    def __call__(self, X1, X2):
        """Compute kernel matrix K(X1, X2)."""
        raise NotImplementedError

    def __add__(self, other):
        """Kernel addition."""
        if isinstance(other, (int, float)) and other == 0:
            return self
        return ComposedKernel(self, other, "add")

    def __mul__(self, other):
        """Kernel multiplication (product)."""
        if isinstance(other, (int, float)):
            return ScaledKernel(self, other)
        return ComposedKernel(self, other, "mul")

    def __rmul__(self, other):
        return self.__mul__(other)


class RBFKernel(Kernel):
    """Radial Basis Function (squared exponential) kernel.

    k(x, x') = σ² exp(- ||x - x'||² / (2 ℓ²))

    Suitable for smooth, non-periodic patterns.
    """

    def __init__(self, variance=1.0, lengthscale=1.0):
        self.variance = variance
        self.lengthscale = lengthscale

    def __call__(self, X1, X2):
        """Compute RBF kernel matrix."""
        if X1.ndim == 1:
            X1 = X1.reshape(-1, 1)
        if X2.ndim == 1:
            X2 = X2.reshape(-1, 1)

        dist_sq = cdist(X1, X2, metric='sqeuclidean')
        return self.variance * np.exp(-dist_sq / (2 * self.lengthscale ** 2))


class MaternKernel(Kernel):
    """Matérn kernel: smoother version of RBF.

    k(x, x') = σ² (2^(1-ν) / Γ(ν)) * (√(2ν) * r / ℓ)^ν * K_ν(√(2ν) * r / ℓ)

    where r = ||x - x'|| and K_ν is the modified Bessel function.
    ν=3/2 or 5/2 are common choices for financial data (less smooth than RBF).
    """

    def __init__(self, variance=1.0, lengthscale=1.0, nu=5.0/2.0):
        self.variance = variance
        self.lengthscale = lengthscale
        self.nu = nu

    def __call__(self, X1, X2):
        """Compute Matérn kernel matrix."""
        if X1.ndim == 1:
            X1 = X1.reshape(-1, 1)
        if X2.ndim == 1:
            X2 = X2.reshape(-1, 1)

        dist = cdist(X1, X2, metric='euclidean')
        scaled_dist = np.sqrt(2 * self.nu) * dist / self.lengthscale

        # Avoid division by zero
        scaled_dist = np.clip(scaled_dist, 1e-10, None)

        # Modified Bessel function of second kind
        from scipy.special import kv
        bessel_term = kv(self.nu, scaled_dist)

        coeff = (2 ** (1 - self.nu)) / gamma(self.nu)
        return self.variance * coeff * (scaled_dist ** self.nu) * bessel_term


class ExpSineSquaredKernel(Kernel):
    """Periodic kernel (exponential sine squared).

    k(x, x') = σ² exp(- 2 sin²(π * ||x - x'|| / p) / ℓ²)

    Suitable for periodic patterns (hourly, daily seasonality in financial data).
    """

    def __init__(self, variance=1.0, lengthscale=1.0, period=1.0):
        self.variance = variance
        self.lengthscale = lengthscale
        self.period = period

    def __call__(self, X1, X2):
        """Compute periodic kernel matrix."""
        if X1.ndim == 1:
            X1 = X1.reshape(-1, 1)
        if X2.ndim == 1:
            X2 = X2.reshape(-1, 1)

        dist = cdist(X1, X2, metric='euclidean')
        sine_arg = np.pi * dist / self.period
        exponent = -2 * np.sin(sine_arg) ** 2 / (self.lengthscale ** 2)
        return self.variance * np.exp(exponent)


class ScaledKernel(Kernel):
    """Kernel scaled by a constant factor."""

    def __init__(self, kernel, scale):
        self.kernel = kernel
        self.scale = scale

    def __call__(self, X1, X2):
        return self.scale * self.kernel(X1, X2)


class ComposedKernel(Kernel):
    """Composition of two kernels (addition or multiplication)."""

    def __init__(self, kernel1, kernel2, operation="add"):
        self.kernel1 = kernel1
        self.kernel2 = kernel2
        self.operation = operation

    def __call__(self, X1, X2):
        if self.operation == "add":
            return self.kernel1(X1, X2) + self.kernel2(X1, X2)
        elif self.operation == "mul":
            return self.kernel1(X1, X2) * self.kernel2(X1, X2)
        else:
            raise ValueError(f"Unknown operation: {self.operation}")


def compose_kernel(kernels, operation="add"):
    """Compose multiple kernels.

    Args:
        kernels: list of Kernel objects
        operation: "add" or "mul"

    Returns:
        Composed kernel
    """
    if len(kernels) == 0:
        raise ValueError("At least one kernel required")

    result = kernels[0]
    for kernel in kernels[1:]:
        if operation == "add":
            result = result + kernel
        elif operation == "mul":
            result = result * kernel
        else:
            raise ValueError(f"Unknown operation: {operation}")

    return result
