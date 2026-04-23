"""Kernel bank for KernelSynth, faithful to Chronos paper (Ansari et al. 2024) Table 2.

Six kernels with the exact discrete hyperparameter sets:

    Constant        κ(x,x') = C,                     C = 1
    White Noise     κ(x,x') = σ_n · 1{x=x'},          σ_n ∈ {0.1, 1}
    Linear          κ(x,x') = σ² + x·x',              σ ∈ {0, 1, 10}
    RBF             κ(x,x') = exp(-||x-x'||² / 2l²),  l ∈ {0.1, 1, 10}
    RationalQuad    κ(x,x') = (1 + ||x-x'||²/2α)^-α,  α ∈ {0.1, 1, 10}
    Periodic        κ(x,x') = exp(-2 sin²(π||x-x'||/p) / l²)
                    p ∈ {24, 48, 96, 168, 336, 672, 7, 14, 30, 60, 365, 730,
                         4, 26, 52, 6, 12, 40, 10}
                    (Periodic uses l=1 implicit per the paper formula display)

These are designed to be used with KernelSynth (chronos_kernel_synth.py), which
samples J ≤ 5 kernels iid with replacement and composes them with random {+, ×}.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.spatial.distance import cdist


# Type alias: a Kernel is a callable (X1, X2) -> covariance matrix
KernelFn = Callable[[np.ndarray, np.ndarray], np.ndarray]


# --- the six basis kernels --------------------------------------------------

def constant_kernel(C: float = 1.0) -> KernelFn:
    """κ(x,x') = C — adds a constant offset to the GP."""
    def k(X1, X2):
        return np.full((len(X1), len(X2)), C, dtype=np.float64)
    return k


def white_noise_kernel(sigma_n: float) -> KernelFn:
    """κ(x,x') = σ_n · 1{x=x'} — i.i.d. noise; only diagonal when X1==X2."""
    def k(X1, X2):
        # Fast path: same array → identity-shaped diag
        if X1 is X2 or (X1.shape == X2.shape and np.array_equal(X1, X2)):
            return sigma_n * np.eye(len(X1), dtype=np.float64)
        # Otherwise mark exact equality element-wise
        d = cdist(X1.reshape(-1, 1), X2.reshape(-1, 1)) == 0.0
        return sigma_n * d.astype(np.float64)
    return k


def linear_kernel(sigma: float) -> KernelFn:
    """κ(x,x') = σ² + x·x' — non-stationary, gives linear trends."""
    def k(X1, X2):
        x1 = X1.reshape(-1, 1) if X1.ndim == 1 else X1
        x2 = X2.reshape(-1, 1) if X2.ndim == 1 else X2
        return sigma ** 2 + x1 @ x2.T
    return k


def rbf_kernel(lengthscale: float) -> KernelFn:
    """κ(x,x') = exp(-||x-x'||² / (2 l²)) — smooth local variation."""
    inv_2l2 = 1.0 / (2.0 * lengthscale ** 2)
    def k(X1, X2):
        x1 = X1.reshape(-1, 1) if X1.ndim == 1 else X1
        x2 = X2.reshape(-1, 1) if X2.ndim == 1 else X2
        d2 = cdist(x1, x2, metric="sqeuclidean")
        return np.exp(-d2 * inv_2l2)
    return k


def rational_quadratic_kernel(alpha: float, lengthscale: float = 1.0) -> KernelFn:
    """κ(x,x') = (1 + ||x-x'||² / (2 α l²))^(-α).

    The paper's table omits l (uses l=1 implicit). We expose it for flexibility
    but default to 1, matching the paper.
    """
    def k(X1, X2):
        x1 = X1.reshape(-1, 1) if X1.ndim == 1 else X1
        x2 = X2.reshape(-1, 1) if X2.ndim == 1 else X2
        d2 = cdist(x1, x2, metric="sqeuclidean")
        return (1.0 + d2 / (2.0 * alpha * lengthscale ** 2)) ** (-alpha)
    return k


def periodic_kernel(period: float, lengthscale: float = 1.0) -> KernelFn:
    """κ(x,x') = exp(-2 sin²(π ||x-x'|| / p) / l²) — captures seasonality of period p.

    Per the paper's table, only `period` varies; lengthscale is fixed at 1.
    """
    inv_l2 = 1.0 / (lengthscale ** 2)
    def k(X1, X2):
        x1 = X1.reshape(-1, 1) if X1.ndim == 1 else X1
        x2 = X2.reshape(-1, 1) if X2.ndim == 1 else X2
        d = cdist(x1, x2, metric="euclidean")
        return np.exp(-2.0 * np.sin(np.pi * d / period) ** 2 * inv_l2)
    return k


# --- kernel bank (paper Table 2) -------------------------------------------

@dataclass(frozen=True)
class KernelSpec:
    """A discrete kernel-template + hyperparameter draw, identifiable for logging."""
    name: str
    params: dict
    fn: KernelFn

    def __call__(self, X1, X2):
        return self.fn(X1, X2)


def build_kernel_bank() -> list[KernelSpec]:
    """Return the full kernel bank K from Chronos paper Table 2.

    Each (kernel-template, hyperparameter-value) pair becomes one entry in K.
    Total = 1 (Const) + 2 (WN) + 3 (Linear) + 3 (RBF) + 3 (RQ) + 19 (Periodic) = 31 entries.
    KernelSynth then samples j entries iid with replacement.
    """
    bank: list[KernelSpec] = []

    # Constant
    bank.append(KernelSpec("Const", {"C": 1}, constant_kernel(C=1.0)))

    # White Noise
    for sigma_n in (0.1, 1.0):
        bank.append(KernelSpec("WhiteNoise", {"sigma_n": sigma_n},
                               white_noise_kernel(sigma_n=sigma_n)))

    # Linear
    for sigma in (0.0, 1.0, 10.0):
        bank.append(KernelSpec("Linear", {"sigma": sigma}, linear_kernel(sigma=sigma)))

    # RBF
    for l in (0.1, 1.0, 10.0):
        bank.append(KernelSpec("RBF", {"l": l}, rbf_kernel(lengthscale=l)))

    # Rational Quadratic
    for alpha in (0.1, 1.0, 10.0):
        bank.append(KernelSpec("RationalQuadratic", {"alpha": alpha},
                               rational_quadratic_kernel(alpha=alpha)))

    # Periodic — exact period set from paper Table 2
    periods = (24, 48, 96, 168, 336, 672,
               7, 14, 30, 60, 365, 730,
               4, 26, 52, 6, 12, 40, 10)
    for p in periods:
        bank.append(KernelSpec("Periodic", {"p": p}, periodic_kernel(period=float(p))))

    return bank


# --- helpers for composition ------------------------------------------------

def add_kernels(k1: KernelFn, k2: KernelFn) -> KernelFn:
    def k(X1, X2):
        return k1(X1, X2) + k2(X1, X2)
    return k


def mul_kernels(k1: KernelFn, k2: KernelFn) -> KernelFn:
    def k(X1, X2):
        return k1(X1, X2) * k2(X1, X2)
    return k
