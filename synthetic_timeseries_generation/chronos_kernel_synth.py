"""KernelSynth — paper-faithful implementation of Chronos Algorithm 2.

Algorithm 2 (Ansari et al. 2024, Appendix A):
    Input: kernel bank K, max kernels per series J = 5, length l_syn = 1024
    Output: synthetic time series x_{1:l_syn}

    1: j ~ U{1, J}                                       sample number of kernels
    2: {κ_1, ..., κ_j} iid~ K                            sample j kernels with replacement
    3: κ* ← κ_1
    4: for i ∈ 2..j do
    5:     ★ ~ {+, ×}                                    random binary operator
    6:     κ* ← κ* ★ κ_i                                 compose
    7: end for
    8: x_{1:l_syn} ~ GP(0, κ*(t,t'))                     sample from the GP prior
    9: return x_{1:l_syn}

Total of 1M synthetic series in the original paper. Each series is univariate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .chronos_kernels import (
    KernelSpec,
    add_kernels,
    build_kernel_bank,
    mul_kernels,
)


@dataclass
class KernelSynthSample:
    """A single KernelSynth draw with provenance for debugging / analysis."""
    series: np.ndarray            # shape (l_syn,)
    kernel_specs: list[KernelSpec]
    operators: list[str]          # length len(kernel_specs) - 1; each "+" or "×"

    @property
    def composition_str(self) -> str:
        """Human-readable composition like 'RBF(l=1) × Periodic(p=24) + Linear(σ=10)'."""
        parts = [f"{self.kernel_specs[0].name}({self._fmt(self.kernel_specs[0].params)})"]
        for op, ks in zip(self.operators, self.kernel_specs[1:]):
            parts.append(f" {op} {ks.name}({self._fmt(ks.params)})")
        return "".join(parts)

    @staticmethod
    def _fmt(params: dict) -> str:
        return ",".join(f"{k}={v}" for k, v in params.items())


def sample_gp_prior(
    kernel_fn,
    l_syn: int,
    rng: np.random.Generator,
    jitter: float = 1e-6,
) -> np.ndarray:
    """Draw one sample x ~ GP(0, κ(t,t')) at integer time points t = 0..l_syn-1.

    Uses Cholesky factorisation of the covariance matrix + a small jitter for
    numerical stability. If Cholesky fails, jitter is grown until it succeeds.
    """
    t = np.arange(l_syn, dtype=np.float64).reshape(-1, 1)
    K = kernel_fn(t, t)
    # Symmetrise to absorb floating-point asymmetry
    K = 0.5 * (K + K.T)

    cur_jitter = jitter
    for _ in range(10):
        try:
            L = np.linalg.cholesky(K + cur_jitter * np.eye(l_syn))
            break
        except np.linalg.LinAlgError:
            cur_jitter *= 10.0
    else:
        # Final fallback: eigh decomposition (always succeeds)
        evals, evecs = np.linalg.eigh(K + 1e-3 * np.eye(l_syn))
        evals = np.clip(evals, 0.0, None)
        L = evecs * np.sqrt(evals)

    z = rng.standard_normal(l_syn)
    return L @ z


def kernel_synth(
    *,
    kernel_bank: list[KernelSpec] | None = None,
    j_max: int = 5,
    l_syn: int = 1024,
    rng: np.random.Generator | None = None,
    return_provenance: bool = False,
) -> np.ndarray | KernelSynthSample:
    """Generate one synthetic series via Algorithm 2.

    Parameters
    ----------
    kernel_bank : list of KernelSpec, optional
        Defaults to the paper's Table 2 bank (build_kernel_bank()).
    j_max : int
        Maximum number of kernels to compose (paper J = 5).
    l_syn : int
        Length of the generated series (paper = 1024).
    rng : numpy.random.Generator, optional
        For reproducibility. Defaults to np.random.default_rng().
    return_provenance : bool
        If True, return a KernelSynthSample with the kernel choices recorded.

    Returns
    -------
    np.ndarray of shape (l_syn,)
        OR a KernelSynthSample wrapping the same array plus provenance.
    """
    if rng is None:
        rng = np.random.default_rng()
    if kernel_bank is None:
        kernel_bank = build_kernel_bank()

    # Step 1: number of kernels to sample
    j = int(rng.integers(1, j_max + 1))  # j ∈ {1, ..., J}

    # Step 2: j iid draws from K with replacement
    chosen_idx = rng.integers(0, len(kernel_bank), size=j)
    chosen = [kernel_bank[i] for i in chosen_idx]

    # Steps 3–7: random {+, ×} composition
    composed = chosen[0].fn
    operators: list[str] = []
    for next_spec in chosen[1:]:
        op = "+" if rng.random() < 0.5 else "×"
        operators.append(op)
        composed = (add_kernels if op == "+" else mul_kernels)(composed, next_spec.fn)

    # Step 8: x ~ GP(0, κ*(t,t'))
    series = sample_gp_prior(composed, l_syn=l_syn, rng=rng)

    if return_provenance:
        return KernelSynthSample(series=series, kernel_specs=chosen, operators=operators)
    return series


def generate_kernel_synth_dataset(
    n_series: int,
    *,
    kernel_bank: list[KernelSpec] | None = None,
    j_max: int = 5,
    l_syn: int = 1024,
    seed: int | None = 0,
    log_provenance: bool = False,
) -> np.ndarray | tuple[np.ndarray, list[KernelSynthSample]]:
    """Generate a batch of synthetic series via Algorithm 2.

    Returns
    -------
    np.ndarray of shape (n_series, l_syn)
        plus optional list of provenance records if log_provenance=True.
    """
    rng = np.random.default_rng(seed)
    if kernel_bank is None:
        kernel_bank = build_kernel_bank()

    out = np.empty((n_series, l_syn), dtype=np.float64)
    samples: list[KernelSynthSample] = []
    for i in range(n_series):
        if log_provenance:
            s = kernel_synth(kernel_bank=kernel_bank, j_max=j_max, l_syn=l_syn,
                             rng=rng, return_provenance=True)
            out[i] = s.series
            samples.append(s)
        else:
            out[i] = kernel_synth(kernel_bank=kernel_bank, j_max=j_max,
                                  l_syn=l_syn, rng=rng)
    if log_provenance:
        return out, samples
    return out
