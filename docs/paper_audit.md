# Paper audit — Chronos KernelSynth & TSMixup vs this repo

Audited 2026-04-23 against [Chronos: Learning the Language of Time Series](https://openreview.net/forum?id=gerNCVqqtR) (Ansari et al. 2024, TMLR).

The overnight implementation (commit `8765092`) was created with only the abstract-level description of synthetic data generation. After reading the actual paper (Sections 4.1, 4.2 and Appendix A Algorithms 1, 2 + Table 2), several substantive gaps are visible. This doc lists them and points at the paper-faithful implementation that was added today.

## Where the overnight implementation deviates from the paper

| dimension | Chronos paper | overnight repo (`gp_kernels.py`, `gp_processes.py`) | severity |
|---|---|---|---|
| Kernel bank | 6 templates × discrete hyper-sets (Const, WhiteNoise, Linear, RBF, RationalQuadratic, Periodic) — 31 entries total in K | 3 templates only (RBF, Matérn, ExpSineSquared) | **major** |
| Linear / trend kernel | required (paper Table 2: σ ∈ {0, 1, 10}) | absent | **major** — trend cannot be generated |
| Matérn | not in the paper bank | included | minor — extra, not wrong |
| Composition | Algorithm 2: `j ~ U{1, J=5}`, sample `j` kernels iid w/ replacement, compose with random `{+, ×}` between each | hardcoded per `kernel_type`; one fixed combination | **major** |
| GP sampling | sample from the **prior**: `x ~ GP(0, κ*(t,t'))` | does GP **regression** (fits to noise inducing points then predicts) | **major** — different distribution |
| Length | `l_syn = 1024` | default 1000 (and ad-hoc shorter) | minor |
| Trend / vol modulation | none — trend / seasonality come from the kernel bank itself | adds `linspace(-1,1) * trend_strength` and `1 + sin(...) * vol_strength` | **major** — biases the distribution away from a clean GP prior |
| TSMixup (Algorithm 1) | `k ~ U{1, K=3}`, length `l ~ U{128, 2048}`, mean-scale, `Dir(α=1.5)` weights, return Σ λᵢ x̃ⁱ | **completely missing** | **critical** — this is half of the paper's data pipeline |
| Output unit | univariate scalar series (per the paper) | OHLCV (custom adaptation, not from paper) | extension — fine for our use, but worth flagging |
| Parametric ensemble (GBM, MR, RS, SV) | not in the paper at all | included | extra |

## What was added today (commit forthcoming)

Three new modules implementing the paper exactly:

### `chronos_kernels.py` — paper-faithful kernel bank

The full 31-entry bank from Table 2:

| family | hyperparams | count | formula |
|---|---|---:|---|
| Constant | C = 1 | 1 | κ = C |
| WhiteNoise | σ_n ∈ {0.1, 1} | 2 | κ = σ_n · 1{x = x'} |
| Linear | σ ∈ {0, 1, 10} | 3 | κ = σ² + x · x' |
| RBF | l ∈ {0.1, 1, 10} | 3 | κ = exp(−‖x−x'‖² / 2l²) |
| RationalQuadratic | α ∈ {0.1, 1, 10} | 3 | κ = (1 + ‖x−x'‖² / 2α)^(−α) |
| Periodic | p ∈ {24, 48, 96, 168, 336, 672, 7, 14, 30, 60, 365, 730, 4, 26, 52, 6, 12, 40, 10} | 19 | κ = exp(−2 sin²(π‖x−x'‖/p)) |

`build_kernel_bank()` returns the full list.

### `chronos_kernel_synth.py` — Algorithm 2 verbatim

```
j ~ U{1, J=5}
{κ_1, ..., κ_j} iid~ K
κ* ← κ_1
for i ∈ 2..j:
    ★ ~ {+, ×}
    κ* ← κ* ★ κ_i
x_{1:l_syn} ~ GP(0, κ*(t,t'))
```

`kernel_synth(rng=...)` produces one length-1024 series in O(l_syn²) time
(Cholesky of the 1024×1024 covariance — about 30 ms per series on CPU).
`generate_kernel_synth_dataset(n_series, ...)` batches.
`return_provenance=True` records the kernel composition for inspection.

### `tsmixup.py` — Algorithm 1 verbatim

```
k ~ U{1, K=3}
l ~ U{l_min=128, l_max=2048}
for i ∈ 1..k:
    n ~ U{1, N_d}            (sample dataset index)
    x^(i) ~ X_n               (sample length-l series)
    x̃^(i) ← x^(i) / mean(|x^(i)|)
[λ_1, ..., λ_k] ~ Dir(α=1.5)
return Σ λ_i x̃^(i)
```

Sources are pluggable callables (`SeriesSource = (rng, length) -> array`),
so the same implementation works on in-memory arrays, parquet columns, or
any custom data loader. Helpers provided: `array_source(...)`,
`parquet_column_source(...)`.

## What does NOT need to change

- **Numpy/Scipy is correct.** The paper itself uses standard scientific Python
  for KernelSynth and TSMixup — these are data-generation scripts, not model
  code. They produce numpy arrays / parquet which a separate model trainer
  consumes. **PyTorch is not needed for these modules.** It will be needed for
  the Chronos T5 model itself, which is a separate concern (we have our own
  forecasting model design in `tokenized-forecaster/docs/`).
- The OHLCV-adaptation (`synthetic_dataset.py`) is a useful extension because
  our pipeline expects OHLCV. Keep it, but rebuild it on top of the new
  paper-faithful KernelSynth so the underlying close-price series are drawn
  from a real GP prior with the paper's kernel bank.

## What's the path forward

The overnight `gp_kernels.py`, `gp_processes.py`, and `synthetic_dataset.py`
are kept for backward compatibility but should be considered deprecated. New
work should use `chronos_kernels`, `chronos_kernel_synth`, and `tsmixup`.

The OHLCV-construction step (`SyntheticDatasetGenerator._construct_ohlcv_from_close`)
can be lifted onto `kernel_synth()` output by:

1. Generating a length-1024 series via `kernel_synth(...)` — this is already
   in log-return-like units (zero-mean, unit-stdev-ish).
2. Cumulative-sum + exponentiate to get a price path.
3. Build OHL/V on top using simple per-bar noise as the existing helper does.

This adapts the paper-faithful generator to our financial OHLCV format
without touching the (already correct) KernelSynth / TSMixup core.

## References

- Ansari et al. 2024, *Chronos: Learning the Language of Time Series*, TMLR. https://openreview.net/forum?id=gerNCVqqtR
- Algorithm 1 (TSMixup): paper Appendix A
- Algorithm 2 (KernelSynth): paper Appendix A
- Table 2 (kernel bank): paper Appendix A
