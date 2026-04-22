# Synthetic Time Series Generation

Gaussian Process-based synthetic OHLCV data generation for stock forecasting model training and regularization.

**Inspired by**: Chronos (Ansari et al., 2024) synthetic data generation approach. The paper demonstrates that pretraining on both real and synthetically-generated time series improves zero-shot generalization across unseen domains.

## Two Data Generation Methods

### 1. **Gaussian Process-based Generation** (Primary)

Samples from GP priors with various kernel configurations to generate diverse, realistic-looking price paths. This mimics how different market regimes and symbols exhibit different statistical patterns.

- **Kernels**: RBF (smooth), Matérn (less smooth, more realistic), Periodic (for seasonality)
- **Characteristics**: Smooth, continuous patterns with natural autocorrelation structure
- **Use case**: Improve model robustness to unseen patterns; fill gaps in real data

```python
from synthetic_timeseries_generation import SyntheticDatasetGenerator

gen = SyntheticDatasetGenerator(output_dir="./synthetic")

# Generate 100 symbols, 2000 bars each using Matérn kernel
df, metadata = gen.generate_gp_based(
    n_symbols=100,
    n_bars=2000,
    kernel_type="matern"
)

gen.save_parquet(df, name="gp_synthetic")
```

### 2. **Parametric Ensemble Generation** (Complementary)

Combines multiple well-studied financial process models:

- **GBM** (Geometric Brownian Motion): Standard log-normal price dynamics
- **Mean-Reverting**: Ornstein-Uhlenbeck-like; captures short-term reversals
- **Regime-Switching**: Discrete market states (normal/stress trading)
- **Stochastic Volatility**: Volatility clustering and vol-price correlation (Heston-like)

```python
# Generate 100 symbols (25 from each model)
df, metadata = gen.generate_parametric_ensemble(
    n_symbols=100,
    n_bars=2000,
    models=["gbm", "mean_revert", "regime_switch", "sv"]
)

gen.save_parquet(df, name="parametric_synthetic")
```

## Integration with tokenized-forecaster

The output format is **directly compatible** with the existing tokenized-forecaster pipeline:

```
synthetic_data/
  └── synthetic_year=2024.parquet  # symbol, date, open, high, low, close, volume
```

Can be consolidated and tokenized using the standard pipeline:

```bash
# Copy to tokenized-forecaster data directory
cp synthetic_data/*.parquet ../tokenized-forecaster/data/raw/

# Run standard pipeline (consolidate → fit → apply)
cd ../tokenized-forecaster
pipeline-mh --symbols SYN0000,SYN0001,GBM0000 --force
```

## Installation

```bash
pip install -e .  # with dependencies
pip install -e ".[test]"  # with test suite
```

## Usage

### CLI

```bash
# Generate 50 symbols with Gaussian Processes
generate-gp-synthetic --method gp --n-symbols 50 --n-bars 2000

# Generate with parametric ensemble
generate-gp-synthetic --method parametric --n-symbols 40

# Use different kernels
generate-gp-synthetic --method gp --kernel-type periodic --n-symbols 20

# With seed for reproducibility
generate-gp-synthetic --method gp --n-symbols 10 --seed 42 --output-dir ./synthetic
```

### Python API

```python
from synthetic_timeseries_generation import SyntheticDatasetGenerator, GaussianProcess
from synthetic_timeseries_generation import (
    generate_geometric_brownian_motion,
    generate_mean_reverting_process,
)

# Generate and save
gen = SyntheticDatasetGenerator(seed=42)
parquet_path, metadata_path = gen.generate_and_save(
    method="gp",
    n_symbols=50,
    n_bars=2000,
    kernel_type="matern"
)

# Direct access to lower-level APIs
from synthetic_timeseries_generation import generate_synthetic_paths
paths = generate_synthetic_paths(n_paths=100, path_length=500)

# GBM paths
from synthetic_timeseries_generation import generate_geometric_brownian_motion
prices = generate_geometric_brownian_motion(S0=100, n_steps=1000, n_paths=50)
```

## Module Structure

```
synthetic_timeseries_generation/
  gp_kernels.py              # Kernel implementations (RBF, Matérn, periodic)
  gp_processes.py            # GP inference and sampling
  price_models.py            # Parametric price processes (GBM, MR, RS, SV)
  synthetic_dataset.py       # End-to-end dataset generation
  cli.py                     # CLI entry point
```

## Key Design Decisions

1. **Gaussian Processes for diversity**: GPs with different kernels generate statistically-different but realistic paths. Trains the model to handle variety.

2. **Parametric ensemble for known regimes**: Financial models (GBM, mean-revert, regime-switch, SV) represent distinct market dynamics. Training on all ensures robustness.

3. **No domain-specific tuning to real data**: Synthetic parameters are generic (S0=100, vol~0.1%, etc.), preventing overfitting to observed market conditions.

4. **Parquet + year partition**: Integrates seamlessly with existing tokenized-forecaster infrastructure.

## Why Both Methods?

- **GP method**: Generates novel patterns the model might not have seen; improves generalization
- **Parametric method**: Validates that the model learns correct statistical properties of financial time series

This mirrors the Chronos approach: diverse, mathematically-grounded synthetic examples improve zero-shot performance on held-out real datasets.

## Performance Notes

- GP generation: ~500 symbols/minute (CPU, single-threaded)
- Parametric: ~1000 symbols/minute (vectorized)
- Memory: ~100 MB per 10K symbols

For large-scale pretraining (millions of symbols), parallelize across cores using multiprocessing.

## References

- **Chronos** (Ansari et al., 2024): "Chronos: Learning the Language of Time Series" — validates that synthetic + real data pretraining improves generalization
- **Gaussian Processes**: Rasmussen & Williams (2006) — kernel design and inference
- **Stochastic Volatility**: Heston (1993) — volatility clustering model
- **Regime-Switching**: Hamilton (1989) — discrete market states

## Future Extensions

- [ ] Multivariate GP (correlated symbols)
- [ ] Jump-diffusion models (gaps/gaps)
- [ ] Microstructure effects (bid-ask, slippage)
- [ ] Intra-day seasonality in volatility
- [ ] Integration with ARIMA/GARCH for baseline comparison
