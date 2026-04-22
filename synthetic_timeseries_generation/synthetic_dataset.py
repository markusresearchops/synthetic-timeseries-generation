"""
End-to-end synthetic dataset generation for stock forecasting.

Generates synthetic OHLCV data in formats compatible with tokenized-forecaster.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
from datetime import datetime, timedelta

from .gp_processes import generate_synthetic_ohlcv, GaussianProcess
from .gp_kernels import RBFKernel, MaternKernel, ExpSineSquaredKernel, compose_kernel
from .price_models import (
    generate_geometric_brownian_motion,
    generate_mean_reverting_process,
    generate_regime_switching_process,
    generate_stochastic_volatility_process,
)


class SyntheticDatasetGenerator:
    """Generate synthetic OHLCV datasets for model training and regularization."""

    def __init__(self, output_dir=None, seed=None):
        """Initialize generator.

        Args:
            output_dir: directory for output parquet files
            seed: random seed for reproducibility
        """
        self.output_dir = Path(output_dir) if output_dir else Path("./synthetic_data")
        self.seed = seed
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if seed is not None:
            np.random.seed(seed)

    def generate_gp_based(
        self,
        n_symbols=10,
        n_bars=1000,
        start_date="2024-01-01",
        kernel_type="matern",
        noise_level=0.0,
    ):
        """Generate synthetic data using Gaussian Processes.

        This mimics the Chronos approach: diverse patterns from GP priors
        help the model generalize to unseen series.

        Args:
            n_symbols: number of synthetic symbols to generate
            n_bars: bars per symbol
            start_date: start date as string
            kernel_type: "rbf", "matern", or "periodic"
            noise_level: add Gaussian noise to OHLCV

        Returns:
            df: concatenated DataFrame with all synthetic data
            metadata: dict with generation parameters
        """
        all_data = []
        metadata = {
            "method": "gaussian_process",
            "n_symbols": n_symbols,
            "n_bars": n_bars,
            "kernel_type": kernel_type,
            "generated_at": datetime.now().isoformat(),
        }

        # Select kernel
        if kernel_type == "rbf":
            kernel = RBFKernel(variance=1.0, lengthscale=20.0)
        elif kernel_type == "matern":
            kernel = MaternKernel(variance=1.0, lengthscale=20.0, nu=5.0 / 2.0)
        elif kernel_type == "periodic":
            # Compose: trend (RBF) + periodicity
            trend_kernel = RBFKernel(variance=1.0, lengthscale=100.0)
            periodic_kernel = ExpSineSquaredKernel(variance=0.5, lengthscale=5.0, period=50)
            kernel = trend_kernel + 0.3 * periodic_kernel
        else:
            kernel = RBFKernel(variance=1.0, lengthscale=20.0)

        data = generate_synthetic_ohlcv(
            n_paths=n_symbols,
            path_length=n_bars,
            kernel=kernel,
            seed=self.seed,
        )

        # Create timestamps
        start = pd.Timestamp(start_date)
        timestamps = pd.date_range(start, periods=n_bars, freq="1min")

        # Construct DataFrame
        for symbol_idx in range(n_symbols):
            symbol_name = f"SYN{symbol_idx:04d}"

            df_symbol = pd.DataFrame({
                "symbol": symbol_name,
                "date": timestamps,
                "open": data["open"][symbol_idx],
                "high": data["high"][symbol_idx],
                "low": data["low"][symbol_idx],
                "close": data["close"][symbol_idx],
                "volume": data["volume"][symbol_idx].astype(int),
            })

            # Add noise if requested
            if noise_level > 0:
                noise = np.random.normal(0, noise_level, (n_bars, 4))
                for i, col in enumerate(["open", "high", "low", "close"]):
                    df_symbol[col] *= (1 + noise[:, i])

            # Ensure OHLC relationships
            df_symbol["high"] = df_symbol[["open", "high", "low", "close"]].max(axis=1)
            df_symbol["low"] = df_symbol[["open", "high", "low", "close"]].min(axis=1)

            all_data.append(df_symbol)

        df_combined = pd.concat(all_data, ignore_index=True)
        return df_combined, metadata

    def generate_parametric_ensemble(
        self,
        n_symbols=10,
        n_bars=1000,
        start_date="2024-01-01",
        models=None,
    ):
        """Generate synthetic data using ensemble of parametric models.

        Combines GBM, mean-reverting, regime-switching, stochastic vol processes.

        Args:
            n_symbols: total number of symbols across all models
            n_bars: bars per symbol
            start_date: start date string
            models: list of model names (default: all)

        Returns:
            df: concatenated DataFrame
            metadata: dict with model distribution
        """
        if models is None:
            models = ["gbm", "mean_revert", "regime_switch", "sv"]

        all_data = []
        symbols_per_model = n_symbols // len(models)
        metadata = {
            "method": "parametric_ensemble",
            "models": models,
            "n_symbols": n_symbols,
            "n_bars": n_bars,
            "generated_at": datetime.now().isoformat(),
        }

        start = pd.Timestamp(start_date)
        timestamps = pd.date_range(start, periods=n_bars, freq="1min")

        symbol_counter = 0

        # GBM
        if "gbm" in models:
            paths = generate_geometric_brownian_motion(
                S0=100, mu=0.0001, sigma=0.001, n_steps=n_bars, n_paths=symbols_per_model, seed=self.seed
            )
            for i in range(symbols_per_model):
                symbol_name = f"GBM{i:04d}"
                df = self._construct_ohlcv_from_close(paths[i], symbol_name, timestamps)
                all_data.append(df)
                symbol_counter += 1

        # Mean-reverting
        if "mean_revert" in models:
            paths = generate_mean_reverting_process(
                S0=100, long_mean=100, mean_reversion_speed=0.05, sigma=0.015,
                n_steps=n_bars, n_paths=symbols_per_model, seed=self.seed
            )
            for i in range(symbols_per_model):
                symbol_name = f"MR{i:04d}"
                df = self._construct_ohlcv_from_close(paths[i], symbol_name, timestamps)
                all_data.append(df)
                symbol_counter += 1

        # Regime-switching
        if "regime_switch" in models:
            paths, _ = generate_regime_switching_process(
                S0=100, n_steps=n_bars, n_paths=symbols_per_model, seed=self.seed
            )
            for i in range(symbols_per_model):
                symbol_name = f"RS{i:04d}"
                df = self._construct_ohlcv_from_close(paths[i], symbol_name, timestamps)
                all_data.append(df)
                symbol_counter += 1

        # Stochastic volatility
        if "sv" in models:
            paths, _ = generate_stochastic_volatility_process(
                S0=100, n_steps=n_bars, n_paths=symbols_per_model, seed=self.seed
            )
            for i in range(symbols_per_model):
                symbol_name = f"SV{i:04d}"
                df = self._construct_ohlcv_from_close(paths[i], symbol_name, timestamps)
                all_data.append(df)
                symbol_counter += 1

        df_combined = pd.concat(all_data, ignore_index=True)
        return df_combined, metadata

    def _construct_ohlcv_from_close(self, close_prices, symbol_name, timestamps):
        """Helper to construct OHLCV from close prices."""
        n = len(close_prices)
        opens = close_prices.copy()
        opens[1:] = close_prices[:-1] + np.random.normal(0, 0.1 * close_prices[:-1])
        opens[0] = close_prices[0]

        highs = np.maximum(opens, close_prices) + np.abs(np.random.normal(0, 0.2 * close_prices))
        lows = np.minimum(opens, close_prices) - np.abs(np.random.normal(0, 0.2 * close_prices))

        volume = np.random.lognormal(14, 1.0, n)  # ~1M mean

        df = pd.DataFrame({
            "symbol": symbol_name,
            "date": timestamps,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": close_prices,
            "volume": volume.astype(int),
        })
        return df

    def save_parquet(self, df, name="synthetic_data", year=2024):
        """Save to parquet in tokenized-forecaster format.

        Args:
            df: DataFrame with symbol, date, OHLCV columns
            name: base name for output files
            year: year partition

        Returns:
            path: saved parquet file path
        """
        output_path = self.output_dir / f"{name}_year={year}.parquet"
        df.to_parquet(output_path, engine="pyarrow", index=False)
        print(f"Saved {len(df)} rows to {output_path}")
        return output_path

    def save_metadata(self, metadata, name="synthetic_data"):
        """Save generation metadata."""
        meta_path = self.output_dir / f"{name}_metadata.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        return meta_path

    def generate_and_save(
        self,
        method="gp",
        n_symbols=10,
        n_bars=1000,
        output_name="synthetic",
        **kwargs
    ):
        """One-shot: generate and save synthetic dataset.

        Args:
            method: "gp" or "parametric"
            n_symbols: number of symbols
            n_bars: bars per symbol
            output_name: base name for files
            **kwargs: passed to generation method

        Returns:
            (parquet_path, metadata_path)
        """
        if method == "gp":
            df, metadata = self.generate_gp_based(n_symbols=n_symbols, n_bars=n_bars, **kwargs)
        elif method == "parametric":
            df, metadata = self.generate_parametric_ensemble(n_symbols=n_symbols, n_bars=n_bars, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")

        parquet_path = self.save_parquet(df, name=output_name)
        metadata_path = self.save_metadata(metadata, name=output_name)

        return parquet_path, metadata_path
