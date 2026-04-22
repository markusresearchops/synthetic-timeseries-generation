"""Tests for synthetic dataset generation."""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import tempfile

from synthetic_timeseries_generation import SyntheticDatasetGenerator


class TestSyntheticDatasetGenerator:
    def test_gp_generation(self):
        """Test GP-based synthetic data generation."""
        gen = SyntheticDatasetGenerator(seed=42)

        df, metadata = gen.generate_gp_based(
            n_symbols=5,
            n_bars=100,
            kernel_type="matern"
        )

        # Check shape
        assert len(df) == 5 * 100
        assert len(df.columns) >= 6  # symbol, date, OHLCV

        # Check columns
        assert "symbol" in df.columns
        assert "open" in df.columns
        assert "high" in df.columns
        assert "low" in df.columns
        assert "close" in df.columns
        assert "volume" in df.columns

        # Check data types
        assert df["volume"].dtype == int or df["volume"].dtype == np.int64

        # Check OHLC relationships
        assert (df["high"] >= df["low"]).all()
        assert (df["high"] >= df["open"]).all()
        assert (df["high"] >= df["close"]).all()

        # Check metadata
        assert metadata["method"] == "gaussian_process"
        assert metadata["n_symbols"] == 5
        assert metadata["n_bars"] == 100

    def test_parametric_generation(self):
        """Test parametric ensemble generation."""
        gen = SyntheticDatasetGenerator(seed=42)

        df, metadata = gen.generate_parametric_ensemble(
            n_symbols=8,
            n_bars=100,
            models=["gbm", "mean_revert"]
        )

        assert len(df) == 8 * 100
        assert metadata["method"] == "parametric_ensemble"
        assert "gbm" in metadata["models"]

    def test_save_parquet(self):
        """Test parquet saving."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gen = SyntheticDatasetGenerator(output_dir=tmpdir, seed=42)
            df, _ = gen.generate_gp_based(n_symbols=2, n_bars=50)

            path = gen.save_parquet(df, name="test")

            assert path.exists()
            df_loaded = pd.read_parquet(path)
            assert len(df_loaded) == len(df)

    def test_generate_and_save(self):
        """Test one-shot generate and save."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gen = SyntheticDatasetGenerator(output_dir=tmpdir, seed=42)

            parquet_path, metadata_path = gen.generate_and_save(
                method="gp",
                n_symbols=3,
                n_bars=50,
                output_name="test"
            )

            assert parquet_path.exists()
            assert metadata_path.exists()

    def test_reproducibility(self):
        """Test that same seed produces same data."""
        gen1 = SyntheticDatasetGenerator(seed=123)
        df1, _ = gen1.generate_gp_based(n_symbols=2, n_bars=50)

        gen2 = SyntheticDatasetGenerator(seed=123)
        df2, _ = gen2.generate_gp_based(n_symbols=2, n_bars=50)

        # Check that prices are identical (within floating point precision)
        assert np.allclose(df1["close"].values, df2["close"].values)
