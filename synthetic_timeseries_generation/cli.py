"""Command-line interface for synthetic data generation."""

import argparse
import sys
from pathlib import Path

from .synthetic_dataset import SyntheticDatasetGenerator


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic OHLCV data for stock forecasting model training.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  # Generate 50 symbols using Gaussian Processes
  generate-gp-synthetic --method gp --n-symbols 50 --n-bars 2000 --output-dir ./synthetic

  # Generate using parametric ensemble (GBM + mean-revert + regime-switch + SV)
  generate-gp-synthetic --method parametric --n-symbols 40 --output-dir ./synthetic

  # Use Matern kernel for smoother paths
  generate-gp-synthetic --method gp --n-symbols 20 --kernel-type matern --output-dir ./synthetic
        """,
    )

    parser.add_argument(
        "--method",
        choices=["gp", "parametric"],
        default="gp",
        help="Generation method (default: gp)",
    )
    parser.add_argument(
        "--n-symbols",
        type=int,
        default=50,
        help="Number of synthetic symbols (default: 50)",
    )
    parser.add_argument(
        "--n-bars",
        type=int,
        default=2000,
        help="Number of bars per symbol (default: 2000)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./synthetic_data",
        help="Output directory (default: ./synthetic_data)",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="synthetic",
        help="Base name for output files (default: synthetic)",
    )
    parser.add_argument(
        "--kernel-type",
        choices=["rbf", "matern", "periodic"],
        default="matern",
        help="GP kernel type for --method gp (default: matern)",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2024-01-01",
        help="Start date for synthetic data (default: 2024-01-01)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed logging",
    )

    args = parser.parse_args()

    if args.verbose:
        print(f"Generating synthetic data:")
        print(f"  Method: {args.method}")
        print(f"  Symbols: {args.n_symbols}")
        print(f"  Bars: {args.n_bars}")
        print(f"  Output: {args.output_dir}")
        if args.method == "gp":
            print(f"  Kernel: {args.kernel_type}")
        if args.seed:
            print(f"  Seed: {args.seed}")

    gen = SyntheticDatasetGenerator(output_dir=args.output_dir, seed=args.seed)

    try:
        parquet_path, metadata_path = gen.generate_and_save(
            method=args.method,
            n_symbols=args.n_symbols,
            n_bars=args.n_bars,
            start_date=args.start_date,
            output_name=args.output_name,
            kernel_type=args.kernel_type if args.method == "gp" else None,
        )

        if args.verbose:
            print(f"\nSuccess!")
            print(f"  Parquet: {parquet_path}")
            print(f"  Metadata: {metadata_path}")
        else:
            print(f"✓ {parquet_path}")
            print(f"✓ {metadata_path}")

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
