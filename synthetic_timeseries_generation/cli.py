"""CLI: generate synthetic OHLCV bars via paper-faithful KernelSynth + TSMixup.

Output is parquet in tokenized-forecaster's per-symbol-year format
(`year=<Y>.parquet` with a `symbol` column), so the downstream pipeline
can ingest it directly via `consolidate-bars` or by dropping into the
consolidated layout.

Usage:
    chronos-synth kernelsynth --n 100 --output-dir ./out
    chronos-synth tsmixup --source data.parquet --column close --n 1000 --output ./mix.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .chronos_kernel_synth import generate_kernel_synth_dataset
from .ohlcv_adapter import OHLCVConfig, batch_to_ohlcv
from .tsmixup import generate_tsmixup_dataset, parquet_column_source


def _cmd_kernelsynth(args: argparse.Namespace) -> int:
    print(f"generating {args.n} synthetic series via KernelSynth (l_syn={args.length})...")
    series = generate_kernel_synth_dataset(
        n_series=args.n, l_syn=args.length, j_max=args.j_max, seed=args.seed,
    )
    if args.format == "univariate":
        out_path = Path(args.output_dir) / f"kernelsynth_n{args.n}.npy"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, series)
        print(f"wrote {series.shape} → {out_path}")
        return 0

    # OHLCV format
    df = batch_to_ohlcv(
        series,
        symbol_prefix=args.symbol_prefix,
        start_date=args.start_date,
        config=OHLCVConfig(start_price=args.start_price),
        seed=args.seed,
    )
    out_path = Path(args.output_dir) / f"kernelsynth_year={args.year}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"wrote {len(df):,} OHLCV rows × {df['symbol'].nunique()} symbols → {out_path}")
    return 0


def _cmd_tsmixup(args: argparse.Namespace) -> int:
    src_paths = [Path(p) for p in args.source]
    sources = [parquet_column_source(p, args.column) for p in src_paths]
    print(f"generating {args.n} TSMixup augmentations from {len(sources)} source(s) "
          f"on column '{args.column}' (K={args.K}, α={args.alpha})...")
    out = generate_tsmixup_dataset(
        sources, n_series=args.n,
        K=args.K, alpha=args.alpha,
        l_min=args.l_min, l_max=args.l_max, seed=args.seed,
    )
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({
        "augmentation_id": np.arange(len(out), dtype=np.int64),
        "length": [len(x) for x in out],
        "values": [x.tolist() for x in out],
    })
    df.to_parquet(out_path, index=False)
    print(f"wrote {len(df):,} variable-length augmentations → {out_path} "
          f"(lengths {df['length'].min()}–{df['length'].max()})")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    # ---- kernelsynth ----
    ks = sub.add_parser("kernelsynth",
                        help="generate KernelSynth synthetic series (Algorithm 2)")
    ks.add_argument("--n", type=int, default=100, help="number of series")
    ks.add_argument("--length", type=int, default=1024, help="l_syn (paper default 1024)")
    ks.add_argument("--j-max", type=int, default=5, help="J (max kernels per series)")
    ks.add_argument("--seed", type=int, default=0)
    ks.add_argument("--format", choices=("ohlcv", "univariate"), default="ohlcv",
                    help="ohlcv = build OHLCV bars (default); univariate = raw .npy")
    ks.add_argument("--symbol-prefix", default="SYN")
    ks.add_argument("--start-date", default="2024-01-02 14:30")
    ks.add_argument("--start-price", type=float, default=100.0)
    ks.add_argument("--year", type=int, default=2024)
    ks.add_argument("--output-dir", default="./synthetic_data")
    ks.set_defaults(func=_cmd_kernelsynth)

    # ---- tsmixup ----
    tm = sub.add_parser("tsmixup",
                        help="generate TSMixup augmentations from real series (Algorithm 1)")
    tm.add_argument("--source", nargs="+", required=True,
                    help="one or more source parquet files (each treated as one dataset)")
    tm.add_argument("--column", required=True,
                    help="column name to extract (e.g. 'close')")
    tm.add_argument("--n", type=int, default=1000)
    tm.add_argument("--K", type=int, default=3, help="max series to mix (paper K=3)")
    tm.add_argument("--alpha", type=float, default=1.5, help="Dirichlet α (paper 1.5)")
    tm.add_argument("--l-min", type=int, default=128)
    tm.add_argument("--l-max", type=int, default=2048)
    tm.add_argument("--seed", type=int, default=0)
    tm.add_argument("--output", default="./tsmixup_augmentations.parquet")
    tm.set_defaults(func=_cmd_tsmixup)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
