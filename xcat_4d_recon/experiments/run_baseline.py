"""
Step 3c: Run SuPReMo baseline (all three variants).

Executes surrogate-driven, surrogate-free, and surrogate-optimized motion models
using the SuPReMo C++ binary from the 4DCT reference codebase.

Platform: Linux only.  On macOS, use Docker (see README.md).

Usage:
    python experiments/run_baseline.py
    python experiments/run_baseline.py --variants surrogate_driven surrogate_free
    python experiments/run_baseline.py --dry_run   # print commands, don't execute
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from methods.supremo.variants import get_variant_configs
from methods.supremo.runner import run_supremo

# Default path relative to project root
REF_DIR = Path(__file__).resolve().parent.parent.parent / "Reference codebases" / "4DCT-irregular-motion-main"
DEFAULT_BINARY = str(REF_DIR / "runSupremo")
DEFAULT_ANIMATE = str(REF_DIR / "animate")


def main():
    parser = argparse.ArgumentParser(description="Run SuPReMo baseline variants.")
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--output_dir", default="./outputs")
    parser.add_argument("--binary", default=DEFAULT_BINARY,
                        help="Path to runSupremo binary")
    parser.add_argument("--animate_binary", default=DEFAULT_ANIMATE,
                        help="Path to animate binary")
    parser.add_argument("--n_threads", type=int, default=4)
    parser.add_argument("--variants", nargs="+",
                        default=["surrogate_driven", "surrogate_free", "surrogate_optimized"],
                        choices=["surrogate_driven", "surrogate_free", "surrogate_optimized"])
    parser.add_argument("--train_only", action="store_true",
                        help="Fit motion model on training split only (fair eval)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print commands without executing")
    args = parser.parse_args()

    variant_configs = get_variant_configs(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        binary=args.binary,
        animate_binary=args.animate_binary,
        n_threads=args.n_threads,
        train_only=args.train_only,
    )

    for variant_name in args.variants:
        cfg = variant_configs[variant_name]
        print(f"\n{'=' * 60}")
        print(f"Running SuPReMo variant: {variant_name}")
        print(f"{'=' * 60}")
        run_supremo(cfg, dry_run=args.dry_run)

    print("\nBaseline runs complete.")


if __name__ == "__main__":
    main()
