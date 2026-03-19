"""
Run run_benchmarks for each standard model size and print a Markdown table.

Usage:
    python -m cs336_systems.spec_benchmarks [options]

All options from run_benchmarks are forwarded (--mode, --batch-size,
--context-length, --warmup-steps, --timed-steps, --device, --dtype,
--dataset-path). The per-size hyperparameters (d_model, d_ff, num_layers,
num_heads) are fixed by the spec table and cannot be overridden on the CLI.
"""

from __future__ import annotations

import argparse
from typing import Any

import pandas as pd

from cs336_systems.benchmark.run_benchmarks import benchmark

MODEL_SPECS: list[dict[str, Any]] = [
    {"size": "small",  "d_model": 768,  "d_ff": 3072,  "num_layers": 12, "num_heads": 12},
    {"size": "medium", "d_model": 1024, "d_ff": 4096,  "num_layers": 24, "num_heads": 16},
    {"size": "large",  "d_model": 1280, "d_ff": 5120,  "num_layers": 36, "num_heads": 20},
    {"size": "xl",     "d_model": 1600, "d_ff": 6400,  "num_layers": 48, "num_heads": 25},
    {"size": "2.7B",   "d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark all standard model sizes and output a Markdown table."
    )
    parser.add_argument("--vocab-size",     type=int,   default=10000)
    parser.add_argument("--context-length", type=int,   default=1024)
    parser.add_argument("--batch-size",     type=int,   default=4)
    parser.add_argument("--warmup-steps",   type=int,   default=5)
    parser.add_argument("--timed-steps",    type=int,   default=10)
    parser.add_argument(
        "--mode",
        choices=["forward", "forward-backward", "all"],
        default="all",
        help="Which pass(es) to benchmark. 'all' runs both forward and forward-backward.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device override (default: auto-detected by run_benchmarks).",
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "float16", "bfloat16"],
        default="float32",
    )
    parser.add_argument("--dataset-path", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    device = args.device or ("cuda" if __import__("torch").cuda.is_available() else "cpu")
    rows = []

    modes = ("forward", "forward-backward") if args.mode == "all" else (args.mode,)

    for spec in MODEL_SPECS:
        for mode in modes:
            print(f"[running] size={spec['size']}  mode={mode}", flush=True)
            results = benchmark(
                vocab_size=args.vocab_size,
                context_length=args.context_length,
                d_model=spec["d_model"],
                num_layers=spec["num_layers"],
                num_heads=spec["num_heads"],
                d_ff=spec["d_ff"],
                rope_theta=10000.0,
                batch_size=args.batch_size,
                mode=mode,
                device=device,
                dtype=args.dtype,
                warmup_steps=args.warmup_steps,
                timed_steps=args.timed_steps,
                dataset_path=args.dataset_path,
            )
            rows.append({
                "size":              spec["size"],
                "d_model":           spec["d_model"],
                "d_ff":              spec["d_ff"],
                "num_layers":        spec["num_layers"],
                "num_heads":         spec["num_heads"],
                "device":            results["device"],
                "dtype":             results["dtype"],
                "mode":              results["mode"],
                "warmup_steps":      results["warmup_steps"],
                "avg_step_time_ms":  results["avg_step_time_ms"],
                "std_step_time_ms":  results["std_step_time_ms"],
                "tokens_per_second": results["tokens_per_second"],
                "total_time_s":      results["total_time_s"],
            })
            print(
                f"  avg_step_time_ms={results['avg_step_time_ms']} ± {results['std_step_time_ms']}  "
                f"tokens/s={results['tokens_per_second']}",
                flush=True,
            )
        break  # TODO: remove this when we're ready to run all benchmarks

    df = pd.DataFrame(rows, columns=[
        "size", "d_model", "d_ff", "num_layers", "num_heads",
        "device", "dtype", "mode", "warmup_steps",
        "avg_step_time_ms", "std_step_time_ms", "tokens_per_second", "total_time_s",
    ])

    print("\n" + df.to_markdown(index=False))


if __name__ == "__main__":
    main()
