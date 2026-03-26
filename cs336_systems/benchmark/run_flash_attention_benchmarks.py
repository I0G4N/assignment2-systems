from __future__ import annotations

import argparse
import csv
import itertools
import sys
import timeit
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flash_attention.flash_attention_pytorch import flash_attention

AttentionFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]

BATCH_SIZE = 8
DEFAULT_D_MODELS = [16, 32, 64, 128]
DEFAULT_SEQ_LENS = [256, 1024, 4096, 8192, 16384]
BLOCK_COLS = 64
BLOCK_ROWS = 64
DEFAULT_WARMUP_STEPS = 10
DEFAULT_TIMED_STEPS = 100


def flash_attention_impl(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return flash_attention(q, k, v, Bc=BLOCK_COLS, Br=BLOCK_ROWS, causal=False)


def compute_flops(batch_size: int, seq_len: int, d_model: int, is_backward: bool = False) -> float:
    forward_flops = 4.0 * batch_size * seq_len * seq_len * d_model
    return 2.0 * forward_flops if is_backward else forward_flops


def create_tensors(
    batch_size: int,
    seq_len: int,
    d_model: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype, requires_grad=True)
    return q, k, v


def sync_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def zero_grads(*tensors: torch.Tensor) -> None:
    for tensor in tensors:
        tensor.grad = None


def warmup_forward(
    attention_fn: AttentionFn,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    warmup_steps: int,
) -> None:
    for _ in range(warmup_steps):
        zero_grads(q, k, v)
        _ = attention_fn(q, k, v)
        sync_cuda(q.device)


def warmup_backward(
    attention_fn: AttentionFn,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    warmup_steps: int,
) -> None:
    for _ in range(warmup_steps):
        zero_grads(q, k, v)
        out = attention_fn(q, k, v)
        grad_out = torch.randn_like(out)
        out.backward(grad_out)
        sync_cuda(q.device)


def measure_forward(
    attention_fn: AttentionFn,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    warmup_steps: int,
    timed_steps: int,
) -> tuple[float, float]:
    warmup_forward(attention_fn, q, k, v, warmup_steps)

    times_ms: list[float] = []
    for _ in range(timed_steps):
        zero_grads(q, k, v)
        t0 = timeit.default_timer()
        _ = attention_fn(q, k, v)
        sync_cuda(q.device)
        t1 = timeit.default_timer()
        times_ms.append((t1 - t0) * 1000.0)

    times = np.array(times_ms)
    avg_ms = float(np.mean(times))
    std_ms = float(np.std(times, ddof=1)) if timed_steps > 1 else 0.0
    return avg_ms, std_ms


def measure_backward(
    attention_fn: AttentionFn,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    warmup_steps: int,
    timed_steps: int,
) -> tuple[float, float, float]:
    device = q.device
    warmup_backward(attention_fn, q, k, v, warmup_steps)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    zero_grads(q, k, v)
    out = attention_fn(q, k, v)
    sync_cuda(device)
    memory_mb = 0.0
    if device.type == "cuda":
        memory_mb = torch.cuda.memory_allocated(device) / (1024 ** 2)

    times_ms: list[float] = []
    for _ in range(timed_steps):
        zero_grads(q, k, v)
        out = attention_fn(q, k, v)
        grad_out = torch.randn_like(out)
        t0 = timeit.default_timer()
        out.backward(grad_out)
        sync_cuda(device)
        t1 = timeit.default_timer()
        times_ms.append((t1 - t0) * 1000.0)

    times = np.array(times_ms)
    avg_ms = float(np.mean(times))
    std_ms = float(np.std(times, ddof=1)) if timed_steps > 1 else 0.0
    return avg_ms, std_ms, memory_mb


def compute_throughput(
    batch_size: int,
    seq_len: int,
    d_model: int,
    fwd_time_ms: float,
    bwd_time_ms: float,
) -> dict[str, float]:
    fwd_flops = compute_flops(batch_size, seq_len, d_model, is_backward=False)
    bwd_flops = compute_flops(batch_size, seq_len, d_model, is_backward=True)
    total_flops = fwd_flops + bwd_flops
    total_time_s = (fwd_time_ms + bwd_time_ms) / 1000.0
    tokens_per_second = (batch_size * seq_len) / total_time_s if total_time_s > 0 else 0.0
    tflops_per_second = (total_flops / 1e12) / total_time_s if total_time_s > 0 else 0.0
    return {
        "tokens_per_sec": tokens_per_second,
        "tflops_per_sec": tflops_per_second,
        "total_flops": int(total_flops),
    }


def build_attention_variants(
    compile_mode: str,
    compile_backend: str,
) -> list[tuple[str, AttentionFn]]:
    base_fn = flash_attention_impl
    variants: list[tuple[str, AttentionFn]] = []

    if compile_mode in {"none", "both"}:
        variants.append(("eager", base_fn))

    if compile_mode in {"compiled", "both"}:
        compiled_fn = torch.compile(base_fn, backend=compile_backend)
        variants.append(("compiled", compiled_fn))

    return variants


def enrich_speedups(results: list[dict[str, Any]]) -> None:
    eager_times = {
        (row["d_model"], row["seq_len"]): row
        for row in results
        if row["variant"] == "eager"
    }

    for row in results:
        baseline = eager_times.get((row["d_model"], row["seq_len"]))
        if baseline is None:
            row["fwd_speedup_vs_eager"] = None
            row["bwd_speedup_vs_eager"] = None
            continue

        row["fwd_speedup_vs_eager"] = baseline["fwd_time_avg_ms"] / row["fwd_time_avg_ms"]
        row["bwd_speedup_vs_eager"] = baseline["bwd_time_avg_ms"] / row["bwd_time_avg_ms"]


def run_benchmarks(
    *,
    compile_mode: str,
    compile_backend: str,
    device: str,
    dtype_str: str,
    d_models: list[int],
    seq_lens: list[int],
    warmup_steps: int,
    timed_steps: int,
) -> list[dict[str, Any]]:
    device_obj = torch.device(device)
    if device_obj.type == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        device_obj = torch.device("cpu")

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[dtype_str]
    if device_obj.type == "cpu" and dtype in {torch.float16, torch.bfloat16}:
        raise ValueError("float16 and bfloat16 benchmarks require CUDA in this script")

    variants = build_attention_variants(compile_mode, compile_backend)
    total_configs = len(d_models) * len(seq_lens) * len(variants)

    print("Implementation: flash")
    print(f"Compile mode: {compile_mode}")
    if compile_mode != "none":
        print(f"Compile backend: {compile_backend}")
    print(f"Device: {device_obj}, dtype: {dtype}")
    print(f"Batch size: {BATCH_SIZE}, block sizes: ({BLOCK_COLS}, {BLOCK_ROWS})")
    print(f"Warmup steps: {warmup_steps}, timed steps: {timed_steps}")
    print()

    results: list[dict[str, Any]] = []
    progress = 1
    for variant_name, attention_fn in variants:
        print(f"Running variant: {variant_name}")
        for d_model, seq_len in itertools.product(d_models, seq_lens):
            print(
                f"  [{progress:2d}/{total_configs}] variant={variant_name:<8} d_model={d_model:3d} seq_len={seq_len:5d}...",
                end=" ",
                flush=True,
            )
            progress += 1

            try:
                q, k, v = create_tensors(BATCH_SIZE, seq_len, d_model, device_obj, dtype)
                fwd_avg_ms, fwd_std_ms = measure_forward(
                    attention_fn,
                    q,
                    k,
                    v,
                    warmup_steps=warmup_steps,
                    timed_steps=timed_steps,
                )
                bwd_avg_ms, bwd_std_ms, memory_mb = measure_backward(
                    attention_fn,
                    q,
                    k,
                    v,
                    warmup_steps=warmup_steps,
                    timed_steps=timed_steps,
                )
                throughput = compute_throughput(BATCH_SIZE, seq_len, d_model, fwd_avg_ms, bwd_avg_ms)
                results.append(
                    {
                        "variant": variant_name,
                        "d_model": d_model,
                        "seq_len": seq_len,
                        "batch_size": BATCH_SIZE,
                        "fwd_time_avg_ms": round(fwd_avg_ms, 3),
                        "fwd_time_std_ms": round(fwd_std_ms, 3),
                        "bwd_time_avg_ms": round(bwd_avg_ms, 3),
                        "bwd_time_std_ms": round(bwd_std_ms, 3),
                        "memory_mb": round(memory_mb, 2),
                        "tokens_per_sec": round(throughput["tokens_per_sec"], 2),
                        "tflops_per_sec": round(throughput["tflops_per_sec"], 4),
                        "total_flops": throughput["total_flops"],
                    }
                )
                print("ok")
            except RuntimeError as error:
                print(f"failed ({error})")

    enrich_speedups(results)
    return results


def format_speedup(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.2f}x"


def print_results_table(results: list[dict[str, Any]]) -> None:
    if not results:
        print("No results to display")
        return

    print("\n" + "=" * 180)
    print("ATTENTION BENCHMARK RESULTS")
    print("=" * 180)
    print(
        f"{'variant':>10} {'d_model':>8} {'seq_len':>8} {'batch':>7} "
        f"{'fwd_ms':>10} {'bwd_ms':>10} {'mem_mb':>10} {'tok/sec':>12} {'TFLOP/s':>10} {'fwd_spd':>9} {'bwd_spd':>9}"
    )
    print("-" * 180)
    for row in results:
        print(
            f"{row['variant']:>10} {row['d_model']:>8} {row['seq_len']:>8} {row['batch_size']:>7} "
            f"{row['fwd_time_avg_ms']:>10.3f} {row['bwd_time_avg_ms']:>10.3f} {row['memory_mb']:>10.2f} "
            f"{row['tokens_per_sec']:>12.2f} {row['tflops_per_sec']:>10.4f} "
            f"{format_speedup(row['fwd_speedup_vs_eager']):>9} {format_speedup(row['bwd_speedup_vs_eager']):>9}"
        )
    print("=" * 180)


def save_results_csv(results: list[dict[str, Any]], csv_output: str) -> None:
    if not results:
        print("No results to save")
        return

    csv_path = Path(csv_output)
    if not csv_path.is_absolute():
        csv_path = Path(__file__).parent / csv_path

    fieldnames = [
        "variant",
        "d_model",
        "seq_len",
        "batch_size",
        "fwd_time_avg_ms",
        "fwd_time_std_ms",
        "bwd_time_avg_ms",
        "bwd_time_std_ms",
        "memory_mb",
        "tokens_per_sec",
        "tflops_per_sec",
        "total_flops",
        "fwd_speedup_vs_eager",
        "bwd_speedup_vs_eager",
    ]

    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Results saved to: {csv_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark eager and compiled attention implementations across multiple sequence lengths and model dimensions."
    )
    parser.add_argument(
        "--compile",
        dest="compile_mode",
        choices=["none", "compiled", "both"],
        default="none",
        help="Run eager only, compiled only, or compare both eager and compiled variants.",
    )
    parser.add_argument(
        "--compile-backend",
        default="inductor",
        help="Backend passed to torch.compile when compiled variants are requested.",
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on.",
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "float16", "bfloat16"],
        default="float32",
        help="Tensor dtype.",
    )
    parser.add_argument(
        "--d-models",
        nargs="+",
        type=int,
        default=DEFAULT_D_MODELS,
        help="List of embedding dimensions to benchmark.",
    )
    parser.add_argument(
        "--seq-lens",
        nargs="+",
        type=int,
        default=DEFAULT_SEQ_LENS,
        help="List of sequence lengths to benchmark.",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=DEFAULT_WARMUP_STEPS,
        help="Warmup iterations per configuration before timing.",
    )
    parser.add_argument(
        "--timed-steps",
        type=int,
        default=DEFAULT_TIMED_STEPS,
        help="Timed iterations per configuration.",
    )
    parser.add_argument(
        "--csv-output",
        default="flash_attention_results.csv",
        help="CSV file path for benchmark results.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = run_benchmarks(
        compile_mode=args.compile_mode,
        compile_backend=args.compile_backend,
        device=args.device,
        dtype_str=args.dtype,
        d_models=args.d_models,
        seq_lens=args.seq_lens,
        warmup_steps=args.warmup_steps,
        timed_steps=args.timed_steps,
    )
    print_results_table(results)
    save_results_csv(results, args.csv_output)


if __name__ == "__main__":
    main()
