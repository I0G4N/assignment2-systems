#!/usr/bin/env python3
"""
Benchmark script comparing Triton FlashAttention-2 vs naive PyTorch attention.

Uses triton.testing.do_bench for timing. Sweeps over:
  - Sequence lengths: powers of 2 from 128 to 65536
  - Embedding dimensions: 16, 32, 64, 128
  - Dtypes: torch.bfloat16, torch.float32
Always uses batch_size=1 and causal masking.
"""

import itertools
import math
import sys
from pathlib import Path

import torch
import triton
import triton.testing

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flash_attention.flash_attention_triton import FlashAttentionTriton
from cs336_systems.benchmark.run_flash_attention_benchmarks import create_tensors


# ---------------------------------------------------------------------------
# Naive PyTorch attention (standard matmul-softmax-matmul, NOT FlashAttention)
# ---------------------------------------------------------------------------

def naive_attention(Q, K, V, is_causal=True):
    """Standard scaled dot-product attention with optional causal mask."""
    scale = 1.0 / math.sqrt(Q.shape[-1])
    S = torch.matmul(Q, K.transpose(-2, -1)) * scale
    if is_causal:
        seq_len = Q.shape[-2]
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=Q.device, dtype=torch.bool),
            diagonal=1,
        )
        S = S.masked_fill(mask, float("-inf"))
    P = torch.softmax(S, dim=-1)
    return torch.matmul(P, V)


# ---------------------------------------------------------------------------
# Benchmarking helpers
# ---------------------------------------------------------------------------

BATCH_SIZE = 1
SEQ_LENS = [2**i for i in range(7, 17)]  # 128 .. 65536
D_KS = [16, 32, 64, 128]
DTYPES = [torch.bfloat16, torch.float32]


def _try_bench(fn, grad_to_none=None):
    """Run triton.testing.do_bench; return ms or 'OOM'."""
    try:
        ms = triton.testing.do_bench(fn, grad_to_none=grad_to_none)
        return ms
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return "OOM"


def bench_config(seq_len, d_k, dtype, device="cuda"):
    """Benchmark one (seq_len, d_k, dtype) configuration."""
    result = {
        "seq_len": seq_len,
        "d_k": d_k,
        "dtype": str(dtype).split(".")[-1],
    }

    # ---- create inputs ----
    Q, K, V = create_tensors(BATCH_SIZE, seq_len, d_k, torch.device(device), dtype)
    grad_out = torch.randn(BATCH_SIZE, seq_len, d_k, device=device, dtype=dtype)

    # ---- Triton forward ----
    def triton_fwd():
        return FlashAttentionTriton.apply(Q, K, V, True)

    result["triton_fwd_ms"] = _try_bench(triton_fwd, grad_to_none=[Q, K, V])

    # ---- Triton backward ----
    if result["triton_fwd_ms"] != "OOM":
        try:
            out_tri = triton_fwd()
            result["triton_bwd_ms"] = _try_bench(
                lambda: out_tri.backward(grad_out, retain_graph=True),
                grad_to_none=[Q, K, V],
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            result["triton_bwd_ms"] = "OOM"
    else:
        result["triton_bwd_ms"] = "OOM"

    # ---- Triton forward + backward ----
    def triton_fwd_bwd():
        o = FlashAttentionTriton.apply(Q, K, V, True)
        o.backward(grad_out)

    result["triton_fwd_bwd_ms"] = _try_bench(triton_fwd_bwd, grad_to_none=[Q, K, V])

    # ---- Naive PyTorch forward ----
    def pytorch_fwd():
        return naive_attention(Q, K, V, is_causal=True)

    result["pytorch_fwd_ms"] = _try_bench(pytorch_fwd, grad_to_none=[Q, K, V])

    # ---- Naive PyTorch backward ----
    if result["pytorch_fwd_ms"] != "OOM":
        try:
            out_pt = pytorch_fwd()
            result["pytorch_bwd_ms"] = _try_bench(
                lambda: out_pt.backward(grad_out, retain_graph=True),
                grad_to_none=[Q, K, V],
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            result["pytorch_bwd_ms"] = "OOM"
    else:
        result["pytorch_bwd_ms"] = "OOM"

    # ---- Naive PyTorch forward + backward ----
    def pytorch_fwd_bwd():
        o = naive_attention(Q, K, V, is_causal=True)
        o.backward(grad_out)

    result["pytorch_fwd_bwd_ms"] = _try_bench(pytorch_fwd_bwd, grad_to_none=[Q, K, V])

    return result


# ---------------------------------------------------------------------------
# Table formatting
# ---------------------------------------------------------------------------

def _fmt(val):
    """Format a latency value for display."""
    if isinstance(val, str):
        return val
    return f"{val:.3f}"


def _speedup(triton_val, pytorch_val):
    """Compute speedup of Triton over PyTorch."""
    if isinstance(triton_val, str) or isinstance(pytorch_val, str):
        return "N/A"
    if triton_val == 0:
        return "N/A"
    return f"{pytorch_val / triton_val:.2f}x"


def print_table(results):
    """Print results as a formatted table."""
    header = (
        f"{'seq_len':>8} {'d_k':>4} {'dtype':>10} | "
        f"{'tri_fwd':>10} {'pt_fwd':>10} {'spd_fwd':>8} | "
        f"{'tri_bwd':>10} {'pt_bwd':>10} {'spd_bwd':>8} | "
        f"{'tri_f+b':>10} {'pt_f+b':>10} {'spd_f+b':>8}"
    )
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)

    for r in results:
        tri_fwd = _fmt(r["triton_fwd_ms"])
        pt_fwd = _fmt(r["pytorch_fwd_ms"])
        tri_bwd = _fmt(r["triton_bwd_ms"])
        pt_bwd = _fmt(r["pytorch_bwd_ms"])
        tri_fb = _fmt(r["triton_fwd_bwd_ms"])
        pt_fb = _fmt(r["pytorch_fwd_bwd_ms"])

        spd_fwd = _speedup(r["triton_fwd_ms"], r["pytorch_fwd_ms"])
        spd_bwd = _speedup(r["triton_bwd_ms"], r["pytorch_bwd_ms"])
        spd_fb = _speedup(r["triton_fwd_bwd_ms"], r["pytorch_fwd_bwd_ms"])

        print(
            f"{r['seq_len']:>8} {r['d_k']:>4} {r['dtype']:>10} | "
            f"{tri_fwd:>10} {pt_fwd:>10} {spd_fwd:>8} | "
            f"{tri_bwd:>10} {pt_bwd:>10} {spd_bwd:>8} | "
            f"{tri_fb:>10} {pt_fb:>10} {spd_fb:>8}"
        )

    print(sep)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = "cuda"
    assert torch.cuda.is_available(), "CUDA is required for this benchmark"

    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Batch size: {BATCH_SIZE}, Causal: True")
    print(f"Sequence lengths: {SEQ_LENS}")
    print(f"Embedding dims:   {D_KS}")
    print(f"Dtypes:           {[str(d).split('.')[-1] for d in DTYPES]}")
    print()

    all_results = []
    configs = list(itertools.product(SEQ_LENS, D_KS, DTYPES))
    total = len(configs)

    for i, (seq_len, d_k, dtype) in enumerate(configs):
        print(
            f"[{i + 1}/{total}] seq_len={seq_len:>5}, d_k={d_k:>3}, "
            f"dtype={str(dtype).split('.')[-1]}"
        )
        result = bench_config(seq_len, d_k, dtype, device)
        all_results.append(result)
        # Free GPU memory between configs
        torch.cuda.empty_cache()

    print("\n" + "=" * 40)
    print("  RESULTS")
    print("=" * 40 + "\n")
    print_table(all_results)


if __name__ == "__main__":
    main()
