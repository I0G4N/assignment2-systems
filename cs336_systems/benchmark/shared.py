from __future__ import annotations

import argparse
from enum import auto
import timeit
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from cs336_basics.data import get_batch
from cs336_basics.model import BasicsTransformerLM

BenchmarkStepFn = Callable[
	[BasicsTransformerLM, torch.Tensor, torch.Tensor, str, torch.device, bool, Any | None],
	None,
]

VALID_MODES = {"forward", "forward-backward", "forward-backward-optimizer"}


def build_benchmark_arg_parser(
	*,
	description: str,
	mode_default: str,
	vocab_size_default: int = 10000,
	context_length_default: int = 128,
	d_model_default: int = 1024,
	num_layers_default: int = 24,
	num_heads_default: int = 16,
	d_ff_default: int = 4096,
	rope_theta_default: float = 10000.0,
	batch_size_default: int = 4,
	warmup_steps_default: int = 10,
	timed_steps_default: int = 20,
) -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description=description)
	parser.add_argument("--vocab-size", type=int, default=vocab_size_default)
	parser.add_argument("--context-length", type=int, default=context_length_default)
	parser.add_argument("--d-model", type=int, default=d_model_default)
	parser.add_argument("--num-layers", type=int, default=num_layers_default)
	parser.add_argument("--num-heads", type=int, default=num_heads_default)
	parser.add_argument("--d-ff", type=int, default=d_ff_default)
	parser.add_argument("--rope-theta", type=float, default=rope_theta_default)
	parser.add_argument("--batch-size", type=int, default=batch_size_default)
	parser.add_argument(
		"--dataset-path",
		type=str,
		default=None,
		help="Optional path to tokenized dataset (.npy or .pt). If omitted, random tokens are used.",
	)
	parser.add_argument("--warmup-steps", type=int, default=warmup_steps_default)
	parser.add_argument("--timed-steps", type=int, default=timed_steps_default)
	parser.add_argument(
		"--mode",
		choices=sorted(VALID_MODES),
		default=mode_default,
		help="Benchmark mode: forward, forward-backward, or forward-backward-optimizer.",
	)
	parser.add_argument(
		"--memory-profiler-filename",
		type=str,
		default=None,
		help="Optional filename to save memory profiling results (e.g. from torch.profiler). If omitted, memory profiling is skipped.",
	)
	parser.add_argument(
		"--device",
		type=str,
		default="cuda" if torch.cuda.is_available() else "cpu",
	)
	parser.add_argument(
		"--dtype",
		choices=["float32", "float16", "bfloat16"],
		default="float32",
		help="Model parameter dtype.",
	)
	parser.add_argument(
		"--mixed-precision",
		action="store_true",
		help="Run forward and loss under BF16 autocast while keeping model parameters at the requested dtype.",
	)
	return parser


def torch_dtype(dtype_name: str) -> torch.dtype:
	dtype_map = {
		"float32": torch.float32,
		"float16": torch.float16,
		"bfloat16": torch.bfloat16,
	}
	return dtype_map[dtype_name]


def _sync_cuda(device: torch.device) -> None:
	if device.type == "cuda":
		torch.cuda.synchronize(device)


def autocast_context(device: torch.device, mixed_precision: bool):
	if not mixed_precision:
		return nullcontext()
	if device.type != "cuda":
		raise ValueError("BF16 mixed precision requires CUDA for this benchmark script.")
	return torch.autocast(device_type=device.type, dtype=torch.bfloat16)


def load_dataset(dataset_path: str) -> np.ndarray:
	path = Path(dataset_path)
	if not path.exists():
		raise FileNotFoundError(f"dataset path does not exist: {dataset_path}")

	if path.suffix == ".npy":
		dataset = np.load(path)
	elif path.suffix == ".pt":
		loaded = torch.load(path, map_location="cpu")
		if isinstance(loaded, torch.Tensor):
			dataset = loaded.detach().cpu().numpy()
		else:
			raise ValueError(".pt dataset must contain a single torch.Tensor of token ids")
	else:
		raise ValueError("unsupported dataset format; expected .npy or .pt")

	if dataset.ndim != 1:
		dataset = dataset.reshape(-1)

	if not np.issubdtype(dataset.dtype, np.integer):
		dataset = dataset.astype(np.int64)

	return dataset


def make_batch(
	batch_size: int,
	context_length: int,
	vocab_size: int,
	device: torch.device,
	dataset: np.ndarray | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
	if dataset is None:
		dataset_len = max(batch_size * context_length * 4, context_length + 2)
		dataset = np.random.randint(0, vocab_size, size=(dataset_len,), dtype=np.int64)

	if len(dataset) <= context_length + 1:
		raise ValueError("dataset must contain at least context_length + 2 tokens")

	return get_batch(dataset, batch_size=batch_size, context_length=context_length, device=str(device))


def validate_inputs(mode: str, device: str, dtype: str, mixed_precision: bool) -> tuple[torch.device, torch.dtype]:
	if mode not in VALID_MODES:
		raise ValueError("benchmark() supports only single modes: forward, forward-backward, forward-backward-optimizer")

	_device = torch.device(device)
	_dtype = torch_dtype(dtype)

	if mixed_precision and _dtype != torch.float32:
		raise ValueError("BF16 mixed precision expects float32 model parameters. Use --dtype float32 with --mixed-precision.")
	if _device.type == "cpu" and _dtype in (torch.float16, torch.bfloat16):
		raise ValueError("float16/bfloat16 are only recommended with CUDA for this benchmark script.")

	return _device, _dtype


def run_benchmark(
	*,
	vocab_size: int,
	context_length: int,
	d_model: int,
	num_layers: int,
	num_heads: int,
	d_ff: int,
	rope_theta: float,
	batch_size: int,
	mode: str,
	device: str,
	dtype: str,
	mixed_precision: bool,
	warmup_steps: int,
	timed_steps: int,
	run_step: BenchmarkStepFn,
	warmup_step: BenchmarkStepFn | None = None,
	optimizer_factory: Callable[[Any], Any] | None = None,
	dataset_path: str | None = None,
	memory_profiler_filename: str | None = None,
) -> dict:
	_device, _dtype = validate_inputs(mode, device, dtype, mixed_precision)

	model = BasicsTransformerLM(
		vocab_size=vocab_size,
		context_length=context_length,
		d_model=d_model,
		num_layers=num_layers,
		num_heads=num_heads,
		d_ff=d_ff,
		rope_theta=rope_theta,
	).to(device=_device, dtype=_dtype)
	model.train(mode != "forward")

	optimizer = optimizer_factory(model.parameters()) if mode == "forward-backward-optimizer" and optimizer_factory else None
	if mode == "forward-backward-optimizer" and optimizer is None:
		raise ValueError("optimizer mode requires an optimizer factory")

	dataset = load_dataset(dataset_path) if dataset_path else None
	x, y = make_batch(
		batch_size=batch_size,
		context_length=context_length,
		vocab_size=vocab_size,
		device=_device,
		dataset=dataset,
	)

	step_for_warmup = warmup_step or run_step
	with autocast_context(_device, mixed_precision):
		for _ in range(warmup_steps):
			step_for_warmup(model, x, y, mode, optimizer)

	if memory_profiler_filename is not None:
		torch.cuda.memory._record_memory_history(max_entries=100000)

	step_times_ms: list[float] = []
	with autocast_context(_device, mixed_precision):
		for _ in range(timed_steps):
			t0 = timeit.default_timer()
			run_step(model, x, y, mode, optimizer)
			_sync_cuda(_device)
			t1 = timeit.default_timer()
			step_times_ms.append((t1 - t0) * 1000.0)

	if memory_profiler_filename is not None:
		torch.cuda.memory._dump_snapshot(memory_profiler_filename)
		torch.cuda.memory._record_memory_history(enabled=None)

	times = np.array(step_times_ms)
	avg_ms = float(np.mean(times))
	std_ms = float(np.std(times, ddof=1)) if timed_steps > 1 else 0.0
	elapsed_s = float(np.sum(times)) / 1000.0
	tokens_per_step = batch_size * context_length
	tokens_per_second = (tokens_per_step * timed_steps) / elapsed_s

	return {
		"mode":              mode,
		"device":            str(_device),
		"dtype":             str(_dtype),
		"mixed_precision":   mixed_precision,
		"warmup_steps":      warmup_steps,
		"timed_steps":       timed_steps,
		"total_time_s":      round(elapsed_s, 6),
		"avg_step_time_ms":  round(avg_ms, 3),
		"std_step_time_ms":  round(std_ms, 3),
		"tokens_per_second": round(tokens_per_second, 2),
		"memory_profiler_filename": memory_profiler_filename,
	}