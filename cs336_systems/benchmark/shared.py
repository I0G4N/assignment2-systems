from __future__ import annotations

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


def torch_dtype(dtype_name: str) -> torch.dtype:
	dtype_map = {
		"float32": torch.float32,
		"float16": torch.float16,
		"bfloat16": torch.bfloat16,
	}
	return dtype_map[dtype_name]


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
	for _ in range(warmup_steps):
		step_for_warmup(model, x, y, mode, _device, mixed_precision, optimizer)

	step_times_ms: list[float] = []
	for _ in range(timed_steps):
		t0 = timeit.default_timer()
		run_step(model, x, y, mode, _device, mixed_precision, optimizer)
		t1 = timeit.default_timer()
		step_times_ms.append((t1 - t0) * 1000.0)

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
	}