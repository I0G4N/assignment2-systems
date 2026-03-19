from __future__ import annotations

import argparse
import timeit
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from cs336_basics.data import get_batch
from cs336_basics.model import BasicsTransformerLM


def _parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Benchmark forward-only or forward+backward steps for BasicsTransformerLM."
	)
	parser.add_argument("--vocab-size", type=int, default=50257)
	parser.add_argument("--context-length", type=int, default=1024)
	parser.add_argument("--d-model", type=int, default=768)
	parser.add_argument("--num-layers", type=int, default=12)
	parser.add_argument("--num-heads", type=int, default=12)
	parser.add_argument("--d-ff", type=int, default=3072)
	parser.add_argument("--rope-theta", type=float, default=10000.0)
	parser.add_argument("--batch-size", type=int, default=8)
	parser.add_argument(
		"--dataset-path",
		type=str,
		default=None,
		help="Optional path to tokenized dataset (.npy or .pt). If omitted, random tokens are used.",
	)
	parser.add_argument("--warmup-steps", type=int, default=10)
	parser.add_argument("--timed-steps", type=int, default=50)
	parser.add_argument(
		"--mode",
		choices=["forward", "forward-backward"],
		default="forward-backward",
		help="Benchmark just forward pass or full forward + backward pass.",
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
	return parser.parse_args()


def _torch_dtype(dtype_name: str) -> torch.dtype:
	dtype_map = {
		"float32": torch.float32,
		"float16": torch.float16,
		"bfloat16": torch.bfloat16,
	}
	return dtype_map[dtype_name]


def _sync_cuda(device: torch.device) -> None:
	if device.type == "cuda":
		torch.cuda.synchronize(device)


def _load_dataset(dataset_path: str) -> np.ndarray:
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


def _make_batch(
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


def _run_step(model: BasicsTransformerLM, x: torch.Tensor, y: torch.Tensor, mode: str) -> None:
	if mode == "forward":
		with torch.no_grad():
			_ = model(x)
		return

	logits = model(x)
	loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
	loss.backward()
	model.zero_grad(set_to_none=True)


def benchmark(
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
	warmup_steps: int,
	timed_steps: int,
	dataset_path: str | None = None,
) -> dict:
	"""Run the benchmark and return a dict of timing metrics."""
	_device = torch.device(device)
	_dtype = _torch_dtype(dtype)

	if _device.type == "cpu" and _dtype in (torch.float16, torch.bfloat16):
		raise ValueError("float16/bfloat16 are only recommended with CUDA for this benchmark script.")

	model = BasicsTransformerLM(
		vocab_size=vocab_size,
		context_length=context_length,
		d_model=d_model,
		num_layers=num_layers,
		num_heads=num_heads,
		d_ff=d_ff,
		rope_theta=rope_theta,
	).to(device=_device, dtype=_dtype)
	model.train(mode=mode == "forward-backward")

	dataset = _load_dataset(dataset_path) if dataset_path else None

	x, y = _make_batch(
		batch_size=batch_size,
		context_length=context_length,
		vocab_size=vocab_size,
		device=_device,
		dataset=dataset,
	)

	for _ in range(warmup_steps):
		_run_step(model, x, y, mode)
		_sync_cuda(_device)

	step_times_ms: list[float] = []
	for _ in range(timed_steps):
		t0 = timeit.default_timer()
		_run_step(model, x, y, mode)
		_sync_cuda(_device)
		t1 = timeit.default_timer()
		step_times_ms.append((t1 - t0) * 1000.0)

	times = np.array(step_times_ms)
	avg_ms = float(np.mean(times))
	std_ms = float(np.std(times, ddof=1))
	elapsed_s = float(np.sum(times)) / 1000.0
	tokens_per_step = batch_size * context_length
	tokens_per_second = (tokens_per_step * timed_steps) / elapsed_s

	return {
		"mode":              mode,
		"device":            str(_device),
		"dtype":             str(_dtype),
		"warmup_steps":      warmup_steps,
		"timed_steps":       timed_steps,
		"total_time_s":      round(elapsed_s, 6),
		"avg_step_time_ms":  round(avg_ms, 3),
		"std_step_time_ms":  round(std_ms, 3),
		"tokens_per_second": round(tokens_per_second, 2),
	}


def main() -> None:
	args = _parse_args()
	results = benchmark(
		vocab_size=args.vocab_size,
		context_length=args.context_length,
		d_model=args.d_model,
		num_layers=args.num_layers,
		num_heads=args.num_heads,
		d_ff=args.d_ff,
		rope_theta=args.rope_theta,
		batch_size=args.batch_size,
		mode=args.mode,
		device=args.device,
		dtype=args.dtype,
		warmup_steps=args.warmup_steps,
		timed_steps=args.timed_steps,
		dataset_path=args.dataset_path,
	)
	for key, val in results.items():
		print(f"{key}: {val}")


if __name__ == "__main__":
	main()



