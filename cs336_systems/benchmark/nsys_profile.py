from __future__ import annotations

import argparse
import timeit
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.cuda.nvtx as nvtx

from cs336_basics import optimizer
from cs336_basics.data import get_batch
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW



def _parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Nsys profile forward, forward-backward or forward-backward-optimizer steps for BasicsTransformerLM."
	)
	parser.add_argument("--vocab-size", type=int, default=10000)
	parser.add_argument("--context-length", type=int, default=128)
	parser.add_argument("--d-model", type=int, default=1024)
	parser.add_argument("--num-layers", type=int, default=24)
	parser.add_argument("--num-heads", type=int, default=16)
	parser.add_argument("--d-ff", type=int, default=4096)
	parser.add_argument("--rope-theta", type=float, default=10000.0)
	parser.add_argument("--batch-size", type=int, default=4)
	parser.add_argument(
		"--dataset-path",
		type=str,
		default=None,
		help="Optional path to tokenized dataset (.npy or .pt). If omitted, random tokens are used.",
	)
	parser.add_argument("--warmup-steps", type=int, default=5)
	parser.add_argument("--timed-steps", type=int, default=5)
	parser.add_argument(
		"--mode",
		choices=["forward", "forward-backward", "forward-backward-optimizer"],
		default="forward-backward-optimizer",
		help="Benchmark mode: forward, forward-backward, or forward-backward-optimizer.",
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


@nvtx.range("run_step")
def _run_step(
	model: BasicsTransformerLM,
	x: torch.Tensor,
	y: torch.Tensor,
	mode: str,
	device: torch.device,
	optimizer: AdamW | None = None,
) -> None:
	if mode == "forward-backward-optimizer":
		optimizer.zero_grad(set_to_none=True)
	with nvtx.range("model_forward"):
		logits = model(x)
	if mode != "forward":
		with nvtx.range("loss_backward"):
			loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
			loss.backward()
	if mode == "forward-backward-optimizer":
		with nvtx.range("optimizer_step"):
			optimizer.step()
	return


def _warmup_step(
	model: BasicsTransformerLM,
	x: torch.Tensor,
	y: torch.Tensor,
	mode: str,
	device: torch.device,
	optimizer: AdamW | None = None,
) -> None:
	if mode == "forward-backward-optimizer":
		optimizer.zero_grad(set_to_none=True)
	logits = model(x)
	if mode != "forward":
		loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
		loss.backward()
	if mode == "forward-backward-optimizer":
		optimizer.step()
	return


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
	if mode not in {"forward", "forward-backward", "forward-backward-optimizer"}:
		raise ValueError("benchmark() supports only single modes: forward, forward-backward, forward-backward-optimizer")

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
	model.train(mode != "forward")

	optimizer = AdamW(model.parameters(), lr=1e-3) if mode == "forward-backward-optimizer" else None

	dataset = _load_dataset(dataset_path) if dataset_path else None

	x, y = _make_batch(
		batch_size=batch_size,
		context_length=context_length,
		vocab_size=vocab_size,
		device=_device,
		dataset=dataset,
	)

	for _ in range(warmup_steps):
		_warmup_step(model, x, y, mode, _device, optimizer=optimizer)

	step_times_ms: list[float] = []
	for _ in range(timed_steps):
		t0 = timeit.default_timer()
		_run_step(model, x, y, mode, _device, optimizer=optimizer)
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
