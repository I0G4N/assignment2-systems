from __future__ import annotations

import argparse
import torch
import torch.nn.functional as F

from cs336_basics.model import BasicsTransformerLM
from cs336_systems.benchmark.shared import autocast_context, build_benchmark_arg_parser, run_benchmark
from torch.optim import AdamW


def _parse_args() -> argparse.Namespace:
	parser = build_benchmark_arg_parser(
		description="Benchmark forward, forward-backward, forward-backward-optimizer steps for BasicsTransformerLM.",
		mode_default="forward-backward",
	)
	return parser.parse_args()


def _sync_cuda(device: torch.device) -> None:
	if device.type == "cuda":
		torch.cuda.synchronize(device)


def _run_step(
	model: BasicsTransformerLM,
	x: torch.Tensor,
	y: torch.Tensor,
	mode: str,
	device: torch.device,
	mixed_precision: bool,
	optimizer: AdamW | None = None,
) -> None:
	if mode == "forward":
		with torch.no_grad():
			with autocast_context(device, mixed_precision):
				_ = model(x)
		_sync_cuda(device)
		return

	if mode == "forward-backward":
		with autocast_context(device, mixed_precision):
			logits = model(x)
			_sync_cuda(device)
			loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
			_sync_cuda(device)
		loss.backward()
		_sync_cuda(device)
		model.zero_grad(set_to_none=True)
		return

	if mode == "forward-backward-optimizer":
		if optimizer is None:
			raise ValueError("optimizer mode requires an AdamW optimizer")
		optimizer.zero_grad(set_to_none=True)
		with autocast_context(device, mixed_precision):
			logits = model(x)
			_sync_cuda(device)
			loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
			_sync_cuda(device)
		loss.backward()
		_sync_cuda(device)
		optimizer.step()
		_sync_cuda(device)
		return

	raise ValueError(f"unsupported mode: {mode}")


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
	mixed_precision: bool,
	warmup_steps: int,
	timed_steps: int,
	dataset_path: str | None = None,
) -> dict:
	"""Run the benchmark and return a dict of timing metrics."""
	return run_benchmark(
		vocab_size=vocab_size,
		context_length=context_length,
		d_model=d_model,
		num_layers=num_layers,
		num_heads=num_heads,
		d_ff=d_ff,
		rope_theta=rope_theta,
		batch_size=batch_size,
		mode=mode,
		device=device,
		dtype=dtype,
		mixed_precision=mixed_precision,
		warmup_steps=warmup_steps,
		timed_steps=timed_steps,
		run_step=_run_step,
		optimizer_factory=lambda params: AdamW(params, lr=1e-3),
		dataset_path=dataset_path,
	)


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
		mixed_precision=args.mixed_precision,
		warmup_steps=args.warmup_steps,
		timed_steps=args.timed_steps,
		dataset_path=args.dataset_path,
	)
	for key, val in results.items():
		print(f"{key}: {val}")


if __name__ == "__main__":
	main()
