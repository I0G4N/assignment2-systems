from __future__ import annotations

import argparse
import torch
import torch.nn.functional as F
import torch.cuda.nvtx as nvtx

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW
from cs336_systems.benchmark.shared import autocast_context, build_benchmark_arg_parser, run_benchmark



def _parse_args() -> argparse.Namespace:
	parser = build_benchmark_arg_parser(
		description="Nsys profile forward, forward-backward or forward-backward-optimizer steps for BasicsTransformerLM.",
		mode_default="forward-backward-optimizer",
	)
	return parser.parse_args()


@nvtx.range("run_step")
def _run_step(
	model: BasicsTransformerLM,
	x: torch.Tensor,
	y: torch.Tensor,
	mode: str,
	device: torch.device,
	mixed_precision: bool,
	optimizer: AdamW | None = None,
) -> None:
	if mode == "forward-backward-optimizer":
		optimizer.zero_grad(set_to_none=True)
	with nvtx.range("model_forward"):
		with autocast_context(device, mixed_precision):
			logits = model(x)
	if mode != "forward":
		with nvtx.range("loss_backward"):
			with autocast_context(device, mixed_precision):
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
	mixed_precision: bool,
	optimizer: AdamW | None = None,
) -> None:
	if mode == "forward-backward-optimizer":
		optimizer.zero_grad(set_to_none=True)
	with autocast_context(device, mixed_precision):
		logits = model(x)
	if mode != "forward":
		with autocast_context(device, mixed_precision):
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
		warmup_step=_warmup_step,
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
