from __future__ import annotations

import argparse
import torch
import torch.nn.functional as F
import torch.cuda.nvtx as nvtx

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW
from cs336_systems.benchmark.shared import autocast_context, run_benchmark



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
	parser.add_argument(
		"--mixed-precision",
		action="store_true",
		help="Run forward and loss under BF16 autocast while keeping model parameters at the requested dtype.",
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
