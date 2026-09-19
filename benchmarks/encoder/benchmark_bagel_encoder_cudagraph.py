# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Measure BAGEL's fixed-resolution encoder, including graph dispatch/copies.

This loads only ViT and connector weights. It does not measure LM execution,
TTFT, or serving throughput. Use --arm eager / graph in separate processes
for independent validation; --arm both is a shared-weight screening run.
"""

import argparse
import hashlib
import inspect
import json
import os
import statistics
import tempfile
import time
from pathlib import Path

import torch
from safetensors import safe_open
from torch import nn
from vllm import _version
from vllm.config import CompilationConfig, ModelConfig, SchedulerConfig, VllmConfig, set_current_vllm_config
from vllm.distributed import cleanup_dist_env_and_memory, init_distributed_environment, initialize_model_parallel
from vllm.model_executor.models.bagel import BagelVisionMLP, PositionEmbedding
from vllm.model_executor.models.siglip import SiglipVisionModel
from vllm.utils.torch_utils import set_default_torch_dtype
from vllm.v1.worker.encoder_cudagraph import EncoderCudaGraphManager

from vllm_omni.model_executor.models.bagel.bagel import OmniBagelForConditionalGeneration
from vllm_omni.platforms import current_omni_platform


def load_encoder(config: VllmConfig, weights: list[Path]) -> OmniBagelForConditionalGeneration:
    """Build the same encoder modules as BAGEL, without its language model/VAE."""
    hf_config = config.model_config.hf_config
    # Match upstream BAGEL's correction for the released checkpoint.
    if hf_config.vit_config.num_hidden_layers == 27:
        hf_config.vit_config.num_hidden_layers = 26
    model = OmniBagelForConditionalGeneration.__new__(OmniBagelForConditionalGeneration)
    nn.Module.__init__(model)
    model.config = hf_config
    model.vit_model = SiglipVisionModel(hf_config.vit_config)
    hidden = hf_config.llm_config.hidden_size
    model.connector = BagelVisionMLP(hf_config.vit_config.hidden_size, hidden, hidden, hf_config.connector_act)
    model.vit_pos_embed = PositionEmbedding(hf_config.vit_max_num_patch_per_side, hidden)
    model.to(device=current_omni_platform.get_torch_device(), dtype=config.model_config.dtype).eval()
    loaded = set()
    for path in weights:
        with safe_open(str(path), framework="pt", device="cpu") as checkpoint:
            loaded.update(
                model.load_weights(
                    (key, checkpoint.get_tensor(key))
                    for key in checkpoint.keys()
                    if key.startswith(("vit_model.", "connector."))
                )
            )
    missing = set(dict(model.named_parameters())) - loaded
    if missing:
        raise ValueError(f"Encoder weights missing from checkpoint: {sorted(missing)}")
    return model


def percentile(values: list[float], fraction: float) -> float:
    values = sorted(values)
    index = (len(values) - 1) * fraction
    lower = int(index)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (index - lower)


@torch.inference_mode()
def benchmark(args: argparse.Namespace, config: VllmConfig) -> dict:
    model = load_encoder(config, args.weights)
    manager = (
        EncoderCudaGraphManager(config, current_omni_platform.get_torch_device(), config.model_config.dtype, model)
        if args.arm != "eager"
        else None
    )
    size = model.config.vit_config.image_size
    device = model.vit_model.device
    generator = torch.Generator(device=device).manual_seed(args.seed)
    inputs = {
        batch: {
            "pixel_values": torch.randn(
                batch, 3, size, size, device=device, dtype=model.vit_model.dtype, generator=generator
            )
        }
        for batch in args.batch_sizes
    }
    source = Path(inspect.getfile(type(model)))
    report = {
        "scope": "encoder_only",
        "arm": args.arm,
        "pid": os.getpid(),
        "seed": args.seed,
        "vllm": _version.__version__,
        "vllm_commit": _version.__commit_id__,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": current_omni_platform.get_device_name(),
        "source": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "weights": [str(path) for path in args.weights],
        "image_size": size,
        "attention_backends": sorted(
            {str(module.attn_backend) for module in model.modules() if hasattr(module, "attn_backend")}
        ),
        "warmup_per_arm": args.warmup,
        "repetitions_per_arm": args.repetitions,
        "batches": {},
    }
    try:
        for kwargs in inputs.values():
            for _ in range(args.warmup):
                model._process_image_input(kwargs)
        if args.arm != "eager":
            manager.capture(torch.cuda.graph_pool_handle())
        captures = manager.get_cumulative_stats()["num_budgets"] if manager is not None else 0
        for batch, kwargs in inputs.items():
            expected = model._process_image_input(kwargs)
            max_abs = None
            if args.arm != "eager":
                actual = manager.execute(kwargs)
                for output, reference in zip(actual, expected, strict=True):
                    torch.testing.assert_close(output, reference, atol=0.02, rtol=0.02)
                max_abs = max((a.float() - b.float()).abs().max().item() for a, b in zip(actual, expected, strict=True))
            order = (
                ["eager", "graph", "graph", "eager", "graph", "eager", "eager", "graph"]
                if args.arm == "both"
                else [args.arm] * 4
            )
            rows = []
            for arm in order:
                forward = model._process_image_input if arm == "eager" else manager.execute
                for _ in range(args.warmup):
                    forward(kwargs)
                current_omni_platform.synchronize()
                times = []
                for _ in range(args.repetitions):
                    start = time.perf_counter()
                    output = forward(kwargs)
                    current_omni_platform.synchronize()
                    times.append(1000 * (time.perf_counter() - start))
                    del output
                rows.append(
                    {
                        "arm": arm,
                        "ms": times,
                        "mean_ms": statistics.mean(times),
                        "p50_ms": percentile(times, 0.5),
                        "p90_ms": percentile(times, 0.9),
                        "p99_ms": percentile(times, 0.99),
                    }
                )
            report["batches"][batch] = {"max_abs": max_abs, "rows": rows}
            print(f"batch={batch}: {[(row['arm'], row['p50_ms']) for row in rows]}", flush=True)
        report["graph_stats"] = manager.get_cumulative_stats() if manager is not None else None
        if manager is not None:
            assert report["graph_stats"]["num_budgets"] == captures
        report["peak_allocated_bytes"] = torch.accelerator.memory.max_memory_allocated()
        report["peak_reserved_bytes"] = torch.accelerator.memory.max_memory_reserved()
        return report
    finally:
        if manager is not None:
            manager.clear()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Local BAGEL directory containing config and tokenizer metadata")
    parser.add_argument(
        "--weights", type=Path, nargs="+", required=True, help="Safetensors files containing ViT/connector"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=["both", "eager", "graph"], default="both")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if min(args.batch_sizes) < 1 or args.warmup < 1 or args.repetitions < 1:
        parser.error("batch sizes, warmup and repetitions must be positive")
    model_config = ModelConfig(model=args.model, skip_tokenizer_init=True, dtype="bfloat16", max_model_len=8192)
    tokens = (model_config.hf_config.vit_config.image_size // model_config.hf_config.vit_config.patch_size) ** 2
    config = VllmConfig(
        model_config=model_config,
        scheduler_config=SchedulerConfig(
            max_model_len=model_config.max_model_len,
            is_encoder_decoder=model_config.is_encoder_decoder,
            max_num_seqs=max(args.batch_sizes),
            max_num_batched_tokens=tokens * max(args.batch_sizes),
        ),
        compilation_config=CompilationConfig(
            encoder_cudagraph_token_budgets=[tokens * b for b in sorted(set(args.batch_sizes))],
            encoder_cudagraph_max_vision_items_per_batch=max(args.batch_sizes),
        ),
    )
    with (
        tempfile.TemporaryDirectory() as directory,
        set_current_vllm_config(config),
        set_default_torch_dtype(model_config.dtype),
    ):
        init_distributed_environment(
            world_size=1, rank=0, distributed_init_method=f"file://{directory}/dist", local_rank=0
        )
        initialize_model_parallel(tensor_model_parallel_size=1)
        try:
            report = benchmark(args, config)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2))
        finally:
            cleanup_dist_env_and_memory()


if __name__ == "__main__":
    main()
