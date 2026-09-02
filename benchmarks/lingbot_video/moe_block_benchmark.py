# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Benchmark one LingBot-Video MoE block at the measured production shape."""

from __future__ import annotations

import argparse
import inspect
import json
import statistics
import time
from contextlib import contextmanager
from pathlib import Path

import torch
from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
from vllm.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.utils.network_utils import (
    get_distributed_init_method,
    get_ip,
    get_open_port,
)
from vllm.utils.torch_utils import set_default_torch_dtype
from vllm.v1.worker.workspace import init_workspace_manager

from vllm_omni.diffusion.models.lingbot_video.lingbot_video_transformer import (
    LingBotVideoSparseMoeBlock,
)


@contextmanager
def _single_rank_runtime(vllm_config: VllmConfig):
    with (
        set_current_vllm_config(vllm_config),
        set_default_torch_dtype(torch.bfloat16),
    ):
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method=get_distributed_init_method(
                get_ip(),
                get_open_port(),
            ),
            backend="nccl",
        )
        initialize_model_parallel()
        init_workspace_manager(torch.device("cuda"))
        try:
            yield
        finally:
            destroy_model_parallel()
            destroy_distributed_environment()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-tokens", type=int, default=49145)
    parser.add_argument("--padding-tokens", type=int, default=0)
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--num-experts", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--intermediate-size", type=int, default=768)
    parser.add_argument("--num-expert-groups", type=int, default=4)
    parser.add_argument("--top-k-groups", type=int, default=2)
    parser.add_argument("--route-scale", type=float, default=2.5)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-tensor", type=Path, default=None)
    parser.add_argument("--routing-artifact", type=Path, default=None)
    return parser.parse_args()


def _initialize_block(block: LingBotVideoSparseMoeBlock, seed: int) -> str:
    def generator(offset: int) -> torch.Generator:
        return torch.Generator(device="cuda").manual_seed(seed + offset)

    def initialize(parameter: torch.Tensor, offset: int) -> None:
        values = torch.empty(
            parameter.shape,
            dtype=parameter.dtype,
            device=parameter.device,
        )
        values.normal_(
            mean=0.0,
            std=0.02,
            generator=generator(offset),
        )
        parameter.copy_(values)

    runner = block.experts
    with torch.no_grad():
        if hasattr(runner, "routed_experts"):
            runner.gate.to(dtype=torch.float32)
            correction_bias = runner.routed_experts.e_score_correction_bias
            correction_bias.data = correction_bias.data.float()
            initialize(runner.gate.weight, 0)
            correction_bias.zero_()
            intermediate_size = runner.routed_experts.w13_weight.shape[1] // 2
            initialize(
                runner.routed_experts.w13_weight[:, :intermediate_size],
                1,
            )
            initialize(runner.routed_experts.w2_weight, 2)
            initialize(
                runner.routed_experts.w13_weight[:, intermediate_size:],
                3,
            )
            runner.routed_experts.quant_method.process_weights_after_loading(runner.routed_experts)
            backend = "common_fused_moe"
        else:
            block.router.to(dtype=torch.float32)
            initialize(block.router.weight, 0)
            block.router.e_score_correction_bias.zero_()
            initialize(runner.w1, 1)
            initialize(runner.w2, 2)
            initialize(runner.w3, 3)
            backend = "torch_grouped_mm"
        if block.shared_experts is not None:
            for offset, parameter in enumerate(
                block.shared_experts.parameters(),
                start=4,
            ):
                initialize(parameter, offset)
    return backend


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((len(ordered) - 1) * fraction)))
    return ordered[index]


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("LingBot MoE block benchmark requires CUDA.")
    if args.padding_tokens < 0 or args.padding_tokens >= args.num_tokens:
        raise ValueError("--padding-tokens must be in [0, num_tokens).")

    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    vllm_config = VllmConfig(device_config=DeviceConfig(device="cuda"))

    with _single_rank_runtime(vllm_config):
        block_kwargs: dict[str, object] = {
            "hidden_size": args.hidden_size,
            "num_experts": args.num_experts,
            "top_k": args.top_k,
            "moe_intermediate_size": args.intermediate_size,
            "score_func": "sigmoid",
            "norm_topk_prob": True,
            "n_group": args.num_expert_groups,
            "topk_group": args.top_k_groups,
            "routed_scaling_factor": args.route_scale,
            "n_shared_experts": 1,
        }
        if "prefix" in inspect.signature(LingBotVideoSparseMoeBlock.__init__).parameters:
            block_kwargs["prefix"] = "benchmark.lingbot_moe"
        block = LingBotVideoSparseMoeBlock(
            **block_kwargs,
        ).to(device=device, dtype=torch.bfloat16)
        backend = _initialize_block(block, args.seed)
        hidden_states = torch.randn(
            1,
            args.num_tokens,
            args.hidden_size,
            generator=torch.Generator(device=device).manual_seed(
                args.seed + 100,
            ),
            device=device,
            dtype=torch.bfloat16,
        )
        padding_mask = None
        if args.padding_tokens:
            padding_mask = torch.ones(
                args.num_tokens,
                device=device,
                dtype=torch.float32,
            )
            padding_mask[-args.padding_tokens :] = 0

        routing_artifact = None
        if args.routing_artifact is not None:
            tokens = hidden_states.reshape(-1, args.hidden_size)
            if padding_mask is not None:
                tokens = tokens.index_select(
                    0,
                    torch.where(padding_mask.bool())[0],
                )
            with torch.inference_mode():
                if hasattr(block.experts, "routed_experts"):
                    router_logits, _ = block.experts.gate(tokens)
                    top_scores, top_indices = block.experts.router.select_experts(
                        hidden_states=tokens,
                        router_logits=router_logits,
                        topk_indices_dtype=None,
                    )
                    gate_weight = block.experts.gate.weight
                else:
                    router_logits = torch.nn.functional.linear(
                        tokens.float(),
                        block.router.weight.float(),
                    )
                    top_indices, top_scores = block.router(tokens)
                    gate_weight = block.router.weight
            routing_artifact = {
                "gate_weight": gate_weight.cpu(),
                "router_logits": router_logits.cpu(),
                "top_indices": top_indices.cpu(),
                "top_scores": top_scores.cpu(),
            }

        def run_once() -> torch.Tensor:
            with torch.inference_mode():
                return block(hidden_states, padding_mask=padding_mask)

        for _ in range(args.warmups):
            output = run_once()
        torch.accelerator.synchronize()
        torch.accelerator.empty_cache()
        torch.accelerator.reset_peak_memory_stats(device)

        durations_ms: list[float] = []
        start_wall = time.perf_counter()
        for _ in range(args.repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            output = run_once()
            end.record()
            end.synchronize()
            durations_ms.append(float(start.elapsed_time(end)))
        wall_seconds = time.perf_counter() - start_wall

    result = {
        "backend": backend,
        "fast_path": {
            "runner": type(block.experts).__name__,
            "routed_experts": type(getattr(block.experts, "routed_experts", block.experts)).__name__,
            "quant_method": type(
                getattr(
                    getattr(block.experts, "routed_experts", None),
                    "quant_method",
                    None,
                )
            ).__name__,
        },
        "shape": {
            "num_tokens": args.num_tokens,
            "valid_tokens": args.num_tokens - args.padding_tokens,
            "hidden_size": args.hidden_size,
            "num_experts": args.num_experts,
            "top_k": args.top_k,
            "intermediate_size": args.intermediate_size,
            "num_expert_groups": args.num_expert_groups,
            "top_k_groups": args.top_k_groups,
            "route_scale": args.route_scale,
        },
        "measurement": {
            "warmups": args.warmups,
            "repeats": args.repeats,
            "median_ms": statistics.median(durations_ms),
            "p95_ms": _percentile(durations_ms, 0.95),
            "mean_ms": statistics.mean(durations_ms),
            "stdev_ms": statistics.stdev(durations_ms) if len(durations_ms) > 1 else 0.0,
            "min_ms": min(durations_ms),
            "max_ms": max(durations_ms),
            "wall_seconds": wall_seconds,
            "peak_allocated_mb": torch.accelerator.max_memory_allocated(device) / 1_000_000,
            "peak_reserved_mb": torch.accelerator.max_memory_reserved(device) / 1_000_000,
        },
        "output": {
            "shape": list(output.shape),
            "finite": bool(torch.isfinite(output).all()),
            "checksum": float(output.float().sum()),
        },
    }
    payload = json.dumps(result, indent=2, sort_keys=True)
    print(payload)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(payload + "\n", encoding="utf-8")
    if args.output_tensor is not None:
        args.output_tensor.parent.mkdir(parents=True, exist_ok=True)
        torch.save(output.detach().cpu(), args.output_tensor)
    if args.routing_artifact is not None:
        assert routing_artifact is not None
        args.routing_artifact.parent.mkdir(parents=True, exist_ok=True)
        torch.save(routing_artifact, args.routing_artifact)
    return 0 if result["output"]["finite"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
