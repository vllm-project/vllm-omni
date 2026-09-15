# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Reproduce LingBot-Video MoE block and transformer numerical parity."""

from __future__ import annotations

import argparse
import gc
import importlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scope",
        choices=("block", "transformer", "all"),
        default="block",
        help="Run the lightweight sparse-block check, the real transformer check, or both.",
    )
    parser.add_argument(
        "--official-repo",
        type=Path,
        required=True,
        help="Local checkout of https://github.com/Robbyant/lingbot-video.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Local MoE checkpoint directory.",
    )
    parser.add_argument("--transformer-subfolder", default="transformer")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-frames", type=int, default=1)
    parser.add_argument("--height", type=int, default=8)
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--text-length", type=int, default=16)
    return parser.parse_args()


def _load_official_module(official_repo: Path):
    transformer_file = official_repo / "lingbot_video" / "transformer_lingbot_video.py"
    if not transformer_file.is_file():
        raise FileNotFoundError(f"Upstream transformer not found: {transformer_file}")
    sys.path.insert(0, str(official_repo))
    module = importlib.import_module("lingbot_video.transformer_lingbot_video")
    module_path = Path(module.__file__).resolve()
    if official_repo.resolve() not in module_path.parents:
        raise RuntimeError(f"Imported LingBot module from {module_path}, not {official_repo.resolve()}")
    return module


def _load_native_module():
    return importlib.import_module("vllm_omni.diffusion.models.lingbot_video.lingbot_video_transformer")


def _tensor_metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, Any]:
    actual_float = actual.float()
    expected_float = expected.float()
    diff = (actual_float - expected_float).abs()
    actual_flat = actual_float.flatten().double()
    expected_flat = expected_float.flatten().double()
    denominator = actual_flat.norm() * expected_flat.norm()
    cosine = float(torch.dot(actual_flat, expected_flat) / denominator) if float(denominator) > 0 else 1.0
    expected_norm = expected_flat.norm()
    return {
        "shape": list(actual.shape),
        "equal": bool(torch.equal(actual, expected)),
        "max_abs": float(diff.max()) if diff.numel() else 0.0,
        "mean_abs": float(diff.mean()) if diff.numel() else 0.0,
        "rmse": float(torch.sqrt(torch.mean(diff.square()))) if diff.numel() else 0.0,
        "relative_l2": (
            float((actual_flat - expected_flat).norm() / expected_norm) if float(expected_norm) > 0 else 0.0
        ),
        "cosine": cosine,
        "finite": bool(torch.isfinite(actual).all() and torch.isfinite(expected).all()),
    }


def _initialize_block(module: torch.nn.Module, seed: int) -> None:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        for parameter in module.parameters():
            values = torch.randn(parameter.shape, generator=generator, dtype=torch.float32)
            parameter.copy_((values * 0.02).to(parameter.dtype))
        module.router.e_score_correction_bias.copy_(torch.tensor([0.9, 0.8, 1.0, 0.0, 0.7, 0.6, 0.5, 0.4]))


def _copy_block_weights_to_common_runner(official, native) -> None:
    runner = native.experts
    routed_experts = runner.routed_experts
    intermediate_size = official.experts.w1.shape[1]
    with torch.no_grad():
        runner.gate.weight.copy_(official.router.weight)
        routed_experts.e_score_correction_bias.copy_(official.router.e_score_correction_bias)
        routed_experts.w13_weight[:, :intermediate_size].copy_(official.experts.w1)
        routed_experts.w13_weight[:, intermediate_size:].copy_(official.experts.w3)
        routed_experts.w2_weight.copy_(official.experts.w2)
        native.shared_experts.load_state_dict(
            official.shared_experts.state_dict(),
            strict=True,
        )


def _apply_padding(
    scores: torch.Tensor,
    padding_mask: torch.Tensor,
    route_scale: float,
) -> torch.Tensor:
    scores = scores * padding_mask.unsqueeze(-1).to(scores.dtype)
    scores = scores / (scores.sum(dim=-1, keepdim=True) + 1e-9)
    return scores * route_scale


def _run_block_parity(
    official_module,
    native_module,
    *,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    from vllm.utils.torch_utils import set_default_torch_dtype

    common = {
        "hidden_size": 16,
        "num_experts": 8,
        "top_k": 2,
        "moe_intermediate_size": 8,
        "score_func": "sigmoid",
        "norm_topk_prob": True,
        "n_group": 4,
        "topk_group": 1,
        "routed_scaling_factor": 1.5,
        "n_shared_experts": 1,
    }
    official = official_module.LingBotVideoSparseMoeBlock(
        intermediate_size=32,
        **common,
    )
    with set_default_torch_dtype(torch.bfloat16):
        native = native_module.LingBotVideoSparseMoeBlock(**common)
    official = official.to(device=device, dtype=torch.bfloat16).eval()
    native = native.to(device=device, dtype=torch.bfloat16).eval()
    official.router.to(dtype=torch.float32)
    native.experts.gate.to(dtype=torch.float32)
    correction_bias = native.experts.routed_experts.e_score_correction_bias
    correction_bias.data = correction_bias.data.float()
    _initialize_block(official, seed)
    _copy_block_weights_to_common_runner(official, native)
    native.experts.routed_experts.quant_method.process_weights_after_loading(native.experts.routed_experts)

    generator = torch.Generator(device=device).manual_seed(seed)
    hidden_states = torch.randn(
        2,
        5,
        common["hidden_size"],
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    padding_mask = torch.tensor(
        [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        device=device,
        dtype=torch.float32,
    )
    tokens = hidden_states.reshape(-1, common["hidden_size"])
    valid_indices = torch.where(padding_mask.bool())[0]
    valid_tokens = tokens.index_select(0, valid_indices)

    with torch.inference_mode():
        official_router = official.router(tokens)
        official_indices, official_scores = official_router[:2]

        official_scores = _apply_padding(
            official_scores,
            padding_mask,
            official.router.route_scale,
        )
        native_logits, _ = native.experts.gate(valid_tokens)
        native_scores, native_indices = native.experts.router.select_experts(
            hidden_states=valid_tokens,
            router_logits=native_logits,
            topk_indices_dtype=None,
        )
        official_routed = official._run_selected_experts(
            tokens,
            official_scores,
            official_indices,
        )
        native_routed = tokens.new_zeros(tokens.shape)
        native_routed.index_copy_(
            0,
            valid_indices,
            native._run_routed_experts(valid_tokens),
        )
        official_shared = official.shared_experts(hidden_states)
        native_shared = native.shared_experts(hidden_states)
        official_output = official(hidden_states, padding_mask=padding_mask)
        native_output = native(hidden_states, padding_mask=padding_mask)

    official_valid_indices = official_indices.index_select(0, valid_indices)
    official_valid_scores = official_scores.index_select(0, valid_indices).float()
    native_indices, native_order = torch.sort(native_indices.long(), dim=-1)
    official_valid_indices, official_order = torch.sort(
        official_valid_indices.long(),
        dim=-1,
    )
    native_scores = torch.gather(native_scores.float(), 1, native_order)
    official_valid_scores = torch.gather(
        official_valid_scores,
        1,
        official_order,
    )
    result = {
        "router_indices_equal": bool(torch.equal(native_indices, official_valid_indices)),
        "router_scores": _tensor_metrics(
            native_scores,
            official_valid_scores,
        ),
        "routed_output": _tensor_metrics(native_routed, official_routed),
        "shared_output": _tensor_metrics(native_shared, official_shared),
        "final_output": _tensor_metrics(native_output, official_output),
    }
    result["exact"] = bool(
        result["router_indices_equal"]
        and result["router_scores"]["equal"]
        and result["routed_output"]["equal"]
        and result["shared_output"]["equal"]
        and result["final_output"]["equal"]
    )
    result["strict"] = bool(
        result["router_indices_equal"]
        and torch.allclose(
            native_scores,
            official_valid_scores,
            rtol=3e-3,
            atol=2e-4,
        )
        and result["final_output"]["relative_l2"] <= 5e-3
        and result["final_output"]["cosine"] >= 0.99999
        and result["final_output"]["max_abs"] <= 2e-2
    )
    return result


def _release_cuda() -> None:
    gc.collect()
    torch.accelerator.empty_cache()
    torch.accelerator.synchronize()


def _make_transformer_inputs(
    config,
    args: argparse.Namespace,
) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    hidden_states = torch.randn(
        1,
        int(config.in_channels),
        args.num_frames,
        args.height,
        args.width,
        generator=generator,
        dtype=torch.float32,
    ).to(torch.bfloat16)
    encoder_hidden_states = torch.randn(
        1,
        args.text_length,
        int(config.text_dim),
        generator=generator,
        dtype=torch.float32,
    ).to(torch.bfloat16)
    return {
        "hidden_states": hidden_states,
        "timestep": torch.tensor([500.0], dtype=torch.float32),
        "encoder_hidden_states": encoder_hidden_states,
        "encoder_attention_mask": torch.ones(1, args.text_length, dtype=torch.long),
    }


def _load_official_transformer(
    transformer_cls,
    args: argparse.Namespace,
    device: torch.device,
):
    start = time.perf_counter()
    model = transformer_cls.from_pretrained(
        args.model,
        subfolder=args.transformer_subfolder,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    model = model.to(device=device, dtype=torch.bfloat16).eval()
    torch.accelerator.synchronize()
    return model, time.perf_counter() - start


def _load_native_transformer(
    transformer_cls,
    args: argparse.Namespace,
    device: torch.device,
):
    from safetensors import safe_open

    from vllm_omni.diffusion.data import TransformerConfig
    from vllm_omni.diffusion.utils.tf_utils import get_transformer_config_kwargs

    transformer_dir = Path(args.model) / args.transformer_subfolder
    config_path = transformer_dir / "config.json"
    index_path = transformer_dir / "diffusion_pytorch_model.safetensors.index.json"
    if not config_path.is_file() or not index_path.is_file():
        raise FileNotFoundError(
            f"Native transformer parity requires a local sharded checkpoint with {config_path} and {index_path}."
        )

    start = time.perf_counter()
    config = TransformerConfig.from_dict(json.loads(config_path.read_text(encoding="utf-8")))
    kwargs = get_transformer_config_kwargs(config, transformer_cls)
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device(device):
            model = transformer_cls(
                **kwargs,
                prefix="parity.transformer",
            )
    finally:
        torch.set_default_dtype(previous_dtype)
    model.to(dtype=torch.bfloat16)

    weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
    shard_names = sorted(set(weight_map.values()))
    for shard_name in shard_names:
        shard_path = transformer_dir / shard_name
        with safe_open(shard_path, framework="pt", device="cpu") as shard:
            model.load_weights((name, shard.get_tensor(name)) for name in shard.keys())
    for block in model.blocks:
        if not hasattr(block.ffn, "experts"):
            continue
        routed_experts = getattr(block.ffn.experts, "routed_experts", None)
        if routed_experts is not None:
            routed_experts.quant_method.process_weights_after_loading(routed_experts)
    model.eval()
    torch.accelerator.synchronize()
    return model, time.perf_counter() - start


def _forward_transformer(
    model,
    inputs: dict[str, torch.Tensor],
    device: torch.device,
    attention_context,
) -> tuple[torch.Tensor, float, float]:
    torch.accelerator.reset_peak_memory_stats(device)
    gpu_inputs = {name: value.to(device) for name, value in inputs.items()}
    start = time.perf_counter()
    with torch.inference_mode(), attention_context:
        output = model(**gpu_inputs, return_dict=False)[0]
    torch.accelerator.synchronize()
    elapsed = time.perf_counter() - start
    peak_gib = torch.accelerator.max_memory_reserved(device) / (1024**3)
    return output.cpu(), elapsed, peak_gib


def _run_transformer_parity(
    official_module,
    native_module,
    *,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    if args.model is None:
        raise ValueError("--model is required for transformer parity.")

    from diffusers.models.attention_dispatch import attention_backend
    from torch.nn.attention import SDPBackend, sdpa_kernel

    from vllm_omni.diffusion.config import set_current_diffusion_config
    from vllm_omni.diffusion.data import AttentionConfig, AttentionSpec, OmniDiffusionConfig

    official, official_load = _load_official_transformer(
        official_module.LingBotVideoTransformer3DModel,
        args,
        device,
    )
    inputs = _make_transformer_inputs(official.config, args)
    official_output, official_forward, official_peak = _forward_transformer(
        official,
        inputs,
        device,
        attention_backend("_native_math"),
    )
    del official
    _release_cuda()

    native_config = OmniDiffusionConfig(
        diffusion_attention_config=AttentionConfig(
            default=AttentionSpec(backend="TORCH_SDPA"),
        ),
    )
    with set_current_diffusion_config(native_config):
        native, native_load = _load_native_transformer(
            native_module.LingBotVideoTransformer3DModel,
            args,
            device,
        )
    native_backend_prefs = {block.attn.attn.backend_pref for block in native.blocks}
    if native_backend_prefs != {"TORCH_SDPA"}:
        raise RuntimeError(
            f"Failed to force the native transformer to TORCH_SDPA: resolved preferences were {native_backend_prefs}."
        )
    native_output, native_forward, native_peak = _forward_transformer(
        native,
        inputs,
        device,
        sdpa_kernel(SDPBackend.MATH),
    )
    del native
    _release_cuda()

    metrics = _tensor_metrics(native_output, official_output)
    return {
        "backends": {
            "official": "diffusers:_native_math",
            "native": "TORCH_SDPA+SDPBackend.MATH",
        },
        "official": {
            "load_seconds": official_load,
            "forward_seconds": official_forward,
            "peak_reserved_gib": official_peak,
        },
        "native": {
            "load_seconds": native_load,
            "forward_seconds": native_forward,
            "peak_reserved_gib": native_peak,
        },
        "output": metrics,
        "exact": metrics["equal"],
        "strict": bool(metrics["relative_l2"] <= 5e-3 and metrics["cosine"] >= 0.99999 and metrics["max_abs"] <= 2e-2),
    }


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("LingBot MoE parity requires an available CUDA device.")

    os.environ["DIFFUSERS_ATTN_BACKEND"] = "_native_math"
    os.environ["DIFFUSION_ATTENTION_BACKEND"] = "TORCH_SDPA"
    os.environ["LINGBOT_MOE_EXPERT_BACKEND"] = "grouped_mm"
    os.environ["LINGBOT_MOE_PAD_BACKEND"] = "loop"
    os.environ["LINGBOT_MOE_REORDER_BACKEND"] = "sort"
    os.environ["LINGBOT_MOE_RESTORE_BACKEND"] = "scatter"

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
    from vllm.v1.worker.workspace import init_workspace_manager

    official_module = _load_official_module(args.official_repo)
    native_module = _load_native_module()
    vllm_config = VllmConfig(
        device_config=DeviceConfig(device=str(device)),
    )

    result: dict[str, Any] = {
        "settings": {
            "scope": args.scope,
            "seed": args.seed,
            "device": str(device),
            "official_repo": str(args.official_repo.resolve()),
            "model": args.model,
        }
    }
    with set_current_vllm_config(vllm_config):
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
        init_workspace_manager(device)
        try:
            if args.scope in {"block", "all"}:
                result["block"] = _run_block_parity(
                    official_module,
                    native_module,
                    device=device,
                    seed=args.seed,
                )
            if args.scope in {"transformer", "all"}:
                result["transformer"] = _run_transformer_parity(
                    official_module,
                    native_module,
                    args=args,
                    device=device,
                )
        finally:
            destroy_model_parallel()
            destroy_distributed_environment()

    selected = [section for name, section in result.items() if name in {"block", "transformer"}]
    result["exact"] = all(section["exact"] for section in selected)
    result["strict"] = all(section["strict"] for section in selected)
    payload = json.dumps(result, indent=2, sort_keys=True)
    print(payload)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(payload + "\n", encoding="utf-8")
    return 0 if result["strict"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
