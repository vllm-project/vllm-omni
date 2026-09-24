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
        help="Local MoE checkpoint path or a checkpoint already present in the Hugging Face cache.",
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
    denominator = actual_float.norm() * expected_float.norm()
    cosine = (
        float(torch.dot(actual_float.flatten(), expected_float.flatten()) / denominator)
        if float(denominator) > 0
        else 1.0
    )
    return {
        "shape": list(actual.shape),
        "equal": bool(torch.equal(actual, expected)),
        "max_abs": float(diff.max()) if diff.numel() else 0.0,
        "mean_abs": float(diff.mean()) if diff.numel() else 0.0,
        "rmse": float(torch.sqrt(torch.mean(diff.square()))) if diff.numel() else 0.0,
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
    if not hasattr(torch, "_grouped_mm"):
        raise RuntimeError("The block parity check requires torch._grouped_mm.")

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
    native = native_module.LingBotVideoSparseMoeBlock(**common)
    _initialize_block(official, seed)
    native.load_state_dict(official.state_dict(), strict=True)
    official = official.to(device=device, dtype=torch.bfloat16).eval()
    native = native.to(device=device, dtype=torch.bfloat16).eval()

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

    with torch.inference_mode():
        official_router = official.router(tokens)
        official_indices, official_scores = official_router[:2]
        native_indices, native_scores = native.router(tokens)

        official_scores = _apply_padding(
            official_scores,
            padding_mask,
            official.router.route_scale,
        )
        native_scores = _apply_padding(
            native_scores,
            padding_mask,
            native.router.route_scale,
        )
        official_routed = official._run_selected_experts(
            tokens,
            official_scores,
            official_indices,
        )
        native_routed = native._run_selected_experts(
            tokens,
            native_scores,
            native_indices,
        )
        official_shared = official.shared_experts(hidden_states)
        native_shared = native.shared_experts(hidden_states)
        official_output = official(hidden_states, padding_mask=padding_mask)
        native_output = native(hidden_states, padding_mask=padding_mask)

    result = {
        "router_indices_equal": bool(torch.equal(native_indices, official_indices)),
        "router_scores": _tensor_metrics(native_scores, official_scores),
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


def _load_transformer(
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

    official, official_load = _load_transformer(
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
        native, native_load = _load_transformer(
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

    official_module = _load_official_module(args.official_repo)
    native_module = _load_native_module()

    result: dict[str, Any] = {
        "settings": {
            "scope": args.scope,
            "seed": args.seed,
            "device": str(device),
            "official_repo": str(args.official_repo.resolve()),
            "model": args.model,
        }
    }
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

    result["exact"] = all(section["exact"] for name, section in result.items() if name in {"block", "transformer"})
    payload = json.dumps(result, indent=2, sort_keys=True)
    print(payload)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(payload + "\n", encoding="utf-8")
    return 0 if result["exact"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
