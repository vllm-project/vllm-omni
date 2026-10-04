# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Build a fixed-schedule MiniMax H3 AdaLN output sidecar.

Base H3 emits the SGLang-compatible v2 format. Supplying a FastH3 adapter emits
v3: the adapter's timestep and AdaLN edits are fused before projection, and the
result is bound to the exact adapter and four-step sampling contract.
"""

from __future__ import annotations

import argparse
import json
import math
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors.torch import safe_open, save_file

from vllm_omni.diffusion.models.minimax_h3.adaln_cache import (
    MiniMaxH3AdalnCacheBinding,
    fingerprint_minimax_h3_fasth3_adapter,
)
from vllm_omni.diffusion.models.minimax_h3.denoise_loop import (
    MINIMAX_H3_AUDIO_REF_COND_TIMESTEP,
    MINIMAX_H3_IMGVID_COND_TIMESTEP,
)
from vllm_omni.diffusion.models.minimax_h3.time_request import (
    minimax_h3_time_shift_sigmas,
)
from vllm_omni.diffusion.sched.sigma_schedule import DMD2SigmaSchedule

_HIDDEN_SIZE = 5376
_TIMESTEP_INPUT_DIM = 256
_NUM_BLOCKS = 50
_NUM_REFINER_BLOCKS = 2
_ATTENTION_HEAD_DIM = 128
_BLOCK_PARAM_WIDTH = 18 * _HIDDEN_SIZE
_FINAL_PARAM_WIDTH = 2 * _HIDDEN_SIZE
_CACHE_MODES = {
    "t2va": ("video", "audio"),
    "fl2va": ("video", "audio", "image"),
    "ref2va-image": ("video", "audio", "image"),
    "ref2va-audio": ("video", "audio", "audio_ref"),
    "ref2va-mixed": ("video", "audio", "image", "audio_ref"),
}
_MODE_VARIANTS = {
    "t2va": "fl2va",
    "fl2va": "fl2va",
    "ref2va-image": "ref2va",
    "ref2va-audio": "ref2va",
    "ref2va-mixed": "ref2va",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Build a fixed-schedule MiniMax H3 AdaLN cache from local native transformer weights.")
    )
    parser.add_argument("--transformer-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-variant", choices=("fl2va", "ref2va"), required=True)
    parser.add_argument("--mode", choices=tuple(_CACHE_MODES), default="t2va")
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--flow-shift", type=float, default=12.0)
    parser.add_argument("--audio-flow-shift", type=float, default=3.0)
    parser.add_argument(
        "--imgvid-cond-noise-aug",
        type=float,
        default=MINIMAX_H3_IMGVID_COND_TIMESTEP,
    )
    parser.add_argument(
        "--audio-cond-noise-aug",
        type=float,
        default=MINIMAX_H3_AUDIO_REF_COND_TIMESTEP,
    )
    parser.add_argument(
        "--timesteps",
        type=float,
        nargs="+",
        help="Override the scheduler-derived timestep plan with one exact plan.",
    )
    parser.add_argument(
        "--fasth3-adapter",
        type=Path,
        help=(
            "FastVideo FastH3 adapter file or variant directory. Its edits to "
            "the timestep and AdaLN projections are fused before building a "
            "format-v3 sidecar."
        ),
    )
    parser.add_argument(
        "--base-schedule",
        type=float,
        nargs="+",
        help=(
            "Exact pre-shift rectified-flow positions. Required with "
            "--fasth3-adapter; FastH3 Preview v1 uses "
            "0.999 0.749 0.5 0.25 0.0."
        ),
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _cache_timestep_plans(args: argparse.Namespace) -> list[torch.Tensor]:
    if args.timesteps is not None:
        return [torch.tensor(args.timesteps, dtype=torch.float32).unique(sorted=True)]

    if args.base_schedule is None:
        video_sigmas = minimax_h3_time_shift_sigmas(
            num_steps=args.num_inference_steps,
            shift_scale=args.flow_shift,
        )
        audio_sigmas = minimax_h3_time_shift_sigmas(
            num_steps=args.num_inference_steps,
            shift_scale=args.audio_flow_shift,
        )
    else:
        # An explicit distilled schedule owns its cardinality. In particular,
        # FastH3's five positions mean four transformer evaluations regardless
        # of the legacy base-H3 --num-inference-steps default (50).
        schedule = DMD2SigmaSchedule.from_positions(args.base_schedule)
        video_sigmas = schedule.shifted_sigmas(args.flow_shift)
        audio_sigmas = schedule.shifted_sigmas(args.audio_flow_shift)
    fields = _CACHE_MODES[args.mode]
    plans = []
    for video_sigma, audio_sigma in zip(video_sigmas[:-1], audio_sigmas[:-1]):
        video_timestep = 1.0 - video_sigma
        audio_timestep = 1.0 - audio_sigma
        candidates = {
            "video": video_timestep,
            "audio": audio_timestep,
            "image": max(video_timestep, args.imgvid_cond_noise_aug),
            "audio_ref": max(audio_timestep, args.audio_cond_noise_aug),
        }
        plans.append(torch.tensor([candidates[field] for field in fields], dtype=torch.float32).unique(sorted=True))

    deduplicated = []
    seen = set()
    for plan in plans:
        key = tuple(plan.tolist())
        if key not in seen:
            seen.add(key)
            deduplicated.append(plan)
    return deduplicated


def _time_embed(
    timesteps: torch.Tensor,
    *,
    proj_in_weight: torch.Tensor,
    proj_in_bias: torch.Tensor,
    proj_out_weight: torch.Tensor,
    proj_out_bias: torch.Tensor,
) -> torch.Tensor:
    half = _TIMESTEP_INPUT_DIM // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float32, device=timesteps.device) / half)
    args = timesteps[:, None] * freqs[None]
    t_freq = torch.cat((torch.cos(args), torch.sin(args)), dim=-1)
    return F.linear(
        F.silu(F.linear(t_freq, proj_in_weight, proj_in_bias)),
        proj_out_weight,
        proj_out_bias,
    )


def _load_tensor(
    name: str,
    *,
    weight_map: dict[str, str],
    files: dict[str, Any],
    device: torch.device,
) -> torch.Tensor:
    tensor_file = files[weight_map[name]]
    return tensor_file.get_tensor(name).to(device)


def _validate_build_contract(args: argparse.Namespace) -> DMD2SigmaSchedule | None:
    """Validate the mutually exclusive base-H3 and FastH3 build contracts."""
    if args.num_inference_steps < 2 and args.timesteps is None:
        raise ValueError("--num-inference-steps must be at least 2")
    mode_variant = _MODE_VARIANTS[args.mode]
    if args.model_variant != mode_variant:
        raise ValueError(f"--mode {args.mode} requires {mode_variant}")
    if not math.isfinite(args.flow_shift) or args.flow_shift <= 0:
        raise ValueError("--flow-shift must be finite and positive")
    if not math.isfinite(args.audio_flow_shift) or args.audio_flow_shift <= 0:
        raise ValueError("--audio-flow-shift must be finite and positive")

    has_adapter = args.fasth3_adapter is not None
    has_schedule = args.base_schedule is not None
    if has_adapter != has_schedule:
        raise ValueError("--fasth3-adapter and --base-schedule must be supplied together")
    if not has_adapter:
        return None
    if args.timesteps is not None:
        raise ValueError("--timesteps cannot override an adapter-bound FastH3 base schedule")
    if args.model_variant != "fl2va" or args.mode != "t2va":
        raise ValueError("FastH3 Preview v1 AdaLN sidecars require --model-variant fl2va and --mode t2va")
    assert args.base_schedule is not None
    return DMD2SigmaSchedule.from_positions(args.base_schedule)


def _load_fasth3_fusion(
    args: argparse.Namespace,
    *,
    schedule: DMD2SigmaSchedule | None,
):
    """Load and validate the canonical FastH3 fusion only when requested."""
    if args.fasth3_adapter is None:
        return None, None
    assert schedule is not None
    from vllm_omni.diffusion.models.minimax_h3.fasth3 import FastH3WeightFusion

    fusion = FastH3WeightFusion.from_path(
        args.fasth3_adapter,
        head_dim=_ATTENTION_HEAD_DIM,
        num_blocks=_NUM_BLOCKS,
        num_refiner_blocks=_NUM_REFINER_BLOCKS,
    )
    if fusion is None:
        raise ValueError(f"--fasth3-adapter is not a recognized FastH3 release: {args.fasth3_adapter}")
    if schedule.base_schedule != fusion.base_schedule:
        raise ValueError(
            "--base-schedule does not match the FastH3 release ladder: "
            f"{schedule.base_schedule!r} != {fusion.base_schedule!r}"
        )
    identity = fingerprint_minimax_h3_fasth3_adapter(fusion.source)
    return fusion, identity


def _fuse_required_tensor(
    name: str,
    tensor: torch.Tensor,
    *,
    fusion,
) -> torch.Tensor:
    """Fuse a FastH3 patch and reject a sidecar built from an unedited weight."""
    if fusion is None:
        return tensor
    fused = fusion.fuse(name, tensor)
    if fused is tensor:
        raise ValueError(f"FastH3 adapter does not patch required AdaLN input {name!r}")
    return fused


def main() -> None:
    args = _parse_args()
    schedule = _validate_build_contract(args)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("MiniMax H3 AdaLN cache must be built on CUDA")

    index_path = args.transformer_path / "model.safetensors.index.json"
    with index_path.open(encoding="utf-8") as index_file:
        weight_map = json.load(index_file)["weight_map"]

    fasth3_fusion, fasth3_identity = _load_fasth3_fusion(
        args,
        schedule=schedule,
    )

    plans = _cache_timestep_plans(args)
    if not plans or any(plan.numel() == 0 for plan in plans):
        raise ValueError("AdaLN cache must cover at least one timestep plan")
    max_plan_length = max(plan.numel() for plan in plans)
    plan_timesteps = torch.zeros((len(plans), max_plan_length), dtype=torch.float32)
    plan_lengths = torch.tensor([plan.numel() for plan in plans], dtype=torch.int64)
    block_params = torch.empty(
        (len(plans), max_plan_length, _NUM_BLOCKS, _BLOCK_PARAM_WIDTH),
        dtype=torch.bfloat16,
    )
    final_params = torch.empty(
        (len(plans), max_plan_length, _FINAL_PARAM_WIDTH),
        dtype=torch.bfloat16,
    )

    print(
        f"Building {len(plans)} AdaLN plans (width <= {max_plan_length}); "
        f"sidecar tensors use "
        f"{(block_params.numel() + final_params.numel()) * 2 / 2**30:.2f} GiB"
    )
    with ExitStack() as stack:
        files = {
            filename: stack.enter_context(
                safe_open(
                    str(args.transformer_path / filename),
                    framework="pt",
                    device="cpu",
                )
            )
            for filename in set(weight_map.values())
        }
        time_kwargs = {}
        for module, name in (
            ("proj_in", "weight"),
            ("proj_in", "bias"),
            ("proj_out", "weight"),
            ("proj_out", "bias"),
        ):
            parameter_name = f"time_embedder.{module}.{name}"
            time_kwargs[f"{module}_{name}"] = _fuse_required_tensor(
                parameter_name,
                _load_tensor(
                    parameter_name,
                    weight_map=weight_map,
                    files=files,
                    device=device,
                ),
                fusion=fasth3_fusion,
            )
        adaln_inputs = []
        for plan_index, plan in enumerate(plans):
            plan_length = plan.numel()
            plan_timesteps[plan_index, :plan_length].copy_(plan)
            adaln_inputs.append(F.silu(_time_embed(plan.to(device), **time_kwargs)).to(torch.bfloat16))

        for index in range(_NUM_BLOCKS):
            prefix = f"blocks.{index}.adaln_proj.linear"
            weight_name = f"{prefix}.weight"
            bias_name = f"{prefix}.bias"
            weight = _fuse_required_tensor(
                weight_name,
                _load_tensor(
                    weight_name,
                    weight_map=weight_map,
                    files=files,
                    device=device,
                ),
                fusion=fasth3_fusion,
            )
            bias = _fuse_required_tensor(
                bias_name,
                _load_tensor(
                    bias_name,
                    weight_map=weight_map,
                    files=files,
                    device=device,
                ),
                fusion=fasth3_fusion,
            )
            for plan_index, adaln_input in enumerate(adaln_inputs):
                plan_length = adaln_input.shape[0]
                block_params[plan_index, :plan_length, index].copy_(F.linear(adaln_input, weight, bias).cpu())
            del weight, bias
            if index == 0 or (index + 1) % 5 == 0:
                print(f"Projected AdaLN block {index + 1}/{_NUM_BLOCKS}")

        prefix = "final_layer.adaln_proj.linear"
        weight_name = f"{prefix}.weight"
        bias_name = f"{prefix}.bias"
        weight = _fuse_required_tensor(
            weight_name,
            _load_tensor(
                weight_name,
                weight_map=weight_map,
                files=files,
                device=device,
            ),
            fusion=fasth3_fusion,
        )
        bias = _fuse_required_tensor(
            bias_name,
            _load_tensor(
                bias_name,
                weight_map=weight_map,
                files=files,
                device=device,
            ),
            fusion=fasth3_fusion,
        )
        for plan_index, adaln_input in enumerate(adaln_inputs):
            plan_length = adaln_input.shape[0]
            final_params[plan_index, :plan_length].copy_(F.linear(adaln_input, weight, bias).cpu())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if fasth3_identity is None:
        binding = MiniMaxH3AdalnCacheBinding(
            format_version="2",
            model_variant=args.model_variant,
        )
    else:
        assert schedule is not None
        binding = MiniMaxH3AdalnCacheBinding.for_fasth3(
            adapter=fasth3_identity,
            model_variant=args.model_variant,
            mode=args.mode,
            base_schedule=schedule.base_schedule,
            flow_shift=args.flow_shift,
            audio_flow_shift=args.audio_flow_shift,
        )
    save_file(
        {
            "plan_timesteps": plan_timesteps,
            "plan_lengths": plan_lengths,
            "block_params": block_params,
            "final_params": final_params,
        },
        str(args.output),
        metadata=binding.to_metadata(),
    )
    print(f"Wrote MiniMax H3 AdaLN cache v{binding.format_version}: {args.output}")


if __name__ == "__main__":
    main()
