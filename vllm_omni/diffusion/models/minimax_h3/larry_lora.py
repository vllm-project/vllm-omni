# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Load the unmodified Larry v4-600 EMA single-file adapter at request time."""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from safetensors import safe_open
from vllm.logger import init_logger
from vllm.lora.lora_model import LoRAModel
from vllm.lora.lora_weights import LoRALayerWeights, PackedLoRALayerWeights
from vllm.lora.peft_helper import PEFTHelper

from vllm_omni.errors import OmniClientError
from vllm_omni.lora.request import LoRARequest

if TYPE_CHECKING:
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    from .pipeline_minimax_h3 import MiniMaxH3Pipeline

logger = init_logger(__name__)
LARRY_FILENAME = "minimax_h3_turbo_v4_step600_ema.safetensors"
LARRY_TARGET_PATTERN = (
    r"^transformer\.(?:blocks\.\d+\.(?:attn\.(?:qkv_proj|out_proj)|"
    r"mlp\.(?:fc1|fc2)|adaln_proj\.linear)|token_refiner\.blocks\.\d+\."
    r"(?:attn\.(?:qkv_proj|out_proj)|mlp\.(?:fc1|fc2))|final_layer\.adaln_proj\.linear)$"
)


def larry_targets() -> dict[str, tuple[int, int]]:
    dims = {
        "attn.qkv_proj": (5376, 21504),
        "attn.out_proj": (7168, 5376),
        "mlp.fc1": (5376, 28672),
        "mlp.fc2": (14336, 5376),
    }
    targets = {
        f"{prefix}.{index}.{suffix}": shape
        for prefix, count in (("blocks", 50), ("token_refiner.blocks", 2))
        for index in range(count)
        for suffix, shape in dims.items()
    }
    targets.update({f"blocks.{i}.adaln_proj.linear": (2688, 96768) for i in range(50)})
    targets["final_layer.adaln_proj.linear"] = (2688, 10752)
    return targets


def select_larry_file(lora_path: str | Path) -> Path | None:
    path = Path(lora_path)
    if path.is_dir():
        path = path / LARRY_FILENAME
    return path if path.name == LARRY_FILENAME and path.is_file() else None


def pack_larry_layer(name: str, a: torch.Tensor, b: torch.Tensor) -> LoRALayerWeights:
    """Larry already uses [all Q, all K, all V] and [gate, up] output rows.

    In particular, DO NOT apply FlashGen/native-checkpoint grouped-QKV reorder.
    Packed global B slices are sharded by the existing LoRA layer wrappers.
    """
    rank = a.shape[0]
    slices = 3 if name.endswith(".attn.qkv_proj") else 2 if name.endswith(".mlp.fc1") else 1
    if slices == 1:
        return LoRALayerWeights(module_name=name, rank=rank, lora_alpha=rank, lora_a=a, lora_b=b, scaling=1.0)
    parts = [part.contiguous() for part in b.chunk(slices, dim=0)]
    return PackedLoRALayerWeights(
        module_name=name,
        rank=rank,
        lora_alphas=[rank] * slices,
        lora_a=[a] * slices,
        lora_b=parts,
        scaling=[1.0] * slices,
    )


def load_minimax_h3_larry_lora(
    *,
    partition: str,
    lora_request: LoRARequest,
    lora_path: str | Path,
    dtype: torch.dtype,
    unsupported_offload_mode: str | None = None,
) -> tuple[LoRAModel, PEFTHelper] | None:
    path = select_larry_file(lora_path)
    if path is None:
        return None  # leave LightX2V, FlashGen and generic PEFT to their loaders
    if partition not in {"fl2va", "ref2va", "combined"}:
        raise ValueError(f"Unknown H3 partition for Larry: {partition!r}")
    if unsupported_offload_mode:
        raise ValueError(f"Larry dynamic LoRA does not support {unsupported_offload_mode}")

    targets = larry_targets()
    expected = {f"{name}.lora_{side}.weight" for name in targets for side in ("A", "B")}
    layers = {}
    ranks = set()
    with safe_open(path, framework="pt", device="cpu") as checkpoint:
        actual = set(checkpoint.keys())
        if actual != expected:
            raise ValueError(
                "Larry v4 target set mismatch; "
                f"missing={sorted(expected - actual)[:8]}, "
                f"unexpected={sorted(actual - expected)[:8]}. No tensors are ignored."
            )
        for raw_name, (input_dim, output_dim) in targets.items():
            a = checkpoint.get_tensor(f"{raw_name}.lora_A.weight")
            b = checkpoint.get_tensor(f"{raw_name}.lora_B.weight")
            if a.ndim != 2 or b.ndim != 2:
                raise ValueError(f"Larry A/B must be matrices: {raw_name}")
            rank = a.shape[0]
            if not 1 <= rank <= 128 or tuple(a.shape) != (rank, input_dim) or tuple(b.shape) != (output_dim, rank):
                raise ValueError(
                    f"Larry tensor dimensions mismatch for {raw_name}: A={tuple(a.shape)}, B={tuple(b.shape)}"
                )
            if a.dtype != torch.bfloat16 or b.dtype != torch.bfloat16:
                raise ValueError(f"Expected original BF16 Larry tensors: {raw_name}")
            name = "transformer." + raw_name
            layers[name] = pack_larry_layer(name, a.to(dtype=dtype), b.to(dtype=dtype))
            ranks.add(rank)
    # Allocate buffers for the largest rank, keeping each layer's real rank.
    # Direct LoRALayerWeights construction avoids global-alpha/mixed-rank loss.
    rank = max(ranks)
    peft_helper = PEFTHelper.from_dict({"r": rank, "lora_alpha": rank, "target_modules": LARRY_TARGET_PATTERN})
    model = LoRAModel(lora_model_id=lora_request.lora_int_id, rank=rank, loras=layers)
    logger.info(
        "Loaded Larry v4-600 EMA: %s, %d targets, ranks=%s, alpha/rank=1, partition=%s",
        path,
        len(layers),
        sorted(ranks),
        partition,
    )
    return model, peft_helper


def validate_larry_request(pipeline: MiniMaxH3Pipeline, sampling: OmniDiffusionSamplingParams, task: str) -> None:
    request = sampling.lora_request
    if (
        request is None
        or math.isclose(0.0, float(sampling.lora_scale))
        or request.lora_int_id not in getattr(pipeline, "_larry_lora_ids", ())
    ):
        return
    if task == "ref2va" and pipeline.partition != "ref2va":
        raise OmniClientError("Larry Ref2VA requires --task-type ref2va so it binds the active transformer")
    if pipeline._base_schedule_for_task(task) is not None:
        raise OmniClientError("Larry requires the original H3 checkpoint without a pinned distilled base_schedule")
    steps = sampling.num_inference_steps
    if isinstance(steps, bool) or not isinstance(steps, int) or not 4 <= steps <= 8:
        raise OmniClientError("Larry v4 requires explicit num_inference_steps=4..8; start with 6 or 8")
    extra = sampling.extra_args or {}
    for name, default, expected in (
        ("flow_shift", pipeline.default_video_shift, 12.0),
        ("audio_flow_shift", pipeline.default_audio_shift, 3.0),
    ):
        try:
            value = float(extra.get(name, default))
        except (TypeError, ValueError) as exc:
            raise OmniClientError(f"Larry requires {name}={expected:g}") from exc
        if not math.isclose(value, expected):
            raise OmniClientError(f"Larry requires {name}={expected:g}, got {value:g}")
    # H3 normalizes an omitted or null sampler to Euler.
    if extra.get("sampler") not in (None, "euler"):
        raise OmniClientError("Larry integration currently validates Euler only; set sampler=euler")
