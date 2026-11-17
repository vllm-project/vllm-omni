# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Loader for the native-layout MiniMax-H3 distilled LoRA (FlashGen v1.0).

This contract is keyed by ``key_format=minimax-h3-native`` and is distinct from
the LightX2V Turbo contract in the sibling ``..lora`` module: it keeps the fused
``qkv_proj``, adds ``adaln_proj.linear``, uses rank 64, and declares its own
rectified-flow schedule in safetensors metadata.

The artifact is trained on Ascend NPU, but nothing below is NPU-specific. Only
``torch`` and ``safetensors`` are used, tensors are read on CPU, and the compute
device is chosen later by the pipeline, so the same file serves on CUDA too.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import get_args

import torch
from safetensors import safe_open
from vllm.config.lora import MaxLoRARanks
from vllm.lora.lora_model import LoRAModel
from vllm.lora.lora_weights import LoRALayerWeights, PackedLoRALayerWeights
from vllm.lora.peft_helper import PEFTHelper
from vllm.model_executor.models.utils import WeightsMapper

from vllm_omni.diffusion.sched.sigma_schedule import BASE_SCHEDULE_KEY, DMD2SigmaSchedule
from vllm_omni.lora.request import LoRARequest

from ..minimax_h3_transformer import _reorder_grouped_qkv_to_qkv

_LORA_A_SUFFIX = ".lora_A.default.weight"
_LORA_B_SUFFIX = ".lora_B.default.weight"

_NATIVE_RANK = 64
_NATIVE_ALPHA = 64
_NATIVE_HIDDEN_SIZE = 5376
_NATIVE_ATTENTION_INNER_SIZE = 7168
_NATIVE_FFN_HIDDEN_SIZE = 14336
_NATIVE_TIME_EMBED_DIM = 2688
_NATIVE_BLOCK_ADALN_OUT = 96768
_NATIVE_FINAL_ADALN_OUT = 10752
_NATIVE_NUM_QUERY_GROUPS = 56
_NATIVE_HEADS_PER_GROUP = 1
_NATIVE_HEAD_DIM = 128
_NATIVE_QKV_SLICE = _NATIVE_ATTENTION_INNER_SIZE
_NATIVE_KEY_FORMAT = "minimax-h3-native"
_NATIVE_QKV_LAYOUT = "grouped"
# Public because request validation in the pipeline speaks the same contract.
MINIMAX_H3_NATIVE_INFERENCE_STEPS = 4
_NATIVE_FILENAME = "minimax_h3_t2va_flashgen_4step_v1.0_768p_bf16.safetensors"
_NATIVE_TARGET_SUFFIXES = (
    "attn.qkv_proj",
    "attn.out_proj",
    "mlp.fc1",
    "mlp.fc2",
    "adaln_proj.linear",
)
_NATIVE_TOKEN_REFINER_SUFFIXES = (
    "attn.qkv_proj",
    "attn.out_proj",
    "mlp.fc1",
    "mlp.fc2",
)
_NATIVE_TARGET_DIMS = {
    "attn.qkv_proj": (_NATIVE_HIDDEN_SIZE, 3 * _NATIVE_ATTENTION_INNER_SIZE),
    "attn.out_proj": (_NATIVE_ATTENTION_INNER_SIZE, _NATIVE_HIDDEN_SIZE),
    "mlp.fc1": (_NATIVE_HIDDEN_SIZE, 2 * _NATIVE_FFN_HIDDEN_SIZE),
    "mlp.fc2": (_NATIVE_FFN_HIDDEN_SIZE, _NATIVE_HIDDEN_SIZE),
    "adaln_proj.linear": (_NATIVE_TIME_EMBED_DIM, _NATIVE_BLOCK_ADALN_OUT),
}
_NATIVE_FINAL_ADALN_DIMS = (_NATIVE_TIME_EMBED_DIM, _NATIVE_FINAL_ADALN_OUT)
_NATIVE_EXPECTED_TARGETS = frozenset(
    [*(f"blocks.{block_index}.{suffix}" for block_index in range(50) for suffix in _NATIVE_TARGET_SUFFIXES)]
    + [
        *(
            f"token_refiner.blocks.{block_index}.{suffix}"
            for block_index in range(2)
            for suffix in _NATIVE_TOKEN_REFINER_SUFFIXES
        )
    ]
    + ["final_layer.adaln_proj.linear"]
)
_NATIVE_TARGET_PATTERN = (
    r"^(?:transformer\.blocks\.\d+\.(?:attn\.(?:qkv_proj|out_proj)|mlp\.(?:fc1|fc2)|adaln_proj\.linear)"
    r"|transformer\.token_refiner\.blocks\.\d+\.(?:attn\.(?:qkv_proj|out_proj)|mlp\.(?:fc1|fc2))"
    r"|transformer\.final_layer\.adaln_proj\.linear)$"
)
_NATIVE_WEIGHTS_MAPPER = WeightsMapper(
    orig_to_new_substr={
        "transformer.": "",
        ".lora_A.default.": ".lora_A.",
        ".lora_B.default.": ".lora_B.",
    }
)


@dataclass(frozen=True)
class NativeDistilledLoRASpec:
    schedule: DMD2SigmaSchedule
    task: str = "t2va"
    rank: int = _NATIVE_RANK
    qkv_rank: int = _NATIVE_RANK

    def rank_for(self, name: str) -> int:
        return self.qkv_rank if name.endswith(".attn.qkv_proj") else self.rank

    @property
    def targets(self) -> frozenset[str]:
        if self.task == "ref2va":
            return frozenset(name for name in _NATIVE_EXPECTED_TARGETS if not name.endswith("adaln_proj.linear"))
        return _NATIVE_EXPECTED_TARGETS


def _select_native_file(artifact_path: str | Path) -> Path | None:
    path = Path(artifact_path)
    if path.is_file():
        return path if path.suffix == ".safetensors" else None
    if not path.is_dir():
        return None

    candidates = []
    for candidate in sorted(path.glob("*.safetensors")):
        if not candidate.is_file():
            continue
        if candidate.name == _NATIVE_FILENAME:
            candidates.append(candidate)
            continue
        with safe_open(candidate, framework="pt", device="cpu") as checkpoint:
            metadata = checkpoint.metadata() or {}
            if metadata.get("key_format") == _NATIVE_KEY_FORMAT:
                candidates.append(candidate)
    if len(candidates) > 1:
        raise ValueError(f"Multiple MiniMax-H3 native LoRA files in {path}; pass a specific file")
    return candidates[0] if candidates else None


def _native_target_dims(target: str) -> tuple[int, int]:
    if target == "final_layer.adaln_proj.linear":
        return _NATIVE_FINAL_ADALN_DIMS
    if target.endswith("adaln_proj.linear"):
        return _NATIVE_TARGET_DIMS["adaln_proj.linear"]
    for suffix in ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2"):
        if target.endswith(suffix):
            return _NATIVE_TARGET_DIMS[suffix]
    raise ValueError(f"Unsupported MiniMax-H3 native target: {target!r}")


def _parse_native_base_schedule(metadata: Mapping[str, str]) -> DMD2SigmaSchedule | None:
    """Read the adapter-declared schedule from safetensors string metadata.

    safetensors metadata is string-only, so the native artifact ships the sigma
    positions comma separated. Absent means the adapter declares no schedule.
    """
    raw = metadata.get(BASE_SCHEDULE_KEY)
    if raw is None:
        return None
    positions = [part.strip() for part in str(raw).split(",") if part.strip()]
    if not positions:
        raise ValueError(f"MiniMax-H3 native LoRA {BASE_SCHEDULE_KEY} must not be empty")
    try:
        return DMD2SigmaSchedule.from_positions([float(position) for position in positions])
    except ValueError as exc:
        raise ValueError(f"MiniMax-H3 native LoRA {BASE_SCHEDULE_KEY} is malformed: {exc}") from exc


def _validate_native_metadata(metadata: Mapping[str, str]) -> DMD2SigmaSchedule:
    if metadata.get("key_format") != _NATIVE_KEY_FORMAT:
        raise ValueError(f"MiniMax-H3 native LoRA requires safetensors metadata key_format={_NATIVE_KEY_FORMAT!r}")
    if metadata.get("qkv_layout") != _NATIVE_QKV_LAYOUT:
        raise ValueError(f"MiniMax-H3 native LoRA requires safetensors metadata qkv_layout={_NATIVE_QKV_LAYOUT!r}")
    try:
        rank = int(metadata.get("lora_rank", ""))
        alpha = float(metadata.get("lora_alpha", ""))
    except (TypeError, ValueError) as exc:
        raise ValueError("MiniMax-H3 native LoRA metadata lora_rank/lora_alpha must be numeric") from exc
    if rank != _NATIVE_RANK:
        raise ValueError(f"MiniMax-H3 native LoRA requires lora_rank={_NATIVE_RANK}, got {rank!r}")
    if alpha != _NATIVE_ALPHA:
        raise ValueError(f"MiniMax-H3 native LoRA requires lora_alpha={_NATIVE_ALPHA}, got {alpha!r}")
    tasks = {part.strip().lower() for part in str(metadata.get("tasks", "")).split(",") if part.strip()}
    if tasks != {"t2va"}:
        raise ValueError("MiniMax-H3 native LoRA v1.0 supports tasks=t2va only")
    schedule = _parse_native_base_schedule(metadata)
    if schedule is None:
        raise ValueError(f"MiniMax-H3 native LoRA requires safetensors metadata {BASE_SCHEDULE_KEY}")
    # Request validation speaks in interval counts, so a mislabeled schedule
    # would otherwise silently change the step count instead of failing here.
    if schedule.num_inference_steps != MINIMAX_H3_NATIVE_INFERENCE_STEPS:
        raise ValueError(
            f"MiniMax-H3 native LoRA v1.0 requires a {MINIMAX_H3_NATIVE_INFERENCE_STEPS}-interval base_schedule "
            f"({MINIMAX_H3_NATIVE_INFERENCE_STEPS + 1} sigma positions), got {schedule.num_inference_steps}"
        )
    return schedule


def _parse_native_v2(metadata: Mapping[str, str], partition: str) -> NativeDistilledLoRASpec:
    task = metadata.get("partition")
    if task not in ("fl2va", "ref2va"):
        raise ValueError("MiniMax-H3 native LoRA v2 requires partition=fl2va or ref2va")
    if partition != task and not (partition == "combined" and task == "fl2va"):
        raise ValueError(f"MiniMax-H3 native LoRA for {task} requires a matching base; service partition={partition}")
    ref2va = task == "ref2va"
    rank = 128 if ref2va else _NATIVE_RANK
    qkv_rank = 384 if ref2va else rank
    expected = {
        "tasks": task,
        "target_profile": "h3_ref2va_208_v1" if ref2va else "h3_i2va_259_v1",
        "qkv_layout": "grouped",
        "fc1_layout": "gate_up",
        "lora_rank": str(rank),
        "scaling_mode": "unit_delta",
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"MiniMax-H3 native LoRA v2 requires {key}={value!r}")
    if metadata.get("sampler", "euler") != "euler":
        raise ValueError("MiniMax-H3 native LoRA v2 requires an Euler-compatible export")
    try:
        overrides = json.loads(metadata.get("rank_overrides", ""))
        scale, video_shift, audio_shift = (
            float(metadata.get(key, "")) for key in ("delta_scale", "video_shift", "audio_shift")
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid native LoRA v2 rank_overrides, delta_scale or flow shifts") from exc
    expected_overrides = {"attn.qkv_proj": 384} if ref2va else {}
    if (
        not isinstance(overrides, dict)
        or overrides != expected_overrides
        or any(type(value) is not int for value in overrides.values())
    ):
        raise ValueError(f"MiniMax-H3 native LoRA v2 requires rank_overrides={expected_overrides}")
    if (scale, video_shift, audio_shift) != (1.0, 12.0, 3.0):
        raise ValueError("MiniMax-H3 native LoRA v2 requires delta_scale=1, video_shift=12, audio_shift=3")
    schedule = _parse_native_base_schedule(metadata)
    if schedule is None or schedule.base_schedule != (1.0, 0.7, 0.4, 0.15, 0.0):
        raise ValueError("MiniMax-H3 native LoRA v2 requires base_schedule=1,0.7,0.4,0.15,0")
    if qkv_rank > max(get_args(MaxLoRARanks)):
        raise ValueError(f"Installed vLLM cannot allocate MiniMax-H3 native LoRA rank {qkv_rank}")
    return NativeDistilledLoRASpec(schedule, task=task, rank=rank, qkv_rank=qkv_rank)


def _validate_native_tensors(checkpoint, spec: NativeDistilledLoRASpec | None = None) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    pairs: dict[str, set[str]] = {}
    raw_targets: set[str] = set()
    for name in checkpoint.keys():
        if name.endswith(_LORA_A_SUFFIX):
            raw_target = name[: -len(_LORA_A_SUFFIX)]
            side = "a"
        elif name.endswith(_LORA_B_SUFFIX):
            raw_target = name[: -len(_LORA_B_SUFFIX)]
            side = "b"
        else:
            raise ValueError(f"Unconsumed MiniMax-H3 native tensor: {name!r}")
        if not raw_target.startswith("transformer."):
            raise ValueError(f"MiniMax-H3 native LoRA tensors must use transformer.* keys, got {name!r}")
        mapped_target = raw_target.removeprefix("transformer.")
        raw_targets.add(mapped_target)

        target_sides = pairs.setdefault(mapped_target, set())
        if side in target_sides:
            raise ValueError(f"Duplicate MiniMax-H3 native tensor for {mapped_target}.{side}")
        target_sides.add(side)

        tensor = checkpoint.get_tensor(name)
        if tensor.ndim != 2:
            raise ValueError(f"MiniMax-H3 native LoRA tensors must be matrices, got {name}={tuple(tensor.shape)}")
        input_dim, output_dim = _native_target_dims(mapped_target)
        rank = _NATIVE_RANK if spec is None else spec.rank_for(mapped_target)
        expected_shape = (rank, input_dim) if side == "a" else (output_dim, rank)
        if spec is not None and tensor.dtype != torch.bfloat16:
            raise ValueError(f"MiniMax-H3 native LoRA v2 tensor must be BF16: {name}")
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(
                f"MiniMax-H3 native tensor has invalid global shape: {name}={tuple(tensor.shape)}, "
                f"expected={expected_shape}"
            )
        # The LoRA manager re-parses raw tensor names without the weights mapper, so
        # the PEFT adapter-name segment must be dropped here: `parse_fine_tuned_lora_name`
        # rejects `...lora_A.<adapter>.weight` keys.
        tensors[f"{raw_target}.lora_{'A' if side == 'a' else 'B'}.weight"] = tensor

    incomplete = sorted(target for target, sides in pairs.items() if sides != {"a", "b"})
    if incomplete:
        raise ValueError(f"Incomplete MiniMax-H3 native LoRA pairs: {incomplete}")
    targets = _NATIVE_EXPECTED_TARGETS if spec is None else spec.targets
    missing = sorted(targets - raw_targets)
    unexpected = sorted(raw_targets - targets)
    if missing or unexpected:
        raise ValueError(
            "MiniMax-H3 native target set does not match the supported artifact: "
            f"missing={len(missing)} {missing[:5]}, unexpected={len(unexpected)} {unexpected[:5]}"
        )
    return tensors


def _pack_h3_native_fc1(lora_model: LoRAModel) -> None:
    for module_name, weights in tuple(lora_model.loras.items()):
        if not module_name.endswith(".mlp.fc1"):
            continue
        gate_b, up_b = weights.lora_b.chunk(2, dim=0)
        lora_model.loras[module_name] = PackedLoRALayerWeights(
            module_name=module_name,
            rank=weights.rank,
            lora_alphas=[weights.lora_alpha, weights.lora_alpha],
            lora_a=[weights.lora_a, weights.lora_a],
            lora_b=[gate_b.contiguous(), up_b.contiguous()],
            scaling=[weights.scaling, weights.scaling],
        )


def _pack_h3_native_qkv(lora_model: LoRAModel) -> None:
    for module_name, weights in tuple(lora_model.loras.items()):
        if not module_name.endswith(".attn.qkv_proj"):
            continue
        reordered_b = _reorder_grouped_qkv_to_qkv(
            weights.lora_b,
            num_query_groups=_NATIVE_NUM_QUERY_GROUPS,
            heads_per_group=_NATIVE_HEADS_PER_GROUP,
            head_dim=_NATIVE_HEAD_DIM,
        )
        q_b, k_b, v_b = torch.split(
            reordered_b,
            [_NATIVE_QKV_SLICE, _NATIVE_QKV_SLICE, _NATIVE_QKV_SLICE],
            dim=0,
        )
        lora_model.loras[module_name] = PackedLoRALayerWeights(
            module_name=module_name,
            rank=weights.rank,
            lora_alphas=[weights.lora_alpha, weights.lora_alpha, weights.lora_alpha],
            lora_a=[weights.lora_a, weights.lora_a, weights.lora_a],
            lora_b=[q_b.contiguous(), k_b.contiguous(), v_b.contiguous()],
            scaling=[weights.scaling, weights.scaling, weights.scaling],
        )


def load_minimax_h3_native_distilled_lora(
    *,
    partition: str,
    lora_request: LoRARequest,
    lora_path: str | Path,
    dtype: torch.dtype,
    unsupported_offload_mode: str | None = None,
) -> tuple[LoRAModel, PEFTHelper, NativeDistilledLoRASpec] | None:
    """Load a native-layout MiniMax-H3 distilled LoRA through the legacy manager."""

    lora_file = _select_native_file(lora_path)
    if lora_file is None:
        return None
    with safe_open(lora_file, framework="pt", device="cpu") as checkpoint:
        metadata = checkpoint.metadata() or {}
        if metadata.get("key_format") != _NATIVE_KEY_FORMAT:
            if lora_file.name == _NATIVE_FILENAME:
                raise ValueError(
                    f"MiniMax-H3 native LoRA requires safetensors metadata key_format={_NATIVE_KEY_FORMAT!r}"
                )
            return None
        version = metadata.get("native_format_version")
        if version is None:
            if lora_file.name != _NATIVE_FILENAME:
                raise ValueError(f"MiniMax-H3 native LoRA supports only {_NATIVE_FILENAME!r}, got {lora_file.name!r}")
            sigma_schedule = _validate_native_metadata(metadata)
            if partition == "ref2va":
                raise ValueError("MiniMax-H3 legacy T2VA native LoRA cannot be loaded into the ref2va partition")
            spec = NativeDistilledLoRASpec(sigma_schedule, rank=_NATIVE_RANK, qkv_rank=_NATIVE_RANK)
        elif version == "2":
            spec = _parse_native_v2(metadata, partition)
        else:
            raise ValueError(f"Unsupported MiniMax-H3 native LoRA format version: {version!r}")
        if unsupported_offload_mode is not None:
            raise ValueError(f"MiniMax-H3 native dynamic LoRA does not support {unsupported_offload_mode}")
        tensors = _validate_native_tensors(checkpoint, spec if version == "2" else None)

    peft_helper = PEFTHelper.from_dict(
        {
            "r": spec.qkv_rank,
            "lora_alpha": _NATIVE_ALPHA if version is None else spec.qkv_rank,
            "target_modules": (
                _NATIVE_TARGET_PATTERN
                if spec.task != "ref2va"
                else [f"transformer.{target}" for target in sorted(spec.targets)]
            ),
        }
    )
    if version is None:
        lora_model = LoRAModel.from_lora_tensors(
            lora_model_id=lora_request.lora_int_id,
            tensors=tensors,
            peft_helper=peft_helper,
            device="cpu",
            dtype=dtype,
            weights_mapper=_NATIVE_WEIGHTS_MAPPER,
        )
    else:
        # The exported A already includes its effective scale. Keep each
        # projection's rank while advertising the largest allocation need.
        loras = {}
        for target in sorted(spec.targets):
            rank = spec.rank_for(target)
            loras[target] = LoRALayerWeights(
                module_name=target,
                rank=rank,
                lora_alpha=rank,
                lora_a=tensors[f"transformer.{target}.lora_A.weight"].to(dtype=dtype),
                lora_b=tensors[f"transformer.{target}.lora_B.weight"].to(dtype=dtype),
                scaling=1.0,
            )
        lora_model = LoRAModel(lora_model_id=lora_request.lora_int_id, rank=spec.qkv_rank, loras=loras)
    _pack_h3_native_qkv(lora_model)
    _pack_h3_native_fc1(lora_model)
    return lora_model, peft_helper, spec
