# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path

import torch
from safetensors import safe_open
from vllm.lora.lora_model import LoRAModel
from vllm.lora.lora_weights import PackedLoRALayerWeights
from vllm.lora.peft_helper import PEFTHelper
from vllm.model_executor.models.utils import WeightsMapper

from vllm_omni.diffusion.sched.sigma_schedule import DMD2SigmaSchedule
from vllm_omni.lora.request import LoRARequest

from .minimax_h3_transformer import _reorder_grouped_qkv_to_qkv

_TURBO_RANK = 128
_TURBO_ALPHA = 128
_TURBO_HIDDEN_SIZE = 5376
_TURBO_ATTENTION_INNER_SIZE = 7168
_TURBO_FFN_HIDDEN_SIZE = 14336
_TURBO_FILENAME = "minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors"
_LORA_A_SUFFIX = ".lora_A.default.weight"
_LORA_B_SUFFIX = ".lora_B.default.weight"
_TURBO_TARGETS = frozenset({"to_q", "to_k", "to_v", "out_proj", "fc1", "fc2"})
_TURBO_RAW_TARGET_SUFFIXES = (
    "attn.to_q",
    "attn.to_k",
    "attn.to_v",
    "attn.to_out.0",
    "ff.net.0.proj",
    "ff.net.2",
)
_TURBO_TARGET_DIMS = {
    "attn.to_q": (_TURBO_HIDDEN_SIZE, _TURBO_ATTENTION_INNER_SIZE),
    "attn.to_k": (_TURBO_HIDDEN_SIZE, _TURBO_ATTENTION_INNER_SIZE),
    "attn.to_v": (_TURBO_HIDDEN_SIZE, _TURBO_ATTENTION_INNER_SIZE),
    "attn.to_out.0": (_TURBO_ATTENTION_INNER_SIZE, _TURBO_HIDDEN_SIZE),
    "ff.net.0.proj": (_TURBO_HIDDEN_SIZE, 2 * _TURBO_FFN_HIDDEN_SIZE),
    "ff.net.2": (_TURBO_FFN_HIDDEN_SIZE, _TURBO_HIDDEN_SIZE),
}
_TURBO_EXPECTED_RAW_TARGETS = frozenset(
    f"{prefix}.{block_index}.{suffix}"
    for prefix, block_count in (
        ("transformer_blocks", 50),
        ("token_refiner.refiner_blocks", 2),
    )
    for block_index in range(block_count)
    for suffix in _TURBO_RAW_TARGET_SUFFIXES
)
_TURBO_TARGET_PATTERN = (
    r"^transformer\.(?:token_refiner\.blocks|blocks)\.\d+\."
    r"(?:attn\.(?:to_q|to_k|to_v|out_proj)|mlp\.(?:fc1|fc2))$"
)

_TURBO_WEIGHTS_MAPPER = WeightsMapper(
    orig_to_new_substr={
        "token_refiner.refiner_blocks.": "token_refiner.blocks.",
        "transformer_blocks.": "blocks.",
        ".attn.to_out.0.": ".attn.out_proj.",
        ".ff.net.0.proj.": ".mlp.fc1.",
        ".ff.net.2.": ".mlp.fc2.",
        ".lora_A.default.": ".lora_A.",
        ".lora_B.default.": ".lora_B.",
    }
)


def _select_turbo_file(artifact_path: str | Path) -> Path | None:
    path = Path(artifact_path)
    if path.is_file():
        return path if path.suffix == ".safetensors" else None
    if not path.is_dir():
        return None

    candidate = path / _TURBO_FILENAME
    return candidate if candidate.is_file() else None


def _validate_and_convert_tensors(checkpoint) -> dict[str, torch.Tensor]:
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
            raise ValueError(f"Unconsumed MiniMax-H3 Turbo tensor: {name!r}")
        raw_targets.add(raw_target)

        mapped_name = _TURBO_WEIGHTS_MAPPER.apply_list([name])[0]
        mapped_target = mapped_name.rsplit(".lora_", 1)[0]
        if mapped_target.rsplit(".", 1)[-1] not in _TURBO_TARGETS:
            raise ValueError(f"Unsupported MiniMax-H3 Turbo target: {raw_target!r}")
        target_sides = pairs.setdefault(mapped_target, set())
        if side in target_sides:
            raise ValueError(f"Duplicate MiniMax-H3 Turbo tensor for {mapped_target}.{side}")
        target_sides.add(side)

        tensor = checkpoint.get_tensor(name)
        if tensor.ndim != 2:
            raise ValueError(f"MiniMax-H3 Turbo LoRA tensors must be matrices, got {name}={tuple(tensor.shape)}")
        suffix = next((suffix for suffix in _TURBO_RAW_TARGET_SUFFIXES if raw_target.endswith(suffix)), None)
        if suffix is None:
            raise ValueError(f"MiniMax-H3 Turbo LoRA contains unsupported target: {raw_target}")
        input_dim, output_dim = _TURBO_TARGET_DIMS[suffix]
        expected_shape = (_TURBO_RANK, input_dim) if side == "a" else (output_dim, _TURBO_RANK)
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(
                f"MiniMax-H3 Turbo tensor has invalid global shape: {name}={tuple(tensor.shape)}, "
                f"expected={expected_shape}"
            )
        if side == "b" and ".ff.net.0.proj." in name:
            value, gate = tensor.chunk(2, dim=0)
            tensor = torch.cat((gate, value), dim=0).contiguous()
        tensors[name] = tensor

    incomplete = sorted(target for target, sides in pairs.items() if sides != {"a", "b"})
    if incomplete:
        raise ValueError(f"Incomplete MiniMax-H3 Turbo LoRA pairs: {incomplete}")
    missing = sorted(_TURBO_EXPECTED_RAW_TARGETS - raw_targets)
    unexpected = sorted(raw_targets - _TURBO_EXPECTED_RAW_TARGETS)
    if missing or unexpected:
        raise ValueError(
            "MiniMax-H3 Turbo target set does not match the supported v1.0 artifact: "
            f"missing={len(missing)} {missing[:5]}, unexpected={len(unexpected)} {unexpected[:5]}"
        )
    return tensors


def _pack_h3_turbo_fc1(lora_model: LoRAModel) -> None:
    """Represent H3's fused gate/up projection without generic layout guesses."""

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


def load_minimax_h3_turbo_lora(
    *,
    partition: str,
    lora_request: LoRARequest,
    lora_path: str | Path,
    dtype: torch.dtype,
    unsupported_offload_mode: str | None = None,
) -> tuple[LoRAModel, PEFTHelper] | None:
    """Load the published LightX2V Turbo v1.0 through the legacy manager."""

    lora_file = _select_turbo_file(lora_path)
    if lora_file is None:
        return None
    with safe_open(lora_file, framework="pt", device="cpu") as checkpoint:
        metadata = checkpoint.metadata() or {}
        if metadata.get("key_format") != "minimax-h3-diffusers":
            if lora_file.name == _TURBO_FILENAME:
                raise ValueError(
                    "MiniMax-H3 Turbo v1.0 requires safetensors metadata key_format='minimax-h3-diffusers'"
                )
            return None
        if lora_file.name != _TURBO_FILENAME:
            raise ValueError(f"MiniMax-H3 Turbo supports only {_TURBO_FILENAME!r}, got {lora_file.name!r}")
        raw_alpha = metadata.get("alpha")
        try:
            alpha = float(raw_alpha) if raw_alpha is not None else math.nan
        except ValueError as exc:
            raise ValueError(f"MiniMax-H3 Turbo alpha must be numeric, got {raw_alpha!r}") from exc
        if alpha != _TURBO_ALPHA:
            raise ValueError(f"MiniMax-H3 Turbo v1.0 requires alpha={_TURBO_ALPHA}, got {raw_alpha!r}")
        if partition == "ref2va":
            raise ValueError("MiniMax-H3 Turbo LoRA supports FL2VA/T2VA only")
        if unsupported_offload_mode is not None:
            raise ValueError(f"MiniMax-H3 Turbo dynamic LoRA does not support {unsupported_offload_mode}")
        tensors = _validate_and_convert_tensors(checkpoint)

    peft_helper = PEFTHelper.from_dict(
        {
            "r": _TURBO_RANK,
            "lora_alpha": _TURBO_ALPHA,
            "target_modules": _TURBO_TARGET_PATTERN,
        }
    )
    lora_model = LoRAModel.from_lora_tensors(
        lora_model_id=lora_request.lora_int_id,
        tensors=tensors,
        peft_helper=peft_helper,
        device="cpu",
        dtype=dtype,
        weights_mapper=_TURBO_WEIGHTS_MAPPER,
    )
    _pack_h3_turbo_fc1(lora_model)
    return lora_model, peft_helper


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


def _select_native_file(artifact_path: str | Path) -> Path | None:
    path = Path(artifact_path)
    if path.is_file():
        return path if path.suffix == ".safetensors" else None
    if not path.is_dir():
        return None

    candidate = path / _NATIVE_FILENAME
    return candidate if candidate.is_file() else None


def _native_target_dims(target: str) -> tuple[int, int]:
    if target == "final_layer.adaln_proj.linear":
        return _NATIVE_FINAL_ADALN_DIMS
    if target.endswith("adaln_proj.linear"):
        return _NATIVE_TARGET_DIMS["adaln_proj.linear"]
    for suffix in ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2"):
        if target.endswith(suffix):
            return _NATIVE_TARGET_DIMS[suffix]
    raise ValueError(f"Unsupported MiniMax-H3 native target: {target!r}")


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
    schedule = DMD2SigmaSchedule.from_safetensors_metadata(metadata)
    if schedule is None:
        raise ValueError("MiniMax-H3 native LoRA requires safetensors metadata base_schedule")
    # Request validation speaks in interval counts, so a mislabeled schedule
    # would otherwise silently change the step count instead of failing here.
    if schedule.num_inference_steps != MINIMAX_H3_NATIVE_INFERENCE_STEPS:
        raise ValueError(
            f"MiniMax-H3 native LoRA v1.0 requires a {MINIMAX_H3_NATIVE_INFERENCE_STEPS}-interval base_schedule "
            f"({MINIMAX_H3_NATIVE_INFERENCE_STEPS + 1} sigma positions), got {schedule.num_inference_steps}"
        )
    return schedule


def _validate_native_tensors(checkpoint) -> dict[str, torch.Tensor]:
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
        expected_shape = (_NATIVE_RANK, input_dim) if side == "a" else (output_dim, _NATIVE_RANK)
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(
                f"MiniMax-H3 native tensor has invalid global shape: {name}={tuple(tensor.shape)}, "
                f"expected={expected_shape}"
            )
        tensors[name] = tensor

    incomplete = sorted(target for target, sides in pairs.items() if sides != {"a", "b"})
    if incomplete:
        raise ValueError(f"Incomplete MiniMax-H3 native LoRA pairs: {incomplete}")
    missing = sorted(_NATIVE_EXPECTED_TARGETS - raw_targets)
    unexpected = sorted(raw_targets - _NATIVE_EXPECTED_TARGETS)
    if missing or unexpected:
        raise ValueError(
            "MiniMax-H3 native target set does not match the supported v1.0 artifact: "
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


def load_minimax_h3_native_lora(
    *,
    partition: str,
    lora_request: LoRARequest,
    lora_path: str | Path,
    dtype: torch.dtype,
    unsupported_offload_mode: str | None = None,
) -> tuple[LoRAModel, PEFTHelper, DMD2SigmaSchedule] | None:
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
        if lora_file.name != _NATIVE_FILENAME:
            raise ValueError(f"MiniMax-H3 native LoRA supports only {_NATIVE_FILENAME!r}, got {lora_file.name!r}")
        sigma_schedule = _validate_native_metadata(metadata)
        if partition == "ref2va":
            raise ValueError("MiniMax-H3 native LoRA supports T2VA only")
        if unsupported_offload_mode is not None:
            raise ValueError(f"MiniMax-H3 native dynamic LoRA does not support {unsupported_offload_mode}")
        tensors = _validate_native_tensors(checkpoint)

    peft_helper = PEFTHelper.from_dict(
        {
            "r": _NATIVE_RANK,
            "lora_alpha": _NATIVE_ALPHA,
            "target_modules": _NATIVE_TARGET_PATTERN,
        }
    )
    lora_model = LoRAModel.from_lora_tensors(
        lora_model_id=lora_request.lora_int_id,
        tensors=tensors,
        peft_helper=peft_helper,
        device="cpu",
        dtype=dtype,
        weights_mapper=_NATIVE_WEIGHTS_MAPPER,
    )
    _pack_h3_native_qkv(lora_model)
    _pack_h3_native_fc1(lora_model)
    return lora_model, peft_helper, sigma_schedule
