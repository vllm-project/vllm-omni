# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Loader for the LightX2V Turbo MiniMax-H3 LoRA.

Keyed by ``key_format=minimax-h3-diffusers``. The native FlashGen contract lives
in ``.npu.lora``; the two artifacts differ in rank, target naming and QKV
layout, so each owns its own parsing and packing.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
from safetensors import safe_open
from vllm.lora.lora_model import LoRAModel
from vllm.lora.lora_weights import PackedLoRALayerWeights
from vllm.lora.peft_helper import PEFTHelper
from vllm.model_executor.models.utils import WeightsMapper

from vllm_omni.lora.request import LoRARequest

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
