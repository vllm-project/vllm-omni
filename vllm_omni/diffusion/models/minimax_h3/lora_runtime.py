# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from safetensors import safe_open

from vllm_omni.diffusion.lora_runtime import (
    DiffusionLoRABindingPlan,
    DiffusionLoRADeployment,
    DiffusionLoRASupport,
    LoadedDiffusionLoRA,
    LowRankUpdate,
    create_low_rank_executor,
)

_H3_TURBO_RANK = 128
_H3_TURBO_ALPHA = 128.0
_DIFFUSERS_LORA_A_SUFFIX = ".lora_A.default.weight"
_DIFFUSERS_LORA_B_SUFFIX = ".lora_B.default.weight"
_NATIVE_LORA_A_SUFFIX = ".lora_A.weight"
_NATIVE_LORA_B_SUFFIX = ".lora_B.weight"
_H3_LOGICAL_TARGETS = frozenset({"to_q", "to_k", "to_v", "out_proj", "fc1", "fc2"})


@dataclass(frozen=True)
class _MiniMaxH3LoRAFormat:
    """One checkpoint publication format, not one deployed LoRA identity.

    Multiple adapters may share a format and decoder. The common runtime keeps
    those weight instances separate by ``DiffusionLoRADeployment.name``;
    format matching never inspects that user-provided name or artifact path.
    """

    format_id: str
    matches_metadata: Callable[[Mapping[str, str]], bool]
    decode_checkpoint: Callable[[Any, dict[str, str]], list[LowRankUpdate]]

MINIMAX_H3_LORA_BINDING_PLAN = DiffusionLoRABindingPlan(
    component_names=("transformer",),
    target_modules=("to_q", "to_k", "to_v", "out_proj", "fc1", "fc2"),
    packed_modules_mapping={"qkv_proj": ("to_q", "to_k", "to_v")},
)


def _normalize_h3_target(raw_target: str) -> str:
    if raw_target.startswith("transformer_blocks."):
        target = "blocks." + raw_target.removeprefix("transformer_blocks.")
    elif raw_target.startswith("token_refiner.refiner_blocks."):
        target = "token_refiner.blocks." + raw_target.removeprefix("token_refiner.refiner_blocks.")
    else:
        raise ValueError(f"Unsupported MiniMax-H3 Turbo LoRA target prefix: {raw_target!r}")

    replacements = (
        (".attn.to_out.0", ".attn.out_proj"),
        (".ff.net.0.proj", ".mlp.fc1"),
        (".ff.net.2", ".mlp.fc2"),
    )
    for old, new in replacements:
        target = target.replace(old, new)
    leaf = target.rsplit(".", 1)[-1]
    if leaf not in _H3_LOGICAL_TARGETS:
        raise ValueError(f"Unsupported MiniMax-H3 Turbo LoRA logical target: {target!r}")
    return target


def _select_h3_lora_file(artifact_path: Path) -> Path:
    if artifact_path.is_file():
        if artifact_path.suffix != ".safetensors":
            raise ValueError(f"MiniMax-H3 LoRA must be a safetensors file, got {artifact_path}")
        return artifact_path
    if not artifact_path.is_dir():
        raise ValueError(f"MiniMax-H3 LoRA artifact does not exist: {artifact_path}")

    candidates = sorted(artifact_path.glob("*v1.0*.safetensors"))
    if not candidates:
        candidates = sorted(artifact_path.glob("*.safetensors"))
    if len(candidates) != 1:
        raise ValueError(
            f"MiniMax-H3 LoRA artifact must resolve to exactly one safetensors file, "
            f"found {[path.name for path in candidates]}"
        )
    return candidates[0]


def _collect_pairs(
    checkpoint,
    *,
    a_suffix: str,
    b_suffix: str,
) -> dict[str, dict[str, torch.Tensor]]:
    paired: dict[str, dict[str, torch.Tensor]] = {}
    for key in checkpoint.keys():
        if key.endswith(a_suffix):
            target = key[: -len(a_suffix)]
            side = "a"
        elif key.endswith(b_suffix):
            target = key[: -len(b_suffix)]
            side = "b"
        else:
            raise ValueError(f"Unconsumed MiniMax-H3 LoRA tensor: {key!r}")
        target_pair = paired.setdefault(target, {})
        if side in target_pair:
            raise ValueError(f"Duplicate MiniMax-H3 LoRA tensor for {target}.{side}")
        target_pair[side] = checkpoint.get_tensor(key)
    return paired


def _validate_pair(
    target: str,
    tensors: dict[str, torch.Tensor],
    *,
    expected_rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if set(tensors) != {"a", "b"}:
        raise ValueError(f"Incomplete MiniMax-H3 LoRA pair for {target}: {sorted(tensors)}")
    lora_a = tensors["a"]
    lora_b = tensors["b"]
    if lora_a.ndim != 2 or lora_b.ndim != 2 or lora_a.shape[0] != expected_rank:
        raise ValueError(
            f"MiniMax-H3 LoRA rank must be {expected_rank}, "
            f"got A={tuple(lora_a.shape)}, B={tuple(lora_b.shape)} for {target}"
        )
    if lora_b.shape[1] != expected_rank:
        raise ValueError(f"MiniMax-H3 LoRA B rank mismatch for {target}: {tuple(lora_b.shape)}")
    return lora_a, lora_b


def _load_turbo_updates(checkpoint, metadata: dict[str, str]) -> list[LowRankUpdate]:
    raw_alpha = metadata.get("alpha")
    try:
        alpha = float(raw_alpha) if raw_alpha is not None else math.nan
    except ValueError as exc:
        raise ValueError(f"MiniMax-H3 Turbo alpha must be numeric, got {raw_alpha!r}") from exc
    if alpha != _H3_TURBO_ALPHA:
        raise ValueError(f"MiniMax-H3 Turbo v1.0 requires alpha={_H3_TURBO_ALPHA:g}, got {raw_alpha!r}")

    paired: dict[str, dict[str, torch.Tensor]] = {}
    for raw_target, tensors in _collect_pairs(
        checkpoint,
        a_suffix=_DIFFUSERS_LORA_A_SUFFIX,
        b_suffix=_DIFFUSERS_LORA_B_SUFFIX,
    ).items():
        target = _normalize_h3_target(raw_target)
        if target in paired:
            raise ValueError(f"Duplicate normalized MiniMax-H3 LoRA target: {target}")
        if target.endswith(".mlp.fc1") and "b" in tensors:
            tensor = tensors["b"]
            if tensor.shape[0] % 2:
                raise ValueError(f"MiniMax-H3 Turbo fc1 lora_B rows must split evenly, got {tuple(tensor.shape)}")
            value, gate = tensor.chunk(2, dim=0)
            tensors["b"] = torch.cat((gate, value), dim=0).contiguous()
        paired[target] = tensors

    updates: list[LowRankUpdate] = []
    for target, tensors in sorted(paired.items()):
        lora_a, lora_b = _validate_pair(target, tensors, expected_rank=_H3_TURBO_RANK)
        updates.append(
            LowRankUpdate(
                component="transformer",
                logical_target=target,
                lora_a=lora_a,
                lora_b=lora_b,
                intrinsic_scale=alpha / _H3_TURBO_RANK,
            )
        )
    return updates


def _load_native_updates(checkpoint, metadata: dict[str, str]) -> list[LowRankUpdate]:
    if metadata.get("base_model") != "minimax-h3-fl2va":
        raise ValueError(
            f"Unsupported native MiniMax-H3 LoRA base model {metadata.get('base_model')!r}; expected 'minimax-h3-fl2va'"
        )
    raw_rank = metadata.get("lora_rank")
    try:
        rank = int(raw_rank) if raw_rank is not None else 0
    except ValueError as exc:
        raise ValueError(f"MiniMax-H3 LoRA rank must be an integer, got {raw_rank!r}") from exc
    if rank <= 0:
        raise ValueError(f"MiniMax-H3 LoRA rank must be positive, got {raw_rank!r}")
    raw_alpha = metadata.get("lora_alpha", metadata.get("alpha"))
    try:
        intrinsic_scale = float(raw_alpha) / rank if raw_alpha is not None else 1.0
    except ValueError as exc:
        raise ValueError(f"MiniMax-H3 LoRA alpha must be numeric, got {raw_alpha!r}") from exc
    if not math.isfinite(intrinsic_scale) or intrinsic_scale <= 0:
        raise ValueError(f"MiniMax-H3 LoRA intrinsic scale must be positive and finite, got {intrinsic_scale!r}")

    updates: list[LowRankUpdate] = []
    for raw_target, tensors in sorted(
        _collect_pairs(
            checkpoint,
            a_suffix=_NATIVE_LORA_A_SUFFIX,
            b_suffix=_NATIVE_LORA_B_SUFFIX,
        ).items()
    ):
        if not raw_target.startswith(("diffusion_model.blocks.", "diffusion_model.token_refiner.blocks.")):
            raise ValueError(f"Unsupported native MiniMax-H3 LoRA target prefix: {raw_target!r}")
        target = raw_target.removeprefix("diffusion_model.")
        lora_a, lora_b = _validate_pair(target, tensors, expected_rank=rank)
        leaf = target.rsplit(".", 1)[-1]
        if leaf == "qkv_proj":
            if lora_b.shape[0] % 3:
                raise ValueError(f"MiniMax-H3 qkv_proj lora_B rows must split evenly, got {tuple(lora_b.shape)}")
            prefix = target.rsplit(".", 1)[0]
            for logical_leaf, piece in zip(("to_q", "to_k", "to_v"), lora_b.chunk(3, dim=0), strict=True):
                updates.append(
                    LowRankUpdate(
                        component="transformer",
                        logical_target=f"{prefix}.{logical_leaf}",
                        lora_a=lora_a,
                        lora_b=piece,
                        intrinsic_scale=intrinsic_scale,
                    )
                )
        elif leaf in _H3_LOGICAL_TARGETS:
            updates.append(
                LowRankUpdate(
                    component="transformer",
                    logical_target=target,
                    lora_a=lora_a,
                    lora_b=lora_b,
                    intrinsic_scale=intrinsic_scale,
                )
            )
        else:
            raise ValueError(f"Unsupported native MiniMax-H3 LoRA logical target: {target!r}")
    return updates


def _is_turbo_v1_format(metadata: Mapping[str, str]) -> bool:
    return metadata.get("key_format") == "minimax-h3-diffusers"


def _is_native_fl2va_format(metadata: Mapping[str, str]) -> bool:
    return metadata.get("base_model") == "minimax-h3-fl2va" and "lora_rank" in metadata


# This is the complete H3 checkpoint-format declaration. Add a new entry when
# another H3 publication requires different loading semantics; do not add one
# merely for another adapter that already uses an existing format.
MINIMAX_H3_LORA_FORMATS = (
    _MiniMaxH3LoRAFormat(
        format_id="lightx2v-turbo-v1",
        matches_metadata=_is_turbo_v1_format,
        decode_checkpoint=_load_turbo_updates,
    ),
    _MiniMaxH3LoRAFormat(
        format_id="native-fl2va",
        matches_metadata=_is_native_fl2va_format,
        decode_checkpoint=_load_native_updates,
    ),
)


def _select_h3_lora_format(metadata: Mapping[str, str]) -> _MiniMaxH3LoRAFormat:
    """Select exactly one decoder from metadata, independent of adapter name."""

    matched_formats = tuple(
        format_spec for format_spec in MINIMAX_H3_LORA_FORMATS if format_spec.matches_metadata(metadata)
    )
    if not matched_formats:
        supported = [format_spec.format_id for format_spec in MINIMAX_H3_LORA_FORMATS]
        raise ValueError(f"Unsupported MiniMax-H3 LoRA metadata keys: {sorted(metadata)}; supported formats={supported}")
    if len(matched_formats) != 1:
        raise ValueError(
            f"Ambiguous MiniMax-H3 LoRA format: {[format_spec.format_id for format_spec in matched_formats]}"
        )
    return matched_formats[0]


class MiniMaxH3LoRALoader:
    """Interpret supported Diffusers and native-layout H3 FL2VA LoRAs."""

    def __init__(self, pipeline: nn.Module) -> None:
        partition = getattr(pipeline, "partition", None)
        if partition == "ref2va":
            raise ValueError("MiniMax-H3 diffusion LoRA runtime currently supports FL2VA only")

    def load(
        self,
        deployment: DiffusionLoRADeployment,
        artifact_path: Path,
    ) -> LoadedDiffusionLoRA:
        lora_file = _select_h3_lora_file(artifact_path)
        with safe_open(lora_file, framework="pt", device="cpu") as checkpoint:
            metadata = checkpoint.metadata() or {}
            format_spec = _select_h3_lora_format(metadata)
            updates = format_spec.decode_checkpoint(checkpoint, metadata)
        if not updates:
            raise ValueError(f"MiniMax-H3 LoRA {lora_file} contains no supported updates")
        return LoadedDiffusionLoRA(name=deployment.name, updates=tuple(updates))


def create_minimax_h3_lora_loader(pipeline: nn.Module) -> MiniMaxH3LoRALoader:
    return MiniMaxH3LoRALoader(pipeline)


MINIMAX_H3_DIFFUSION_LORA_SUPPORT = DiffusionLoRASupport(
    loader_factory=create_minimax_h3_lora_loader,
    binding_plan=MINIMAX_H3_LORA_BINDING_PLAN,
    executor_factory=create_low_rank_executor,
    supports_composition=True,
)
