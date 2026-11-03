# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Parse official LTX distilled adapters without materializing their tensors.

The official LTX adapters are raw safetensors files rather than PEFT
checkpoints.  This module is deliberately LTX-specific: it translates the
official names into the locally constructed Diffusers transformer names and
describes the resulting adapter targets.  Applying those targets is the job
of :mod:`ltx2_phase_adapter`.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from safetensors import safe_open

_IGNORED_OFFICIAL_LORA_MODULES = {"text_embedding_projection.aggregate_embed"}
_OFFICIAL_TO_DIFFUSERS_PREFIXES = (
    ("audio_prompt_adaln_single.", "audio_prompt_adaln."),
    ("prompt_adaln_single.", "prompt_adaln."),
    ("audio_adaln_single.", "audio_time_embed."),
    ("adaln_single.", "time_embed."),
    ("audio_patchify_proj", "audio_proj_in"),
    ("patchify_proj", "proj_in"),
    ("av_ca_video_scale_shift_adaln_single.", "av_cross_attn_video_scale_shift."),
    ("av_ca_audio_scale_shift_adaln_single.", "av_cross_attn_audio_scale_shift."),
    ("av_ca_a2v_gate_adaln_single.", "av_cross_attn_video_a2v_gate."),
    ("av_ca_v2a_gate_adaln_single.", "av_cross_attn_audio_v2a_gate."),
)


@dataclass(frozen=True)
class AdapterTarget:
    """One A/B pair mapped onto a local transformer linear layer."""

    # Keep the original Diffusers module name for diagnostics after LTX packs
    # Q/K/V into the runtime module path.
    source_module: str
    module: str
    slice: str | None
    a_key: str
    b_key: str
    a_shape: tuple[int, ...]
    b_shape: tuple[int, ...]


@dataclass(frozen=True)
class AdapterManifest:
    """Tensor-free description of the fixed refinement adapter."""

    path: str
    targets: tuple[AdapterTarget, ...]


def _to_diffusers_module_name(name: str) -> str:
    for official_prefix, diffusers_prefix in _OFFICIAL_TO_DIFFUSERS_PREFIXES:
        if name.startswith(official_prefix):
            return diffusers_prefix + name[len(official_prefix) :]
    return name


def _resolve_lora_target(transformer: nn.Module, module_name: str) -> tuple[str, str | int | None] | None:
    modules = dict(transformer.named_modules())
    if module_name in modules:
        return module_name, None
    for packed_suffix, sub_suffix, shard_id in getattr(transformer, "stacked_params_mapping", ()):
        if sub_suffix in module_name:
            packed_name = module_name.replace(sub_suffix, packed_suffix)
            if packed_name in modules:
                return packed_name, shard_id
    return None


def parse_ltx_adapter(transformer: nn.Module, path: str) -> AdapterManifest:
    """Translate one official raw LTX adapter into local Transformer targets."""
    path = str(Path(path))
    targets: list[AdapterTarget] = []
    unresolved: list[str] = []
    seen: set[tuple[str, str | int | None]] = set()

    with safe_open(path, framework="pt", device="cpu") as handle:
        tensor_names = set(handle.keys())
        official_name_set = {
            tensor_name.removeprefix("diffusion_model.").rsplit(".lora_", 1)[0]
            for tensor_name in tensor_names
            if tensor_name.endswith(".lora_A.weight")
        }
        official_b_names = {
            tensor_name.removeprefix("diffusion_model.").rsplit(".lora_", 1)[0]
            for tensor_name in tensor_names
            if tensor_name.endswith(".lora_B.weight")
        }
        missing_a = sorted(official_b_names - official_name_set - _IGNORED_OFFICIAL_LORA_MODULES)
        if missing_a:
            raise ValueError(f"Missing paired LoRA A tensors for {missing_a} in {path}.")

        official_names = sorted(official_name_set)
        for official_name in official_names:
            if official_name in _IGNORED_OFFICIAL_LORA_MODULES:
                continue
            key_prefix = f"diffusion_model.{official_name}"
            key_a = f"{key_prefix}.lora_A.weight"
            key_b = f"{key_prefix}.lora_B.weight"
            if key_b not in tensor_names:
                raise ValueError(f"Missing paired LoRA tensor {key_b!r} in {path}.")

            a_shape = tuple(handle.get_slice(key_a).get_shape())
            b_shape = tuple(handle.get_slice(key_b).get_shape())
            if len(a_shape) != 2 or len(b_shape) != 2 or b_shape[1] != a_shape[0]:
                raise ValueError(f"Invalid LoRA pair for {official_name}: A={a_shape}, B={b_shape}.")

            source_module = _to_diffusers_module_name(official_name)
            resolved = _resolve_lora_target(transformer, source_module)
            if resolved is None:
                unresolved.append(official_name)
                continue
            module, shard_id = resolved
            target_id = (module, shard_id)
            if target_id in seen:
                raise ValueError(f"Official LTX LoRA contains duplicate target {module!r}, slice {shard_id!r}.")
            seen.add(target_id)
            targets.append(
                AdapterTarget(
                    source_module=source_module,
                    module=module,
                    slice=shard_id,
                    a_key=key_a,
                    b_key=key_b,
                    a_shape=a_shape,
                    b_shape=b_shape,
                )
            )

    if unresolved:
        raise ValueError(f"Official LTX LoRA contains unmapped modules: {unresolved}.")
    if not targets:
        raise ValueError(f"No Transformer LoRA tensors were found in {path}.")
    return AdapterManifest(path=path, targets=tuple(targets))


def iter_adapter_tensors(manifest: AdapterManifest) -> Iterator[tuple[AdapterTarget, torch.Tensor, torch.Tensor]]:
    """Stream adapter pairs in manifest order without a full checkpoint copy."""

    with safe_open(manifest.path, framework="pt", device="cpu") as handle:
        for target in manifest.targets:
            yield (
                target,
                handle.get_tensor(target.a_key),
                handle.get_tensor(target.b_key),
            )
