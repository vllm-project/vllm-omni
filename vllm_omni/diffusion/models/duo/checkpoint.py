# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Strict loader for exported Torchtitan split-QKV Realization student LoRA.

Mappings follow chy/train convert.py. DeCFG is training-only and excluded.
Original base weights must be loaded first; no adapter merge or stacking.
"""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors import safe_open
from torch import nn

from .kernels import split_lora_epilogue
from .weights import FixedRealizationLoRALinear, file_sha256, logical_weight_shape


def _targets(name):
    name = name.replace("token_refiner.blocks.", "token_refiner.refiner_blocks.")
    if name.startswith("blocks."):
        name = name.replace("blocks.", "transformer_blocks.", 1)
    mapping = {
        "video_patch_proj": "proj_in",
        "audio_patch_proj": "audio_proj_in",
        "condition_proj": "context_embedder",
        "time_embedder.proj_in": "time_embedder.linear_1",
        "time_embedder.proj_out": "time_embedder.linear_2",
        "final_layer.adaln_proj.linear": "norm_out.linear",
        "final_layer.video_out": "proj_out",
        "final_layer.audio_out": "audio_proj_out",
    }
    if name in mapping:
        return (mapping[name],)
    if name.endswith(".attn.qkv_proj"):
        prefix = name.removesuffix("qkv_proj")
        return tuple(prefix + "to_" + side for side in ("q", "k", "v"))
    name = name.replace(".attn.out_proj", ".attn.to_out.0")
    name = name.replace(".mlp.fc1", ".ff.net.0.proj").replace(".mlp.fc2", ".ff.net.2")
    return (name,)


class SplitQKVRealizationLoRALinear(nn.Module):
    """Keep independent trained A/B per Q/K/V and native column TP shards."""

    def __init__(self, base, pairs, scale):
        super().__init__()
        self.base_layer = base
        self.scale = scale
        self.lora_a = nn.ParameterList()
        self.lora_b = nn.ParameterList()
        for a, b in pairs:
            if base.tp_size > 1:
                b = b.chunk(base.tp_size, 0)[base.tp_rank].contiguous()
            self.lora_a.append(nn.Parameter(a.to(base.weight.device).clone(), requires_grad=False))
            self.lora_b.append(nn.Parameter(b.to(base.weight.device).clone(), requires_grad=False))

    @property
    def weight(self):
        return self.base_layer.weight

    def forward(self, value):
        base, bias = self.base_layer(value)
        x = value.float()
        deltas = [
            torch.nn.functional.linear(torch.nn.functional.linear(x, a), b)
            for a, b in zip(self.lora_a, self.lora_b, strict=True)
        ]
        return split_lora_epilogue(base, deltas, self.scale), bias


@torch.no_grad()
def load_training_adapter(
    model, path, *, rank=128, alpha=128, role_embedding=False, target_modules=None, manifest_modules=None
):
    if not getattr(model, "_realization_base_loaded", False) or getattr(model, "_realization_adapter_loaded", False):
        raise ValueError("Training adapter requires complete base and cannot be stacked")
    if rank != 128 or alpha != 128 or role_embedding is not False:
        raise ValueError("Exported causal student requires rank/alpha128 and no role embedding")
    all_linears = {
        n: m for n, m in model.named_modules() if hasattr(m, "weight") and m.weight.ndim == 2 and n != "role_embeddings"
    }
    if target_modules is not None:
        all_linears = {
            name: module
            for name, module in all_linears.items()
            if all(
                any(target == suffix or target.endswith("." + suffix) for suffix in target_modules)
                for target in _targets(name)
            )
        }
    # The released adapter covers a manifest-declared subset of the projections
    # (the causal-base export trains the media/attention/FFN projections and
    # leaves the AdaLN, time-embedder and output-norm linears at their base
    # weights). Deriving the inventory from every 2D weight in the module tree
    # would demand LoRA tensors that the checkpoint deliberately does not have,
    # so the file itself defines the set and `manifest_modules` only verifies it.
    by_target = {t: n for n in all_linears for t in _targets(n)}
    expected = {"role_embeddings.weight"}
    replacements = {}
    with safe_open(str(path), framework="pt", device="cpu") as source:
        meta = source.metadata() or {}
        keys = set(source.keys())
        covered = set()
        for key in keys:
            for side in ("A", "B"):
                suffix = f".lora_{side}.default.weight"
                if key.endswith(suffix):
                    target = key[: -len(suffix)]
                    if target not in by_target:
                        raise ValueError("Adapter targets a projection the model does not have: " + target)
                    covered.add(target)
        unknown = covered - set(by_target)
        if unknown:
            raise ValueError("Adapter targets unknown projections: " + ", ".join(sorted(unknown)[:5]))
        modules = {by_target[t]: all_linears[by_target[t]] for t in sorted(covered)}
        expected |= {t + f".lora_{side}.default.weight" for t in covered for side in ("A", "B")}
        if manifest_modules is not None:
            declared = {t for name in manifest_modules for t in _targets(name)}
            if declared != covered:
                raise ValueError(
                    "Adapter inventory disagrees with the release manifest: "
                    f"missing={sorted(declared - covered)[:5]}, extra={sorted(covered - declared)[:5]}"
                )
        if (
            meta.get("layout") != "diffusers_split_qkv"
            or meta.get("rank") != "128"
            or meta.get("alpha") != "128"
            or meta.get("role_embedding") != "false"
        ):
            raise ValueError("Missing explicit exported student layout/rank/role identity")
        if keys != expected:
            raise ValueError(
                "Training adapter inventory mismatch: "
                f"missing={sorted(expected - keys)}, extra={sorted(keys - expected)}"
            )
    replacements = {}
    with safe_open(str(path), framework="pt", device="cpu") as source:
        if (
            meta.get("layout") != "diffusers_split_qkv"
            or meta.get("rank") != "128"
            or meta.get("alpha") != "128"
            or meta.get("role_embedding") != "false"
        ):
            raise ValueError("Missing explicit exported student layout/rank/role identity")
        if set(source.keys()) != expected:
            raise ValueError(
                "Training adapter inventory mismatch: "
                f"missing={sorted(expected - set(source.keys()))}, "
                f"extra={sorted(set(source.keys()) - expected)}"
            )
        for name, module in modules.items():
            rows, cols = logical_weight_shape(module)
            targets = _targets(name)
            pairs = []
            for target in targets:
                a = source.get_tensor(target + ".lora_A.default.weight")
                b = source.get_tensor(target + ".lora_B.default.weight")
                wanted_rows = rows // len(targets)
                if a.shape != (rank, cols) or b.shape != (wanted_rows, rank):
                    raise ValueError("Training adapter geometry mismatch: " + target)
                if (
                    a.dtype != torch.float32
                    or b.dtype != torch.float32
                    or not torch.isfinite(a).all()
                    or not torch.isfinite(b).all()
                ):
                    raise ValueError("Training adapter requires finite FP32: " + target)
                if name.endswith(".mlp.fc1"):
                    # Diffusers [value;gate] -> native [gate;value].
                    value, gate = b.chunk(2, dim=0)
                    b = torch.cat((gate, value), dim=0).contiguous()
                pairs.append((a, b))
            replacements[name] = (
                SplitQKVRealizationLoRALinear(module, pairs, alpha / rank)
                if len(pairs) == 3
                else FixedRealizationLoRALinear(module, *pairs[0], alpha / rank)
            )
        role = source.get_tensor("role_embeddings.weight")
        if (
            role.shape != model.role_embeddings.weight.shape
            or role.dtype != torch.float32
            or not torch.isfinite(role).all()
        ):
            raise ValueError("Invalid student role embedding tensor")
    for name, replacement in replacements.items():
        model.set_submodule(name, replacement)
    model.role_embeddings.weight.copy_(role.to(model.role_embeddings.weight.device))
    model.role_embedding_enabled = False
    model._realization_adapter_loaded = True
    return dict(
        path=str(Path(path).resolve()),
        weights_sha256=file_sha256(path),
        tensor_count=len(expected),
        projections=len(modules),
        target_modules=target_modules,
        rank=rank,
        alpha=alpha,
        qkv_layout="independent_logical_q_k_v",
        role_embedding=False,
        checkpoint_id=meta.get("checkpoint_id"),
        decfg_inference=False,
    )
