# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from collections.abc import Iterable

import regex as re
import torch
from omegaconf import DictConfig
from torch import nn
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.utils import WeightsMapper

from vllm_omni.diffusion.distributed.parallel_state import get_sp_group
from vllm_omni.diffusion.models.hy_nomas.dit import HyNomasTransformer
from vllm_omni.diffusion.models.hy_nomas.layers import get_flash_attention2
from vllm_omni.experimental.ar_diffusion.kv_cache.segmented import ARDiffusionSegmentedKVState

_DIT_NAMES = WeightsMapper(
    orig_to_new_regex={
        re.compile(r"^(double_blocks\.\d+)\.attn\.((?:img|txt)_attn_.+)$"): r"\1.\2",
        re.compile(r"^(txt_in\.individual_token_refiner\.blocks\.\d+)\.attn\.(self_attn_.+)$"): r"\1.\2",
        re.compile(r"^double_blocks\.(\d+)\.action_module\.(.+)$"): r"action_modules.\1.\2",
        re.compile(r"\._checkpoint_wrapped_module\."): ".",
        re.compile(r"_checkpoint_wrapped_module\."): "",
    },
)


def build_transformer(config: DictConfig, device: torch.device, model_config: dict) -> nn.Module:
    if device.type == "cuda":
        get_flash_attention2()
    transformer = HyNomasTransformer(
        model_config, sp_size=config.parallelism.sp_size, device=device, dtype=torch.bfloat16
    )
    return transformer.to(device=device, dtype=torch.bfloat16).eval().requires_grad_(False)


def load_transformer_weights(transformer: nn.Module, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
    return load_component_weights(transformer, _DIT_NAMES.apply(weights))


def load_component_weights(module: nn.Module, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
    parameters = dict(module.named_parameters())
    buffers = dict(module.named_buffers())
    loaded = set()
    for name, tensor in weights:
        if name in loaded:
            raise ValueError(f"Duplicate Hy-Nomas parameter after normalization: {name}")
        target = parameters.get(name, buffers.get(name))
        if target is None:
            raise ValueError(f"Unexpected Hy-Nomas weight: {name}")
        default_weight_loader(target, tensor)
        loaded.add(name)
    missing = parameters.keys() - loaded
    if missing:
        raise ValueError(f"Hy-Nomas checkpoint is missing {len(missing)} parameters: {sorted(missing)[:10]}")
    return loaded


def reference_cache_views(cache: ARDiffusionSegmentedKVState, current_tokens: int) -> list[dict[str, torch.Tensor]]:
    # History is runner-owned; the unwritten suffix remains scratch until a clean forward succeeds.
    cache.prepare_layer_views(current_tokens, 0)
    scratch = torch.empty(
        (current_tokens, cache.cache.num_kv_heads, cache.cache.head_size),
        dtype=cache.cache.dtype,
        device=cache.cache.device,
    )
    empty = scratch[:0]
    views = []
    for layer in range(cache.cache.num_layers):
        key, value = cache.assemble(layer, empty, empty, scratch, scratch)
        views.append({"k": key.unsqueeze(0), "v": value.unsqueeze(0)})
    return views


def commit_cache(cache: ARDiffusionSegmentedKVState, views: list[dict[str, torch.Tensor]]) -> None:
    start = cache.history_tokens
    for layer, view in enumerate(views):
        key, value = view["k"][0, start:], view["v"][0, start:]
        cache.assemble(layer, key[:0], value[:0], key, value)
        cache.stage(layer)


def validate_sequence_parallelism(sp_size: int) -> None:
    if sp_size > 1:
        group = get_sp_group()
        if group.ulysses_world_size != sp_size or group.world_size != sp_size:
            raise ValueError("Hy-Nomas requires the initialized SP group to match ulysses_degree.")
