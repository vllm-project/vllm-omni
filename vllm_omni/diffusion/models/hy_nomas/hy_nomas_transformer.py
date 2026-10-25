# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from collections import Counter
from collections.abc import Iterable

import regex as re
import torch
import torch.distributed as dist
from omegaconf import DictConfig
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.utils import WeightsMapper

from vllm_omni.diffusion.distributed.parallel_state import get_sp_group
from vllm_omni.experimental.ar_diffusion.kv_cache.segmented import ARDiffusionSegmentedKVState

_DIT_NAMES = WeightsMapper(
    orig_to_new_regex={
        re.compile(r"^double_blocks\.(\d+)\.action_module\.(.+)$"): r"action_modules.\1.\2",
        re.compile(r"\._checkpoint_wrapped_module\."): ".",
        re.compile(r"_checkpoint_wrapped_module\."): "",
    },
)


def build_transformer(config: DictConfig, device: torch.device) -> nn.Module:
    from hymm.models.dit.modules import load_model

    transformer = load_model(config.dit.model_type, config, {"device": device, "dtype": torch.bfloat16})
    transformer.nomas_sparse_config = {"enabled": False}
    transformer.sparse_optim = {"enabled": False}
    transformer.sparse_optim_counts = Counter()
    for block in transformer.double_blocks:
        block.compile_cache_prepost = False
    # The reference output head does not consume the factory's device/dtype kwargs.
    return transformer.to(device=device, dtype=torch.bfloat16).eval().requires_grad_(False)


def load_transformer_weights(transformer: nn.Module, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
    parameters = dict(transformer.named_parameters())
    buffers = dict(transformer.named_buffers())
    loaded = set()
    for name, tensor in _DIT_NAMES.apply(weights):
        if name in loaded:
            raise ValueError(f"Duplicate Hy-Nomas DiT parameter after normalization: {name}")
        target = parameters.get(name, buffers.get(name))
        if target is None:
            raise ValueError(f"Unexpected Hy-Nomas DiT weight: {name}")
        default_weight_loader(target, tensor)
        loaded.add(name)
    missing = parameters.keys() - loaded
    if missing:
        raise ValueError(f"Hy-Nomas checkpoint is missing {len(missing)} DiT parameters: {sorted(missing)[:10]}")
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


def initialize_reference_parallelism(sp_size: int, device: torch.device) -> DeviceMesh | None:
    from hymm.distributed.parallel_states import (
        ParallelDims,
        initialize_sequence_parallel_state_v2,
        nccl_info,
        set_sequence_parallel_state,
        set_tensor_parallel_state,
    )

    dims = ParallelDims(dp_replicate=1, dp_shard=1, sp=sp_size, tp=1, pp=1, world_size=sp_size)
    mesh = None
    if sp_size > 1:
        group = get_sp_group()
        if group.ulysses_world_size != sp_size or group.world_size != sp_size:
            raise ValueError("Hy-Nomas requires the initialized SP group to match ulysses_degree.")
        # Reuse the worker's group; vLLM-Omni owns its creation and destruction.
        mesh = DeviceMesh.from_group(group.ulysses_group, device.type, mesh_dim_names=("sp",))
        dims.sp_mesh = mesh
        dims.sp_group = group.ulysses_group
    else:
        dims.sp_group = None
    dims.device_mesh = mesh
    # The upstream single-device path leaves stale SP state when a model is reloaded.
    set_sequence_parallel_state(False)
    set_tensor_parallel_state(False)
    nccl_info.sp_group = None
    nccl_info.tp_group = None
    nccl_info.use_dynamic_ring_attention = False
    nccl_info.sp_stream = None
    nccl_info.sp_rank_list = None
    initialize_sequence_parallel_state_v2(dims, mesh)
    # Use the initialized worker rank instead of the torchrun RANK environment variable.
    nccl_info.global_rank = dist.get_rank() if dist.is_initialized() else 0
    return mesh
