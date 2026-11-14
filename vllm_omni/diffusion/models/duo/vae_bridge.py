# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Load the official Realization checkpoint VAEs, preserving duplex decode precision."""

import importlib
from pathlib import Path

import torch

from vllm_omni.diffusion.models.minimax_h3.vae import (
    _load_component_config,
    _load_remote_component,
)


def load_codec(kind, root, *, trust_remote_code):
    # Use the official loader without its FP16/uint8 whole-video transforms:
    # duplex uses BF16 temporal decoding and owns the 17-frame output clock.
    path = str(Path(root) / "FL2VA" / f"{kind}_vae")
    config = _load_component_config(path)
    model = _load_remote_component(path, config, trust_remote_code=trust_remote_code).model
    for name in ("latents_mean", "latents_std"):
        model.register_buffer(name, torch.tensor(config[name], dtype=torch.float32), persistent=False)
    model = model.float().eval().requires_grad_(False)
    if kind == "video":
        model.parallel_tiling = False
    return model


def set_tile_group(model, group):
    package = type(model).__module__.rsplit(".", 1)[0]
    state = importlib.import_module(f"{package}.parallel").get_parallel_state()
    state.clear()
    state.update(
        group_size=group.world_size,
        group_rank=group.rank_in_group,
        local_process_group=group.device_group,
        sp_size=group.world_size,
        sp_rank=group.rank_in_group,
        sp_enabled=True,
        sp_process_group=group.device_group,
        tp_size=1,
        tp_rank=0,
    )
    model.parallel_tiling = True

    def gather(tasks, num_tiles):
        # Round-robin ownership determines sizes on the host. Avoid the
        # checkpoint's GPU shape exchange / .tolist() synchronization; the
        # equal-size collective also works in the single-code CUDA graph.
        count = (num_tiles + group.world_size - 1) // group.world_size
        stacked = tasks[0].new_zeros((count, *tasks[0].shape))
        torch.stack(tasks, out=stacked[: len(tasks)])
        values = [torch.empty_like(stacked) for _ in range(group.world_size)]
        torch.distributed.all_gather(values, stacked, group=group.device_group)
        return [values[i % group.world_size][i // group.world_size] for i in range(num_tiles)]

    model._all_gather_tiled_results = gather


def cache_bf16_decoder_weights(model):
    """Keep the same rounded linears that BF16 autocast recreates per block."""
    from vllm_omni.diffusion.models.minimax_h3.ops.vae import _decoder_block_linears

    for linear in _decoder_block_linears(model.decoder):
        linear.to(dtype=torch.bfloat16)
