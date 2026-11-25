# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""LongCat-Image merged-row TP composed with Ulysses SP, on four real ranks.

LongCat's single-stream blocks shard ``proj_mlp`` / ``proj_out`` on the feature
dimension (``MergedRowParallelLinear``) and are not gated on SP, unlike FLUX.2-dev's
local TP. ``_sp_plan`` shards the sequence dimension at the model level. This runs a
small LongCat-Image transformer at TP=2 x Ulysses=2 through its own ``load_weights``
and ``_sp_plan`` hooks, and checks that every rank's output matches the TP=1 / SP=1
run on the same checkpoint and inputs.
"""

from __future__ import annotations

import socket
from typing import Any

import pytest
import torch

from tests.helpers.mark import hardware_marks
from vllm_omni.diffusion.data import (
    AttentionConfig,
    AttentionSpec,
    DiffusionParallelConfig,
    OmniDiffusionConfig,
    TransformerConfig,
)

# Selected by the CUDA Ready lane "Diffusion · Distributed & Attention Test · 4-GPU"
# (`core_model and cuda and L4 and cards_4`). No ROCm or XPU lane runs 4-card
# diffusion tests at the core_model level, so those platforms are not marked.
_FOUR_CARD = hardware_marks(res={"cuda": "L4"}, num_cards=4)

# Heads divide by TP x Ulysses; attention_head_dim == sum(axes_dims_rope).
_CONFIG: dict[str, Any] = dict(
    patch_size=1,
    in_channels=8,
    num_layers=1,
    num_single_layers=2,
    attention_head_dim=16,
    num_attention_heads=4,
    joint_attention_dim=16,
    pooled_projection_dim=8,
    axes_dims_rope=[4, 6, 6],
)
_IMAGE_SEQ_LEN = 8  # divisible by the Ulysses degree; LongCat's plan does not pad
_TEXT_SEQ_LEN = 3


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _od_config(tp: int, ulysses: int) -> OmniDiffusionConfig:
    return OmniDiffusionConfig(
        tf_model_config=TransformerConfig(params=dict(_CONFIG)),
        parallel_config=DiffusionParallelConfig(tensor_parallel_size=tp, ulysses_degree=ulysses),
        diffusion_attention_config=AttentionConfig(default=AttentionSpec(backend="TORCH_SDPA")),
    )


def _make_checkpoint(model) -> dict[str, torch.Tensor]:
    """Diffusers-style checkpoint: unsharded tensors, fused QKV split into q/k/v."""
    torch.manual_seed(7)
    checkpoint: dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        weight = torch.randn_like(param, device="cpu") * 0.05
        if ".to_qkv." in name or ".add_kv_proj." in name:
            fused = ".to_qkv." if ".to_qkv." in name else ".add_kv_proj."
            parts = ("to_q", "to_k", "to_v") if fused == ".to_qkv." else ("add_q_proj", "add_k_proj", "add_v_proj")
            prefix, _, suffix = name.rpartition(fused)
            for part, tensor in zip(parts, torch.chunk(weight, 3, dim=0)):
                checkpoint[f"{prefix}.{part}.{suffix}"] = tensor
        else:
            checkpoint[name] = weight
    return checkpoint


def _make_inputs(device: torch.device) -> dict[str, torch.Tensor]:
    torch.manual_seed(1234)
    ids = torch.stack(torch.meshgrid(torch.arange(2), torch.arange(4), indexing="ij"), dim=-1).reshape(-1, 2)
    img_ids = torch.cat([torch.zeros(_IMAGE_SEQ_LEN, 1), ids.float()], dim=-1)
    kwargs = dict(
        hidden_states=torch.randn(1, _IMAGE_SEQ_LEN, _CONFIG["in_channels"]),
        encoder_hidden_states=torch.randn(1, _TEXT_SEQ_LEN, _CONFIG["joint_attention_dim"]),
        timestep=torch.tensor([0.5]),
        img_ids=img_ids,
        txt_ids=torch.zeros(_TEXT_SEQ_LEN, 3),
    )
    return {name: tensor.to(device) for name, tensor in kwargs.items()}


def _worker(rank: int, world_size: int, tp: int, ulysses: int, port: int, checkpoint, queue) -> None:
    from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config

    from vllm_omni.diffusion.config import set_current_diffusion_config
    from vllm_omni.diffusion.distributed.parallel_state import (
        destroy_distributed_env,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm_omni.diffusion.distributed.sp_plan import SequenceParallelConfig
    from vllm_omni.diffusion.forward_context import get_forward_context, set_forward_context
    from vllm_omni.diffusion.hooks.sequence_parallel import apply_sequence_parallel
    from vllm_omni.diffusion.models.longcat_image.longcat_image_transformer import LongCatImageTransformer2DModel
    from vllm_omni.platforms import current_omni_platform

    device = torch.device(f"{current_omni_platform.device_type}:{rank}")
    current_omni_platform.set_device(device)
    with set_current_vllm_config(VllmConfig(device_config=DeviceConfig(device=current_omni_platform.device_type))):
        init_distributed_environment(
            world_size=world_size, rank=rank, local_rank=rank, distributed_init_method=f"tcp://127.0.0.1:{port}"
        )
        try:
            initialize_model_parallel(tensor_parallel_size=tp, ulysses_degree=ulysses)
            od_config = _od_config(tp, ulysses)
            torch.manual_seed(0)
            # Build inside a forward context, as the diffusion worker does: attention
            # layers pick their (Ulysses) parallel strategy at construction.
            with set_current_diffusion_config(od_config), set_forward_context(omni_diffusion_config=od_config):
                model = LongCatImageTransformer2DModel(od_config=od_config).to(device).eval()
            if checkpoint is None:
                checkpoint = _make_checkpoint(model)
            loaded = model.load_weights([(name, weight.clone()) for name, weight in checkpoint.items()])
            if ulysses > 1:
                apply_sequence_parallel(model, SequenceParallelConfig(ulysses_degree=ulysses), model._sp_plan)
            seen_image_lens: list[int] = []
            hook = model.single_transformer_blocks[0].register_forward_pre_hook(
                lambda module, args, kwargs: seen_image_lens.append(kwargs["hidden_states"].shape[1]), with_kwargs=True
            )
            with torch.no_grad(), set_forward_context(omni_diffusion_config=od_config):
                get_forward_context().sp_plan_hooks_applied = ulysses > 1
                (output,) = model(**_make_inputs(device), return_dict=False)
            hook.remove()
            merged = [getattr(block.proj_out, "tp_size", 1) for block in model.single_transformer_blocks]
            queue.put(
                dict(
                    rank=rank,
                    checkpoint=checkpoint if rank == 0 else None,
                    unloaded_params=sorted(set(dict(model.named_parameters())) - loaded),
                    output=output.float().cpu().clone(),
                    proj_out_tp_sizes=merged,
                    single_block_image_len=seen_image_lens[0],
                )
            )
        finally:
            destroy_distributed_env()


def _run(world_size: int, tp: int, ulysses: int, checkpoint) -> list[dict]:
    ctx = torch.multiprocessing.get_context("spawn")
    with ctx.Manager() as manager:
        queue = manager.Queue()
        torch.multiprocessing.spawn(
            _worker, args=(world_size, tp, ulysses, _find_free_port(), checkpoint, queue), nprocs=world_size
        )
        return sorted((queue.get() for _ in range(world_size)), key=lambda report: report["rank"])


@pytest.mark.core_model
@pytest.mark.diffusion
@pytest.mark.parallel
@pytest.mark.parametrize("world_size", [pytest.param(4, marks=_FOUR_CARD)])
def test_longcat_merged_row_tp2_ulysses2_matches_tp1(world_size: int):
    (baseline,) = _run(1, 1, 1, None)
    assert not baseline["unloaded_params"], baseline["unloaded_params"]
    reports = _run(world_size, 2, 2, baseline["checkpoint"])
    for report in reports:
        rank = report["rank"]
        assert report["proj_out_tp_sizes"] == [2] * _CONFIG["num_single_layers"], f"[rank {rank}] merged-row TP off"
        assert not report["unloaded_params"], f"[rank {rank}] unloaded params: {report['unloaded_params']}"
        # The single-stream blocks really ran on this rank's sequence shard.
        assert report["single_block_image_len"] == _IMAGE_SEQ_LEN // 2, f"[rank {rank}] SP did not shard"
        torch.testing.assert_close(report["output"], baseline["output"], rtol=2e-3, atol=2e-3, msg=f"[rank {rank}]")
