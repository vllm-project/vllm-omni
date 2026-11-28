# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Compare the adapters with unsharded Diffusers using dense reference attention."""

import copy
from contextlib import ExitStack, nullcontext
from itertools import product
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from diffusers.models.transformers.transformer_hunyuan_video import HunyuanVideoTransformer3DModel
from diffusers.models.transformers.transformer_ltx import LTXVideoTransformer3DModel

from vllm_omni.diffusion.models.diffusers_adapter import hunyuan_sparsp, ltx_sparsp

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def _gather(value, dim=1, group=None):
    shards = [torch.empty_like(value.contiguous()) for _ in range(dist.get_world_size(group))]
    dist.all_gather(shards, value.contiguous(), group=group)
    return torch.cat(shards, dim=dim)


class _DenseRing:
    def __init__(self, **kwargs):
        self.scale = kwargs["softmax_scale"]

    def __call__(self, q, k, v, meta=None):
        out = F.scaled_dot_product_attention(
            q.transpose(1, 2), _gather(k).transpose(1, 2), _gather(v).transpose(1, 2), scale=self.scale
        )
        return out.transpose(1, 2)


def _check_adapter(model, module, adapter, role, inputs, device):
    model = model.eval().to(device)
    ctx = SimpleNamespace(sparsp_video_layout=None)
    group = SimpleNamespace(
        ring_rank=dist.get_rank(), ring_world_size=2, ring_group=dist.group.WORLD, all_gather=_gather
    )
    for placement, fused in product(("none", "interval_periodic"), (False, True)):
        spec = SimpleNamespace(sparsp=SimpleNamespace(placement=placement))
        config = SimpleNamespace(diffusion_attention_config=SimpleNamespace(per_role={role: spec}))
        local = copy.deepcopy(model)
        if fused:
            if role == "ltx.video_self":
                for block in local.transformer_blocks:
                    block.attn1.fuse_projections()
            else:
                for block in (*local.transformer_blocks, *local.single_transformer_blocks):
                    block.attn.fuse_projections()
        with (
            patch.object(module, "get_forward_context", return_value=ctx),
            patch.object(module, "get_sp_group", return_value=group),
            patch.object(module, "set_current_diffusion_config", return_value=nullcontext()),
            patch.object(module, "Attention", _DenseRing),
            torch.inference_mode(),
            ExitStack() as scopes,
        ):
            installed = adapter(local, config)
            if role == "ltx.video_self":
                # The frequency expansion should see only the local 256 tokens.
                rope_lengths = []
                local.rope.register_forward_hook(lambda _, args, out: rope_lengths.append(out[0].shape[1]))
            else:
                scopes.enter_context(installed.caption_validation_scope({}))
                projection_lengths = []
                projections = ("to_qkv",) if fused else ("to_q", "to_k", "to_v")
                for proj in projections:
                    getattr(local.single_transformer_blocks[0].attn, proj).register_forward_hook(
                        lambda _, args, out: projection_lengths.append(args[0].shape[1])
                    )
            for kwargs in inputs:
                expected = model(**kwargs).sample
                actual = local(**kwargs).sample
                torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
                assert ctx.sparsp_video_layout is None
            if role == "ltx.video_self":
                assert rope_lengths == [256] * len(inputs)
            else:
                # 256 video + 64 caption tokens per joint projection.
                assert projection_lengths == [320] * (len(projections) * len(inputs))
                assert len(installed.validated_captions) == 1
                invalid = dict(inputs[0], encoder_attention_mask=torch.zeros_like(inputs[0]["encoder_attention_mask"]))
                with pytest.raises(ValueError, match="fully valid caption"):
                    local(**invalid)
                assert ctx.sparsp_video_layout is None
                with installed.caption_validation_scope({"callback_on_step_end": lambda: None}):
                    assert installed.validated_captions is None
        if role == "hunyuan_video.video_joint":
            assert installed.validated_captions is None


def _check_models(device):
    torch.manual_seed(20)
    ltx = LTXVideoTransformer3DModel(
        in_channels=8,
        out_channels=8,
        num_layers=1,
        num_attention_heads=2,
        attention_head_dim=16,
        cross_attention_dim=32,
        caption_channels=16,
    )
    args = dict(
        hidden_states=torch.randn(2, 512, 8, device=device),
        encoder_hidden_states=torch.randn(2, 3, 16, device=device),
        encoder_attention_mask=torch.ones(2, 3, device=device),
        timestep=torch.tensor([2, 2], device=device),
        num_frames=2,
        height=16,
        width=16,
    )
    coords = torch.randn(2, 3, 512, device=device)
    _check_adapter(
        ltx,
        ltx_sparsp,
        ltx_sparsp.LTXSparSPAdapter,
        "ltx.video_self",
        [args, dict(args, video_coords=coords, timestep=torch.arange(512, device=device).expand(2, -1))],
        device,
    )

    hy = HunyuanVideoTransformer3DModel(
        in_channels=4,
        out_channels=4,
        num_layers=1,
        num_single_layers=1,
        num_refiner_layers=1,
        num_attention_heads=2,
        attention_head_dim=16,
        rope_axes_dim=(4, 6, 6),
        patch_size=1,
        text_embed_dim=16,
        pooled_projection_dim=12,
    )
    args = dict(
        hidden_states=torch.randn(1, 4, 2, 16, 16, device=device),
        encoder_hidden_states=torch.randn(1, 128, 16, device=device),
        timestep=torch.tensor([2.0], device=device),
        encoder_attention_mask=torch.ones(1, 128, device=device),
        pooled_projections=torch.randn(1, 12, device=device),
        guidance=torch.ones(1, device=device),
    )
    _check_adapter(
        hy, hunyuan_sparsp, hunyuan_sparsp.HunyuanSparSPAdapter, "hunyuan_video.video_joint", [args, args], device
    )


def _check_cpu_rank(rank, rendezvous):
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        _check_models(torch.device("cpu"))
    finally:
        dist.destroy_process_group()


def test_video_adapters_match_unsharded_diffusers(tmp_path):
    mp.spawn(_check_cpu_rank, args=((tmp_path / "gloo").as_uri(),), nprocs=2)
