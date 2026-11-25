# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""SparSP token ownership for the original Diffusers LTX-Video transformer."""

import inspect

from diffusers.models.transformers.transformer_ltx import LTXVideoTransformer3DModel, apply_rotary_emb

from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.attention.sparsp.placement import resolve_video_layout
from vllm_omni.diffusion.config import set_current_diffusion_config
from vllm_omni.diffusion.distributed.parallel_state import get_sp_group
from vllm_omni.diffusion.forward_context import get_forward_context


class LTXSparSPProcessor:
    def __init__(self, attn):
        dim = attn.to_q.out_features // attn.heads
        self.attention = Attention(
            num_heads=attn.heads,
            head_size=dim,
            softmax_scale=dim**-0.5,
            causal=False,
            role="ltx.video_self",
            role_category="self",
        )

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, image_rotary_emb=None):
        if encoder_hidden_states is not None or attention_mask is not None:
            raise ValueError("LTX SparSP requires unmasked video self-attention")
        if getattr(attn, "fused_projections", False):
            q, k, v = attn.to_qkv(hidden_states).chunk(3, dim=-1)
        else:
            q, k, v = attn.to_q(hidden_states), attn.to_k(hidden_states), attn.to_v(hidden_states)
        q, k = attn.norm_q(q), attn.norm_k(k)
        if image_rotary_emb is not None:
            q = apply_rotary_emb(q, image_rotary_emb)
            k = apply_rotary_emb(k, image_rotary_emb)
        heads = (attn.heads, -1)
        result = self.attention(q.unflatten(2, heads), k.unflatten(2, heads), v.unflatten(2, heads))
        return attn.to_out[1](attn.to_out[0](result.flatten(2, 3)))


class LTXSparSPAdapter:
    def __init__(self, transformer, config):
        if not isinstance(transformer, LTXVideoTransformer3DModel):
            raise ValueError("ltx.video_self requires a Diffusers LTXVideoTransformer3DModel")
        spec = config.diffusion_attention_config.per_role["ltx.video_self"].sparsp
        self.placement = spec.placement if spec is not None else "none"
        self.cached_layout = None
        self.signature = inspect.signature(transformer.forward)
        self.group = get_sp_group()
        self.previous = []
        with set_current_diffusion_config(config):
            for block in transformer.transformer_blocks:
                block.attn1.set_processor(LTXSparSPProcessor(block.attn1))
        transformer.register_forward_pre_hook(self.split_inputs, with_kwargs=True)
        # Diffusers constructs RoPE from either generated [B,S,3] coordinates
        # or caller-supplied [B,3,S] coordinates. Split before expanding the
        # frequency dimension so each rank evaluates only its own positions.
        prepare_coords = transformer.rope._prepare_video_coords

        def prepare_local_coords(*args, **kwargs):
            return self.shard(prepare_coords(*args, **kwargs))

        transformer.rope._prepare_video_coords = prepare_local_coords
        transformer.proj_out.register_forward_hook(self.gather)
        transformer.register_forward_hook(self.restore, always_call=True)

    def shard(self, tensor, dim=1):
        degree, rank = self.group.ring_world_size, self.group.ring_rank
        layout = get_forward_context().sparsp_video_layout
        if layout is None:
            return tensor.chunk(degree, dim=dim)[rank]
        indices = layout.permutation.chunk(degree)[rank]
        return tensor.index_select(dim, indices)

    def split_inputs(self, module, args, kwargs):
        ctx = get_forward_context()
        self.previous.append(ctx.sparsp_video_layout)
        inputs = self.signature.bind(*args, **kwargs)
        hidden = inputs.arguments["hidden_states"]
        grid = tuple(inputs.arguments.get(name) for name in ("num_frames", "height", "width"))
        if any(size is None for size in grid) or grid[0] * grid[1] * grid[2] != hidden.shape[1]:
            raise ValueError("LTX SparSP requires a complete video grid")
        self.cached_layout = resolve_video_layout(
            grid, self.group.ring_world_size, hidden.device, self.placement, self.cached_layout
        )
        ctx.sparsp_video_layout = self.cached_layout
        inputs.arguments["hidden_states"] = self.shard(hidden)
        timestep = inputs.arguments["timestep"]
        if timestep.ndim == 2 and timestep.shape[1] == hidden.shape[1]:
            inputs.arguments["timestep"] = self.shard(timestep)
        coords = inputs.arguments.get("video_coords")
        if coords is not None:
            inputs.arguments["video_coords"] = self.shard(coords, dim=2)
        return inputs.args, inputs.kwargs

    def gather(self, module, args, output):
        video = self.group.all_gather(output, dim=1, group=self.group.ring_group)
        layout = get_forward_context().sparsp_video_layout
        return video if layout is None else video.index_select(1, layout.inverse)

    def restore(self, module, args, output):
        get_forward_context().sparsp_video_layout = self.previous.pop()
