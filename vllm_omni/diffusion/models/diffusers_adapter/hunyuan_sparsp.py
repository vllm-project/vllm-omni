# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""SparSP for the original HunyuanVideo text-to-video transformer."""

import inspect
from contextlib import contextmanager
from math import prod

import torch
from diffusers.models.embeddings import apply_rotary_emb
from diffusers.models.transformers.transformer_hunyuan_video import HunyuanVideoTransformer3DModel

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.attention.sparsp.placement import resolve_video_layout, supports_interval_periodic_layout
from vllm_omni.diffusion.config import set_current_diffusion_config
from vllm_omni.diffusion.distributed.parallel_state import get_sp_group
from vllm_omni.diffusion.forward_context import get_forward_context


class HunyuanSparSPProcessor:
    def __init__(self, attn):
        dim = attn.to_q.out_features // attn.heads
        self.attention = Attention(
            num_heads=attn.heads,
            head_size=dim,
            softmax_scale=dim**-0.5,
            causal=False,
            role="hunyuan_video.video_joint",
            role_category="self",
        )

    def __call__(self, attn, hidden_states, encoder_hidden_states, attention_mask=None, image_rotary_emb=None):
        # The dual-stream blocks have separate text projections. In single-stream
        # blocks both streams use the video projections and norms.
        heads = (attn.heads, -1)
        sizes = (hidden_states.shape[1], encoder_hidden_states.shape[1])
        if attn.add_q_proj is None:
            # The single-stream block shares projections and norms across both
            # streams. Rotate only the video prefix of the fresh Q/K outputs.
            hidden_states = torch.cat((hidden_states, encoder_hidden_states), dim=1)
        fused = getattr(attn, "fused_projections", False)
        if fused:
            q, k, v = attn.to_qkv(hidden_states).chunk(3, dim=-1)
        else:
            q, k, v = attn.to_q(hidden_states), attn.to_k(hidden_states), attn.to_v(hidden_states)
        q = attn.norm_q(q.unflatten(2, heads))
        k = attn.norm_k(k.unflatten(2, heads))
        v = v.unflatten(2, heads)
        if attn.add_q_proj is None:
            q_video = q[:, : sizes[0]]
            k_video = k[:, : sizes[0]]
            q_video.copy_(apply_rotary_emb(q_video, image_rotary_emb, sequence_dim=1))
            k_video.copy_(apply_rotary_emb(k_video, image_rotary_emb, sequence_dim=1))
        else:
            q = apply_rotary_emb(q, image_rotary_emb, sequence_dim=1)
            k = apply_rotary_emb(k, image_rotary_emb, sequence_dim=1)
            if fused:
                tq, tk, tv = attn.to_added_qkv(encoder_hidden_states).chunk(3, dim=-1)
            else:
                tq = attn.add_q_proj(encoder_hidden_states)
                tk = attn.add_k_proj(encoder_hidden_states)
                tv = attn.add_v_proj(encoder_hidden_states)
            tq = attn.norm_added_q(tq.unflatten(2, heads))
            tk = attn.norm_added_k(tk.unflatten(2, heads))
            tv = tv.unflatten(2, heads)
            q, k, v = torch.cat((q, tq), 1), torch.cat((k, tk), 1), torch.cat((v, tv), 1)
        meta = None
        if get_forward_context().sparsp_video_layout is not None:
            meta = AttentionMetadata(extra={"sparsp_sharded_text_tokens": sizes[1]})
        output = self.attention(q, k, v, meta)
        video, text = output.flatten(2).split(sizes, dim=1)
        if getattr(attn, "to_out", None) is not None:
            video = attn.to_out[1](attn.to_out[0](video))
        if getattr(attn, "to_add_out", None) is not None:
            text = attn.to_add_out(text)
        return video, text


class HunyuanSparSPAdapter:
    def __init__(self, transformer, config):
        if not isinstance(transformer, HunyuanVideoTransformer3DModel) or transformer.config.image_condition_type:
            raise ValueError("HunyuanVideo SparSP requires the original text-to-video transformer")
        spec = config.diffusion_attention_config.per_role["hunyuan_video.video_joint"].sparsp
        self.placement = spec.placement if spec else "none"
        self.group = get_sp_group()
        self.signature = inspect.signature(transformer.forward)
        self.validated_captions = None
        self.layout = None
        self.previous = []
        with set_current_diffusion_config(config):
            for block in (*transformer.transformer_blocks, *transformer.single_transformer_blocks):
                block.attn.set_processor(HunyuanSparSPProcessor(block.attn))
        transformer.register_forward_pre_hook(self.prepare, with_kwargs=True)
        transformer.x_embedder.register_forward_hook(self.split_video)
        transformer.rope.register_forward_hook(self.split_rope)
        transformer.context_embedder.register_forward_hook(self.split_text)
        transformer.proj_out.register_forward_hook(self.gather)
        transformer.register_forward_hook(self.restore, always_call=True)

    @contextmanager
    def caption_validation_scope(self, kwargs):
        # Diffusers reuses its internally encoded positive and negative masks
        # for each step. Callback or caller-owned masks may change in place.
        reusable = all(
            kwargs.get(key) is None
            for key in ("callback_on_step_end", "prompt_attention_mask", "negative_prompt_attention_mask")
        )
        previous = self.validated_captions
        self.validated_captions = {} if reusable else None
        try:
            yield
        finally:
            self.validated_captions = previous

    def prepare(self, module, args, kwargs):
        ctx = get_forward_context()
        self.previous.append(ctx.sparsp_video_layout)
        inputs = self.signature.bind(*args, **kwargs).arguments
        latent = inputs["hidden_states"]
        frames, height, width = latent.shape[2:]
        patch = module.config.patch_size
        grid = (frames // module.config.patch_size_t, height // patch, width // patch)
        video_tokens = prod(grid)
        text_tokens = inputs["encoder_hidden_states"].shape[1]
        degree = self.group.ring_world_size
        if video_tokens % degree or text_tokens % degree or (video_tokens + text_tokens) % (64 * degree):
            raise ValueError("HunyuanVideo requires equal video and text shards and complete joint K64 blocks")
        # Mirror the pipeline's effective caption-length calculation before it sees sharded tokens.
        mask = inputs["encoder_attention_mask"]
        cache = self.validated_captions
        if cache is None or cache.get(mask) != text_tokens:
            if (mask.sum(1, dtype=torch.int) < text_tokens).any().item():
                raise ValueError("Original HunyuanVideo SparSP requires a fully valid caption token budget")
            if cache is not None:
                cache[mask] = text_tokens
        layout = None
        if self.placement == "interval_periodic" or (
            self.placement == "auto" and supports_interval_periodic_layout(grid, degree)
        ):
            self.layout = layout = resolve_video_layout(grid, degree, latent.device, self.placement, self.layout)
        ctx.sparsp_video_layout = layout

    def video_shard(self, value, dim):
        degree, rank = self.group.ring_world_size, self.group.ring_rank
        layout = get_forward_context().sparsp_video_layout
        if layout is None:
            return value.chunk(degree, dim=dim)[rank].contiguous()
        return value.index_select(dim, layout.permutation.chunk(degree)[rank])

    def split_video(self, module, args, output):
        return self.video_shard(output, 1)

    def split_rope(self, module, args, output):
        return tuple(self.video_shard(table, 0) for table in output)

    def split_text(self, module, args, output):
        return output.chunk(self.group.ring_world_size, dim=1)[self.group.ring_rank].contiguous()

    def gather(self, module, args, output):
        video = self.group.all_gather(output, dim=1, group=self.group.ring_group)
        layout = get_forward_context().sparsp_video_layout
        return video if layout is None else video.index_select(1, layout.inverse)

    def restore(self, module, args, output):
        get_forward_context().sparsp_video_layout = self.previous.pop()
