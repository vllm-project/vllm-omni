# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import inspect

import torch
from diffusers.models.normalization import RMSNorm as DiffusersRMSNorm
from torch import nn

from vllm_omni.diffusion.models.causvid_wan.transformer import stage_layer_range
from vllm_omni.diffusion.models.ltx2.ltx2_transformer import (
    LTX2AudioVideoAttnProcessor,
    LTX2VideoTransformer3DModel,
    apply_interleaved_rotary_emb,
    apply_split_rotary_emb,
)

AUDIO_BLOCK_PARTS = (
    "audio_norm1",
    "audio_attn1",
    "audio_norm2",
    "audio_attn2",
    "audio_to_video_norm",
    "audio_to_video_attn",
    "video_to_audio_norm",
    "video_to_audio_attn",
    "audio_norm3",
    "audio_ff",
    "audio_scale_shift_table",
    "audio_prompt_scale_shift_table",
    "video_a2v_cross_attn_scale_shift_table",
    "audio_a2v_cross_attn_scale_shift_table",
)
AUDIO_MODEL_PARTS = (
    "audio_proj_in",
    "audio_caption_projection",
    "audio_time_embed",
    "audio_prompt_adaln",
    "av_cross_attn_video_scale_shift",
    "av_cross_attn_audio_scale_shift",
    "av_cross_attn_video_a2v_gate",
    "av_cross_attn_audio_v2a_gate",
    "audio_rope",
    "cross_attn_rope",
    "cross_attn_audio_rope",
    "audio_norm_out",
    "audio_proj_out",
    "audio_scale_shift_table",
)


def _drop(module, names):
    for name in names:
        if hasattr(module, name):
            delattr(module, name)


class _CellAttention:
    def __init__(self, owner, layer):
        self.owner, self.layer = owner, layer

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        query_rotary_emb=None,
        key_rotary_emb=None,
    ):
        assert encoder_hidden_states is None and attention_mask is None
        owner = self.owner
        gates = attn.to_gate_logits(hidden_states) if attn.to_gate_logits is not None else None
        query, key, value = LTX2AudioVideoAttnProcessor._project_qkv(attn, hidden_states, hidden_states, True)
        query, key = attn.norm_q(query), attn.norm_k(key)
        batch, tokens, _ = query.shape
        key = key.unflatten(2, (attn.heads, attn.head_dim))
        value = value.unflatten(2, (attn.heads, attn.head_dim))
        if owner.capture_sink:
            owner.sink_pages[self.layer] = (
                key[0].detach().contiguous(),
                value[0].detach().contiguous(),
            )
        context = owner._kv_context
        if context is not None:
            # The deployment's FP32 pool preserves normalized K exactly.
            # V is BF16, so its FP32 storage conversion is also exact.
            context.key_pool.index_copy_(0, context.video_slots, key[0].to(context.key_pool.dtype))
            context.value_pool.index_copy_(0, context.video_slots, value[0].to(context.value_pool.dtype))
            count = context.seq_len // context.block_size
            blocks = context.block_table[0, : count - 1].to(torch.long)
            slots = (
                blocks[:, None] * context.block_size + torch.arange(context.block_size, device=blocks.device)[None, :]
            ).flatten()
            sink_key, sink_value = owner.sink_pages[self.layer]
            key = torch.cat(
                (
                    sink_key.to(key.dtype),
                    context.key_pool.index_select(0, slots).to(key.dtype),
                    key[0],
                ),
                dim=0,
            ).unsqueeze(0)
            value = torch.cat(
                (sink_value, context.value_pool.index_select(0, slots), value[0]),
                dim=0,
            ).unsqueeze(0)
        history_tokens = key.shape[1] - tokens
        own, window = owner.rotary(owner._chunk, owner._latent, history_tokens)
        key = key.flatten(2, 3)
        value = value.to(owner.compute_dtype)
        if attn.rope_type == "split":
            query = apply_split_rotary_emb(query, own, head_dim=attn.head_dim)
            key = apply_split_rotary_emb(key, window, head_dim=attn.head_dim)
        else:
            query, key = (
                apply_interleaved_rotary_emb(query, own),
                apply_interleaved_rotary_emb(key, window),
            )
        query = query.to(value.dtype).unflatten(2, (attn.heads, attn.head_dim))
        key = key.to(value.dtype).unflatten(2, (attn.heads, attn.head_dim))
        output = attn.attn(query, key, value, None).flatten(2, 3).to(query.dtype)
        if gates is not None:
            output = (
                output.view(batch, tokens, attn.heads, attn.head_dim) * (2.0 * torch.sigmoid(gates)).unsqueeze(-1)
            ).reshape(batch, tokens, -1)
        return attn.to_out[1](attn.to_out[0](output))


class StageLTXTransformer(nn.Module):
    def __init__(self, config, *, layer_groups, pp_rank, dtype=torch.bfloat16):
        super().__init__()
        if config.get("use_prompt_embeddings", True) or (
            config["patch_size"],
            config["patch_size_t"],
        ) != (1, 1):
            raise ValueError("Physis LTX requires connector embeddings and unit patches")
        self.layer_groups, self.pp_rank = layer_groups, pp_rank
        self.group = pp_rank % layer_groups
        self.is_layer_group_first, self.is_layer_group_last = (
            self.group == 0,
            self.group == layer_groups - 1,
        )
        self.num_layers = config["num_layers"]
        self.num_heads, self.head_dim = (
            config["num_attention_heads"],
            config["attention_head_dim"],
        )
        self.dim = self.num_heads * self.head_dim
        self.start_layer, self.end_layer = stage_layer_range(self.num_layers, self.group, layer_groups)
        self.compute_dtype = dtype
        device = torch.empty(()).device
        allowed = set(inspect.signature(LTX2VideoTransformer3DModel.__init__).parameters)
        with torch.device("meta"):
            model = LTX2VideoTransformer3DModel(**{key: value for key, value in config.items() if key in allowed})
        model.transformer_blocks = nn.ModuleDict(
            {str(index): model.transformer_blocks[index] for index in range(self.start_layer, self.end_layer)}
        )
        for index, block in model.transformer_blocks.items():
            _drop(block, AUDIO_BLOCK_PARTS)
            # Match the reference's cast-back before modulation arithmetic.
            for name in ("norm1", "norm2", "norm3"):
                original = getattr(block, name)
                with torch.device("meta"):
                    norm = DiffusersRMSNorm(
                        self.dim,
                        eps=config.get("norm_eps", 1e-6),
                        elementwise_affine=config.get("norm_elementwise_affine", False),
                    )
                norm.weight = original.weight
                setattr(block, name, norm)
            block.attn1.set_processor(_CellAttention(self, int(index)))
        _drop(model, AUDIO_MODEL_PARTS)
        if not self.is_layer_group_first:
            _drop(model, ("proj_in", "keyframes_abs_pos_embedding"))
        if not self.is_layer_group_last:
            _drop(model, ("norm_out", "proj_out", "scale_shift_table"))
        self.model = model.to_empty(device=device).to(dtype=dtype).eval().requires_grad_(False)
        self.sink_pages = {}
        self.capture_sink = False
        self._kv_context = None
        self._rotary = {}
        self._embeddings = {}

    @property
    def local_num_layers(self):
        return self.end_layer - self.start_layer

    def load_weights(self, weights):
        expected = set(dict(self.model.named_parameters()))

        def retained():
            for name, value in weights:
                if name.startswith("transformer."):
                    name = name[len("transformer.") :]
                if name in expected:
                    yield name, value
                    continue
                parts = name.split(".")
                if parts[0] in AUDIO_MODEL_PARTS:
                    continue
                if parts[0] == "transformer_blocks":
                    index = int(parts[1])
                    if not self.start_layer <= index < self.end_layer or parts[2] in AUDIO_BLOCK_PARTS:
                        continue
                if not self.is_layer_group_first and parts[0] in (
                    "proj_in",
                    "keyframes_abs_pos_embedding",
                ):
                    continue
                if not self.is_layer_group_last and parts[0] in (
                    "norm_out",
                    "proj_out",
                    "scale_shift_table",
                ):
                    continue
                raise ValueError(f"Unexpected retained LTX checkpoint tensor: {name}")

        loaded = self.model.load_weights(retained())
        missing = expected - loaded
        if missing:
            raise ValueError(f"Missing LTX video checkpoint tensors: {sorted(missing)}")
        return None

    @staticmethod
    def seq_len_for_latent(shape):
        return shape[2] * shape[3] * shape[4]

    def prepare_cell(self, encoder_hidden_states, **kwargs):
        self._text = encoder_hidden_states.float()
        self._text_mask = torch.zeros(
            encoder_hidden_states.shape[0],
            1,
            encoder_hidden_states.shape[1],
            device=encoder_hidden_states.device,
            dtype=self.compute_dtype,
        )
        self._rotary.clear()
        self._embeddings.clear()
        self.fps = float(kwargs.get("fps", 24.0))

    def embed(self, latent):
        batch, channels, frames, height, width = latent.shape
        rows = latent.permute(0, 2, 3, 4, 1).reshape(batch, frames * height * width, channels)
        with torch.autocast(latent.device.type, dtype=self.compute_dtype):
            return self.model.proj_in(rows)

    def embedding(self, timestep, latent):
        batch = latent.shape[0]
        tokens = self.seq_len_for_latent(latent.shape)
        key = float(timestep), tokens
        if key not in self._embeddings:
            with torch.autocast(latent.device.type, dtype=self.compute_dtype):
                per_token = torch.full((batch * tokens,), key[0], device=latent.device, dtype=torch.float32)
                temb, embedded = self.model.time_embed(per_token, batch_size=batch, hidden_dtype=torch.float32)
                temb, embedded = (
                    temb.view(batch, -1, temb.size(-1)),
                    embedded.view(batch, -1, embedded.size(-1)),
                )
                prompt = None
                if self.model.prompt_modulation and self.model.config.use_prompt_adaln_single:
                    prompt, _ = self.model.prompt_adaln(
                        torch.full((batch,), key[0], device=latent.device, dtype=torch.float32),
                        batch_size=batch,
                        hidden_dtype=torch.float32,
                    )
                    prompt = prompt.view(batch, -1, prompt.size(-1))
            self._embeddings[key] = temb, embedded, prompt
        return self._embeddings[key]

    def coords(self, start, stop, latent):
        batch, _, _, height, width = latent.shape
        axes = [
            torch.arange(left, right, device=latent.device, dtype=torch.float32)
            for left, right in ((start, stop), (0, height), (0, width))
        ]
        grid = torch.stack(torch.meshgrid(*axes, indexing="ij"))
        delta = torch.ones((3, 1, 1, 1), device=latent.device, dtype=torch.float32)
        coords = torch.stack((grid, grid + delta), dim=-1).flatten(1, 3).unsqueeze(0).repeat(batch, 1, 1, 1)
        rope = self.model.rope
        coords = coords * torch.tensor(rope.scale_factors, device=latent.device).view(1, 3, 1, 1)
        coords[:, 0] = (coords[:, 0] + rope.causal_offset - rope.scale_factors[0]).clamp(min=0)
        coords[:, 0] = coords[:, 0] / self.fps
        return coords

    def rotary(self, chunk, latent, history_tokens):
        key = chunk, history_tokens
        if key not in self._rotary:
            per_frame = latent.shape[3] * latent.shape[4]
            start = 0 if chunk == 0 else chunk + 2
            stop = start + latent.shape[2]
            first = start - history_tokens // per_frame
            tables = self.model.rope(self.coords(first, stop, latent), device=latent.device)
            axis = 2 if tables[0].ndim == 4 else 1
            own = tuple(table.narrow(axis, history_tokens, latent.shape[2] * per_frame) for table in tables)
            self._rotary = {key: (own, tables)}
        return self._rotary[key]

    def block(self, index, hidden, *, chunk, step, latent, timestep, kv_ctx=None):
        del step
        self._kv_context, self._chunk, self._latent = kv_ctx, chunk, latent
        temb, _, prompt = self.embedding(timestep, latent)
        block = self.model.transformer_blocks[str(index)]
        with torch.autocast(latent.device.type, dtype=self.compute_dtype):
            batch = hidden.shape[0]
            params = block.get_mod_params(block.scale_shift_table, temb, batch)
            shift_sa, scale_sa, gate_sa, shift_ff, scale_ff, gate_ff = params[:6]
            norm = block.norm1(hidden) * (1 + scale_sa) + shift_sa
            hidden = hidden + block.attn1(norm) * gate_sa
            text = self._text
            if block.cross_attn_adaln:
                if prompt is not None:
                    shift_kv, scale_kv = block.get_mod_params(block.prompt_scale_shift_table, prompt, batch)
                else:
                    shift_kv, scale_kv = block.prompt_scale_shift_table[None, None].to(hidden.dtype).unbind(dim=2)
                text = text * (1 + scale_kv) + shift_kv
            norm = block.norm2(hidden)
            if block.video_cross_attn_adaln:
                shift_q, scale_q, gate_q = params[6:9]
                norm = norm * (1 + scale_q) + shift_q
            cross = block.attn2(norm, encoder_hidden_states=text, attention_mask=self._text_mask)
            if block.video_cross_attn_adaln:
                cross = cross * gate_q
            hidden = hidden + cross
            norm = block.norm3(hidden) * (1 + scale_ff) + shift_ff
            output = hidden + block.ff(norm) * gate_ff
        self._kv_context = None
        return output

    def unembed(self, hidden, *, chunk, step, latent, timestep):
        del chunk, step
        _, embedded, _ = self.embedding(timestep, latent)
        with torch.autocast(latent.device.type, dtype=self.compute_dtype):
            values = self.model.scale_shift_table[None, None] + embedded[:, :, None]
            shift, scale = values[:, :, 0], values[:, :, 1]
            out = self.model.proj_out(self.model.norm_out(hidden) * (1 + scale) + shift)
        batch, channels, frames, height, width = latent.shape
        return out.reshape(batch, frames, height, width, channels).permute(0, 4, 1, 2, 3)
