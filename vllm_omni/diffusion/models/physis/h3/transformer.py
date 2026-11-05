# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from collections import OrderedDict
from copy import copy

import torch
import torch.nn.functional as F
from torch import nn

from vllm_omni.diffusion.models.minimax_h3.fasth3 import _resolve_native_target
from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import MiniMaxH3DiTModel
from vllm_omni.diffusion.models.minimax_h3.packed_tokens import (
    minimax_h3_patchify_video_latent,
    minimax_h3_unpatchify_video_tokens,
)

from .positions import phase_aligned_frames, video_positions


def _reference_norm(hidden, norm):
    return F.rms_norm(hidden, (hidden.shape[-1],), norm.weight, norm.variance_epsilon)


def _reference_mlp(hidden, mlp):
    projected, _ = mlp.fc1(hidden)
    gate, value = projected.chunk(2, dim=-1)
    # The native loader stores [gate, value]; keep the reference BF16 SiLU result.
    activated = F.silu(gate)
    hidden, _ = mlp.fc2(activated * value)
    return hidden


class PhysisStageH3Transformer(nn.Module):
    def __init__(self, od_config, transformer_config, *, layer_groups=1, pp_rank=0):
        super().__init__()
        self.layer_groups = layer_groups
        self.group = pp_rank % layer_groups
        self.is_layer_group_first = self.group == 0
        self.is_layer_group_last = self.group == layer_groups - 1
        self.num_layers = int(transformer_config["num_layers"])
        if self.num_layers % layer_groups:
            raise ValueError("H3 Cell layer groups must divide the checkpoint's layer count")
        self.start_layer = self.num_layers * self.group // layer_groups
        self.end_layer = self.num_layers * (self.group + 1) // layer_groups
        local_config = copy(od_config)
        local_config.tf_model_config = {**transformer_config, "num_layers": self.local_num_layers}
        self.backbone = MiniMaxH3DiTModel(local_config, diffusers_weights=True)
        model = self.backbone
        # Keep native global checkpoint names while materializing only owned blocks.
        model.blocks = nn.ModuleDict({str(self.start_layer + i): block for i, block in enumerate(model.blocks)})
        del model.audio_patch_proj
        del model.final_layer.audio_out
        if not self.is_layer_group_first:
            del model.video_patch_proj
            del model.condition_proj
            del model.token_refiner
        if not self.is_layer_group_last:
            del model.final_layer
        attentions = [block.attn for block in model.blocks.values()]
        if self.is_layer_group_first:
            attentions.extend(block.attn for block in model.token_refiner.blocks)
        for attention in attentions:
            for name in ("q_norm", "k_norm"):
                norm = getattr(attention, name)
                reference = nn.RMSNorm(
                    norm.hidden_size, eps=norm.variance_epsilon, device="meta", dtype=norm.weight.dtype
                )
                reference.weight = norm.weight
                setattr(attention, name, reference)
        self.patch_size = model.arch.patch_size
        self.dim = model.arch.hidden_size
        first_block = model.blocks[str(self.start_layer)]
        self.num_heads = first_block.attn.num_heads
        self.head_dim = first_block.attn.head_dim
        self.sink_pages = {}
        self.capture_sink = False
        self._cell_kv_ctx = None
        self._layer = self.start_layer
        self._chunk = 0
        self._latent = None
        self._embeddings = {}
        self._rotary = OrderedDict()
        for block in model.blocks.values():
            block.attn.forward = self._bind_attention(block.attn)

    @property
    def local_num_layers(self):
        return self.end_layer - self.start_layer

    def _bind_attention(self, module):
        def forward(hidden, **kwargs):
            return self._self_attention(module, hidden)

        return forward

    def load_weights(self, weights):
        parameters = dict(self.backbone.named_parameters())
        ignored = {"audio_proj_in", "audio_proj_out"}

        def selected():
            for name, tensor in weights:
                module, _, kind = name.rpartition(".")
                target = _resolve_native_target(module)
                if module in ignored:
                    continue
                if target is None or kind not in {"weight", "bias"}:
                    raise ValueError(f"Unsupported Physis H3 checkpoint tensor: {name}")
                native_name = f"{target[0]}.{kind}"
                if native_name in parameters:
                    yield name, tensor
                elif target[2] is not None and target[2][0] == "blocks.":
                    if self.start_layer <= target[2][1] < self.end_layer:
                        raise ValueError(f"Owned H3 block has no target for {name}")
                elif (
                    not self.is_layer_group_first
                    and target[0].startswith(("video_patch_proj", "condition_proj", "token_refiner"))
                ) or (not self.is_layer_group_last and target[0].startswith("final_layer")):
                    continue
                else:
                    raise ValueError(f"H3 checkpoint has no native parameter for {name}")

        loaded = self.backbone.load_weights(selected())
        missing = set(parameters) - loaded
        if missing:
            raise ValueError(f"Physis H3 checkpoint did not initialize retained tensors: {sorted(missing)}")
        self.backbone.post_load_weights()
        return loaded

    def seq_len_for_latent(self, shape):
        _, _, t, h, w = shape
        pt, ph, pw = self.patch_size
        if t % pt or h % ph or w % pw:
            raise ValueError("H3 latent dimensions must align with its patch size")
        return (t // pt) * (h // ph) * (w // pw)

    def prepare_cell(self, encoder_hidden_states, **kwargs):
        self.text_tokens = encoder_hidden_states.shape[1]
        self._embeddings.clear()
        self._rotary.clear()
        self._text = None
        device = next(self.backbone.parameters()).device
        self.text_positions = torch.zeros(self.text_tokens, 3, device=device, dtype=torch.float64)
        self.text_positions[:, 0] = torch.arange(self.text_tokens, device=device, dtype=torch.float64)
        if self.is_layer_group_first:
            text, _ = self.backbone.condition_proj(encoder_hidden_states[0].to(torch.bfloat16))
            lengths = torch.tensor([0, self.text_tokens], device=device, dtype=torch.int32)
            refiner = self.backbone.token_refiner
            for block in refiner.blocks:
                text = text + block.attn(
                    _reference_norm(text, block.norm1),
                    rope_table=None,
                    cu_seqlens=lengths,
                    max_seqlen=self.text_tokens,
                )
                text = text + _reference_mlp(_reference_norm(text, block.norm2), block.mlp)
            self._text = _reference_norm(text, refiner.final_norm)

    def embed(self, latent):
        model = self.backbone
        rows = minimax_h3_patchify_video_latent(latent, patch_size=self.patch_size)
        video, _ = model.video_patch_proj(rows.float())
        return torch.cat((self._text, video.to(self._text.dtype)), dim=0).unsqueeze(0)

    def _embedding(self, timestep, tokens):
        key = float(timestep)
        if key not in self._embeddings:
            device = next(self.backbone.parameters()).device
            values = torch.tensor([key], device=device, dtype=torch.float32)
            temb = self.backbone.time_embedder(values)
            tags = torch.zeros(tokens, device=device, dtype=torch.long)
            tags[: self.text_tokens] = 1
            inverse = torch.zeros_like(tags)
            self._embeddings[key] = temb, tags, inverse
        return self._embeddings[key]

    def _tables(self, history_tokens):
        shape = self._latent.shape
        per_frame = self.seq_len_for_latent(shape) // shape[2]
        frames = phase_aligned_frames(history_tokens // per_frame, shape[2], self._chunk * shape[2])
        if frames not in self._rotary:
            device = self._latent.device
            window = video_positions(frames, *shape[3:], self.patch_size, self.text_tokens, device)
            own = window[-self.seq_len_for_latent(shape) :]
            positions = torch.cat((self.text_positions, own), dim=0)
            rope = self.backbone.rope
            self._rotary[frames] = (rope(positions[None]), rope(self.text_positions[None]), rope(window[None]))
            if len(self._rotary) > 5:
                self._rotary.popitem(last=False)
        self._rotary.move_to_end(frames)
        return self._rotary[frames]

    def _self_attention(self, attn, hidden):
        qkv, _ = attn.qkv_proj(hidden)
        q_size = attn.num_heads * attn.head_dim
        kv_size = attn.num_kv_heads * attn.head_dim
        query, key, value = qkv.split((q_size, kv_size, kv_size), dim=-1)
        query = attn.q_norm(query.reshape(-1, attn.num_heads, attn.head_dim))
        key = attn.k_norm(key.reshape(-1, attn.num_kv_heads, attn.head_dim))
        value = value.reshape(-1, attn.num_kv_heads, attn.head_dim)
        text = self.text_tokens
        video_key, video_value = key[text:], value[text:]
        if self.capture_sink:
            self.sink_pages[self._layer] = (video_key.detach().contiguous(), video_value.detach().contiguous())
        ctx = self._cell_kv_ctx
        if ctx is not None:
            ctx.key_pool.index_copy_(0, ctx.video_slots, video_key)
            ctx.value_pool.index_copy_(0, ctx.video_slots, video_value)
            count = ctx.seq_len // ctx.block_size
            blocks = ctx.block_table[0, :count].to(torch.long)
            slots = (blocks[:, None] * ctx.block_size + torch.arange(ctx.block_size, device=blocks.device)).flatten()
            sink_key, sink_value = self.sink_pages[self._layer]
            video_key = torch.cat((sink_key, ctx.key_pool.index_select(0, slots)), dim=0)
            video_value = torch.cat((sink_value, ctx.value_pool.index_select(0, slots)), dim=0)
        history_tokens = video_key.shape[0] - (query.shape[0] - text)
        own, prefix, window = self._tables(history_tokens)
        query = attn._apply_rope(query, own)
        text_key = attn._apply_rope(key[:text], prefix)
        video_key = attn._apply_rope(video_key, window)
        key = torch.cat((text_key, video_key), dim=0)
        value = torch.cat((value[:text], video_value), dim=0)
        result = attn.attention(query[None], key[None], value[None], None)[0].flatten(1)
        result, _ = attn.out_proj(result)
        return result

    def block(self, index, hidden, *, chunk, step, latent, timestep, kv_ctx=None):
        self._layer, self._chunk, self._latent = index, chunk, latent
        self._cell_kv_ctx = kv_ctx
        temb, tags, _ = self._embedding(timestep, hidden.shape[1])
        tokens = hidden[0]
        block = self.backbone.blocks[str(index)]
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = block.adaln_proj(temb)
        try:
            # Reference H3 materializes every norm/affine/product/residual in BF16.
            normalized = _reference_norm(tokens, block.norm1)
            modulated = normalized * (1.0 + scale_msa.index_select(0, tags)) + shift_msa.index_select(0, tags)
            attention = self._self_attention(block.attn, modulated)
            tokens = tokens + gate_msa.index_select(0, tags) * attention
            normalized = _reference_norm(tokens, block.norm2)
            modulated = normalized * (1.0 + scale_mlp.index_select(0, tags)) + shift_mlp.index_select(0, tags)
            output = tokens + gate_mlp.index_select(0, tags) * _reference_mlp(modulated, block.mlp)
        finally:
            self._cell_kv_ctx = None
        return output.unsqueeze(0)

    def unembed(self, hidden, *, chunk, step, latent, timestep):
        model = self.backbone
        temb, _, inverse = self._embedding(timestep, hidden.shape[1])
        final = model.final_layer
        shift, scale = final.adaln_proj(temb)
        normalized = _reference_norm(hidden[0], final.norm)
        normalized = normalized * (1.0 + scale.index_select(0, inverse)) + shift.index_select(0, inverse)
        rows, _ = final.video_out(normalized[self.text_tokens :].float())
        pt, ph, pw = self.patch_size
        _, channels, t, h, w = latent.shape
        prediction = minimax_h3_unpatchify_video_tokens(
            rows, latent_shape=(t // pt, h // ph, w // pw, channels), patch_size=self.patch_size
        )
        # Native H3 predicts x0-epsilon; the four-step sampler consumes epsilon-x0.
        return -prediction
