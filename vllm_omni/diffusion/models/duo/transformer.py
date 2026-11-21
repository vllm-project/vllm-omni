# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Streaming Realization forward using official TP modules and paged causal attention."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import MiniMaxH3DiTModel

from .contracts import RealizationSequence, dense_mask
from .kernels import gated_residual, modulate


def rms(module, value):
    # Preserve the training implementation's RMSNorm primitive and cast order.
    return F.rms_norm(value, (value.shape[-1],), module.weight, module.variance_epsilon)


def rotary(value, frequencies, coefficients=None):
    width = frequencies.shape[-1]
    rotated, tail = value[..., :width], value[..., width:]
    a, b = rotated.chunk(2, dim=-1)
    if coefficients is None:
        cos = frequencies.cos().to(value.dtype)[:, None]
        sin = frequencies.sin().to(value.dtype)[:, None]
    else:
        cos, sin = coefficients
    if value.device.type == "cuda" and value.dtype in (torch.bfloat16, torch.float16):
        from .kernels import rotary_cuda

        return rotary_cuda(value, (cos, sin))
    return torch.cat((rotated * cos + torch.cat((-b, a), -1) * sin, tail), -1).contiguous()


def refiner_mask(sequence):
    mask = dense_mask(sequence.refiner_groups, sequence.refiner_groups, sequence.device, refiner=True)
    if sequence.text_valid_rows is not None:
        shared = sequence.refiner_text.shape[1] - sequence.text_indices.numel()
        keys = torch.arange(mask.shape[1], device=sequence.device)
        mask = mask & (keys < shared + sequence.text_valid_rows[0])[None, :]
    return mask


def split_qkv(module, packed):
    widths = [
        module.num_heads * module.head_dim,
        module.num_kv_heads * module.head_dim,
        module.num_kv_heads * module.head_dim,
    ]
    q, k, v = packed.split(widths, dim=-1)
    q = rms(module.q_norm, q.view(-1, module.num_heads, module.head_dim))
    k = rms(module.k_norm, k.view(-1, module.num_kv_heads, module.head_dim))
    v = v.view(-1, module.num_kv_heads, module.head_dim)
    return q, k, v


def attention(module, hidden, mask, frequencies=None, paged=None, rotary_coefficients=None):
    packed, _ = module.qkv_proj(hidden)
    q, k, v = split_qkv(module, packed)
    if frequencies is not None:
        q, k = rotary(q, frequencies, rotary_coefficients), rotary(k, frequencies, rotary_coefficients)
    if paged is not None:
        plan, pool, layer, transaction = paged
        out = plan.forward(q, k, v, pool, layer, module.softmax_scale)
        if transaction is not None:
            pool.write_layer(transaction, layer, k, v)
        out, _ = module.out_proj(out.flatten(1).to(hidden.dtype))
        return out, None
    out = F.scaled_dot_product_attention(
        q.transpose(0, 1)[None],
        k.transpose(0, 1)[None],
        v.transpose(0, 1)[None],
        attn_mask=mask,
        dropout_p=0.0,
        is_causal=False,
        scale=module.softmax_scale,
    )
    out = out[0].transpose(0, 1).flatten(1).to(hidden.dtype)
    out, _ = module.out_proj(out)
    return out, None


def embed_sequence(model, s, *, validate=False, mask_fn=refiner_mask):
    text, _ = model.condition_proj(s.refiner_text[0].to(model.condition_proj.weight.dtype))
    if text.shape[0]:
        rm = mask_fn(s)
        for block in model.token_refiner.blocks:
            out, _ = attention(block.attn, rms(block.norm1, text), rm)
            text = text + out
            text = text + block.mlp(rms(block.norm2, text))
        text = rms(model.token_refiner.final_norm, text)
    hidden = text.new_zeros((s.length, model.hidden_size))
    video, _ = model.video_patch_proj(s.video[0].to(model.video_patch_proj.weight.dtype))
    audio, _ = model.audio_patch_proj(s.audio[0].to(model.audio_patch_proj.weight.dtype))
    hidden = hidden.index_copy(0, s.text_indices, text[s.text_selection])
    hidden = hidden.index_copy(0, s.video_indices, video.to(hidden.dtype))
    hidden = hidden.index_copy(0, s.audio_indices, audio.to(hidden.dtype))
    if model.role_embedding_enabled:
        if validate and (s.roles > 1).any():
            raise ValueError("Invalid Realization role embedding ID")
        roles = model.role_embeddings(s.roles.clamp_min(0)).to(hidden.dtype) * (s.roles >= 0)[:, None]
        hidden = hidden + roles
    return hidden


class DuoTransformer(MiniMaxH3DiTModel):
    """Official Realization architecture with frame-v5 streaming cache semantics."""

    # The parent offline SP hooks address a different forward signature.
    _sp_plan = {}

    def __init__(self, od_config, quant_config=None):
        config = od_config.tf_model_config
        config = config.to_dict() if hasattr(config, "to_dict") else dict(config)
        self.legacy_role_embedding = None
        if quant_config is not None:
            from vllm.model_executor.layers.quantization.modelopt import ModelOptFp8Config

            if (
                not isinstance(quant_config, ModelOptFp8Config)
                or quant_config.quant_method != "FP8_PER_CHANNEL_PER_TOKEN"
                or not quant_config.is_checkpoint_fp8_serialized
            ):
                raise ValueError("Duo supports serialized ModelOpt FP8 per-channel/per-token checkpoints")
            class_name = config.get("_class_name")
            if class_name == "H3NativeModelOptCheckpoint":
                layout = config.get("realization_checkpoint_layout", {})
                if (
                    layout.get("version") != 1
                    or layout.get("merged_lora") is not True
                    or layout.get("qkv") != "original_per_head_interleaved"
                    or layout.get("fc1") != "gate_then_value"
                    or type(layout.get("role_embedding_enabled")) is not bool
                ):
                    raise ValueError(
                        "Legacy Realization FP8 checkpoint requires native layout and role embedding metadata"
                    )
                self.legacy_role_embedding = layout["role_embedding_enabled"]
            elif class_name != "MiniMaxH3DiTModel":
                raise ValueError("Duo FP8 checkpoint requires MiniMaxH3DiTModel architecture")
        pc = od_config.parallel_config
        if pc.tensor_parallel_size not in (1, 2, 4, 8) or pc.sequence_parallel_size not in (None, 1):
            raise ValueError("Duo requires TP1/2/4/8 and SP1; AR context is not CP sharded")
        super().__init__(od_config, quant_config=quant_config, diffusers_weights=False)
        self.quant_config = quant_config
        if (
            self.arch.num_attention_heads % pc.tensor_parallel_size
            or self.arch.ffn_hidden_size % pc.tensor_parallel_size
        ):
            raise ValueError("Realization heads/FFN must divide evenly across TP ranks")
        self.role_embeddings = torch.nn.Embedding(2, self.hidden_size, dtype=torch.float32)
        torch.nn.init.zeros_(self.role_embeddings.weight)
        self.role_embedding_enabled = False
        self._realization_base_loaded = False
        self._realization_adapter_loaded = False

    def load_weights(self, weights):
        from .weights import load_original_stream

        return load_original_stream(self, weights)

    def time_conditioning(self, times):
        """Original M-row projections, reusable only for these exact times."""
        temb = self.time_embedder(times)
        return tuple(block.adaln_proj(temb) for block in self.blocks) + (self.final_layer.adaln_proj(temb),)

    def forward(
        self,
        sequence: RealizationSequence,
        times: torch.Tensor,
        *,
        cache=None,
        commit=False,
        paged_runtime,
        validate=True,
        conditioning=None,
    ):
        if not self._realization_base_loaded:
            raise RuntimeError("Realization forward requires complete strict base load")
        if validate:
            sequence.validate()
        self._realization_has_forwarded = True
        s = sequence
        if validate and (times.ndim != 1 or times.device != s.device or not torch.isfinite(times).all()):
            raise ValueError("Realization times must be finite vector on sequence device")
        if validate and (s.time_ids >= times.numel()).any():
            raise ValueError("Realization time table index out of range")
        if commit and validate:
            if torch.is_grad_enabled():
                raise ValueError("Commit Realization clean KV under no_grad")
            if any(g.kind == "noisy" for g in s.groups) or not torch.all(times == 1) or torch.any(s.time_ids != 0):
                raise ValueError("Realization commit requires clean groups, t=1 and time_ids=0")
        hidden = embed_sequence(self, s, validate=validate)
        temb = self.time_embedder(times) if conditioning is None else None
        frequencies = self.rope(s.positions[None])
        # Q/K use the same BF16 dtype as hidden in the audited unquantized
        # architecture. Preserve sin/cos then cast order, once per forward.
        rotary_coefficients = (
            frequencies.cos().to(hidden.dtype)[:, None],
            frequencies.sin().to(hidden.dtype)[:, None],
        )
        indices = s.time_ids * 3 + s.tags
        for i, block in enumerate(self.blocks):
            shift, scale, gate, ff_shift, ff_scale, ff_gate = (
                block.adaln_proj(temb) if conditioning is None else conditioning[i]
            )
            norm = modulate(rms(block.norm1, hidden), scale, shift, indices)
            paged = (*paged_runtime[:2], i, paged_runtime[2])
            out, _ = attention(block.attn, norm, None, frequencies, paged, rotary_coefficients)
            hidden = gated_residual(hidden, gate, out, indices)
            norm = modulate(rms(block.norm2, hidden), ff_scale, ff_shift, indices)
            hidden = gated_residual(hidden, ff_gate, block.mlp(norm), indices)
        shift, scale = self.final_layer.adaln_proj(temb) if conditioning is None else conditioning[-1]
        hidden = modulate(rms(self.final_layer.norm, hidden), scale, shift, s.time_ids)
        video, _ = self.final_layer.video_out(hidden[s.video_indices].to(self.final_layer.video_out.weight.dtype))
        audio, _ = self.final_layer.audio_out(hidden[s.audio_indices].to(self.final_layer.audio_out.weight.dtype))
        result = paged_runtime[2].candidate if commit else cache
        return (video[None], audio[None]), result
