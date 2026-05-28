# Copyright 2022 The OpenAI Authors and The HuggingFace Inc. team. All rights reserved.
#               2025 Zhipu AI Inc (authors: CogAudio Group Members)
# SPDX-License-Identifier: Apache-2.0
"""WhisperVQEncoder: HF WhisperEncoder + VQ codebook + inter-layer pooling.

Built on standard ``WhisperConfig``.  VQ-specific parameters are patched onto
the config object after ``from_pretrained`` so that callers never need a
separate config class.  GLM-TTS (and future VQ-based TTS models) only need::

    from transformers import WhisperConfig
    from vllm_omni.model_executor.models.common.whisper_vq import WhisperVQEncoder

    cfg = WhisperConfig.from_pretrained(checkpoint_dir)
    cfg.pooling_kernel_size  = None       # passed through from checkpoint
    cfg.pooling_type         = "max"      #       or set by caller
    cfg.pooling_position     = 0
    cfg.quantize_vocab_size  = 32768
    cfg.quantize_position    = 16
    cfg.quantize_encoder_only = True
    model = WhisperVQEncoder(cfg)
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import WhisperConfig
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.whisper.modeling_whisper import WhisperEncoder

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


@dataclass
class QuantizedBaseModelOutput(BaseModelOutput):
    quantized_token_ids: torch.LongTensor | None = None


def vector_quantize(inputs: Tensor, codebook: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Nearest-neighbour codebook lookup."""
    embedding_size = codebook.size(1)
    inputs_flatten = inputs.reshape(-1, embedding_size)
    codebook_sqr = torch.sum(codebook**2, dim=1)
    inputs_sqr = torch.sum(inputs_flatten**2, dim=1, keepdim=True)
    distances = torch.addmm(
        codebook_sqr + inputs_sqr,
        inputs_flatten,
        codebook.t(),
        alpha=-2.0,
        beta=1.0,
    )
    _, indices_flatten = torch.min(distances, dim=1)
    codes_flatten = torch.index_select(codebook, dim=0, index=indices_flatten)
    codes = codes_flatten.view_as(inputs)
    return codes, indices_flatten, distances


_LEGACY_LAYER_REMAP = (
    (re.compile(r"^layers\.(\d+)\.attn\.query\."), r"layers.\1.self_attn.q_proj."),
    (re.compile(r"^layers\.(\d+)\.attn\.key\."), r"layers.\1.self_attn.k_proj."),
    (re.compile(r"^layers\.(\d+)\.attn\.value\."), r"layers.\1.self_attn.v_proj."),
    (re.compile(r"^layers\.(\d+)\.attn\.out\."), r"layers.\1.self_attn.out_proj."),
    (re.compile(r"^layers\.(\d+)\.attn_ln\."), r"layers.\1.self_attn_layer_norm."),
    (re.compile(r"^layers\.(\d+)\.mlp_ln\."), r"layers.\1.final_layer_norm."),
    (re.compile(r"^layers\.(\d+)\.mlp\.0\."), r"layers.\1.fc1."),
    (re.compile(r"^layers\.(\d+)\.mlp\.2\."), r"layers.\1.fc2."),
)


def remap_legacy_whisper_vq_state_dict(state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
    """Map CogAudio/GLM-TTS fork parameter names onto HF WhisperEncoder names."""
    remapped: dict[str, Tensor] = {}
    for name, tensor in state_dict.items():
        new_name = name
        if new_name == "embed_positions":
            new_name = "embed_positions.weight"
        for pattern, replacement in _LEGACY_LAYER_REMAP:
            new_name = pattern.sub(replacement, new_name)
        remapped[new_name] = tensor
    return remapped


# ---------------------------------------------------------------------------
# main encoder class
# ---------------------------------------------------------------------------


class WhisperVQEncoder(WhisperEncoder):
    """HF Whisper encoder with optional VQ codebook and pooling.

    Uses a standard ``WhisperConfig`` with the following VQ-specific attrs
    patched on by the caller (or loaded from the checkpoint's config.json):

    * ``pooling_kernel_size``  -- int | None
    * ``pooling_type``         -- "max" | "avg"
    * ``pooling_position``     -- int (0-based layer index)
    * ``quantize_vocab_size``  -- int | None
    * ``quantize_position``    -- int (0-based layer index)
    * ``quantize_encoder_only`` -- bool
    """

    def __init__(self, config: WhisperConfig):
        super().__init__(config)
        embed_dim = config.d_model
        max_source_positions = config.max_source_positions

        # Truncate layers when quantize_encoder_only is set.
        qpos = int(getattr(config, "quantize_position", 0) or 0)
        if getattr(config, "quantize_encoder_only", False):
            self.layers = nn.ModuleList(list(self.layers[:qpos]))
            self.layer_norm = None
        else:
            qpos = max(qpos, int(config.encoder_layers))  # never exceed total

        # Optional inter-layer pooling.
        self.pooling_layer: nn.Module | None = None
        ksize = getattr(config, "pooling_kernel_size", None)
        if ksize is not None:
            ptype = str(getattr(config, "pooling_type", "max") or "max")
            self.pooling_layer = {
                "max": nn.MaxPool1d(kernel_size=int(ksize)),
                "avg": nn.AvgPool1d(kernel_size=int(ksize)),
            }[ptype]

        # Optional vector-quantization codebook.
        self.codebook: nn.Embedding | None = None
        self.embed_positions2: nn.Embedding | None = None
        vq_sz = getattr(config, "quantize_vocab_size", None)
        if vq_sz is not None:
            self.codebook = nn.Embedding(int(vq_sz), embed_dim)
            pos2_len = max_source_positions
            if ksize is not None:
                pos2_len = math.ceil(max_source_positions / int(ksize))
            self.embed_positions2 = nn.Embedding(pos2_len, embed_dim)

        # cached for forward()
        self._pooling_position = int(getattr(config, "pooling_position", 0) or 0)
        self._quantize_position = qpos if vq_sz is not None else -1

    # ------------------------------------------------------------------
    # forward / helpers
    # ------------------------------------------------------------------

    @property
    def device(self) -> torch.device:  # type: ignore[override]
        return self.conv1.weight.device

    @property
    def dtype(self) -> torch.dtype:  # type: ignore[override]
        return self.conv1.weight.dtype

    def load_state_dict(
        self,
        state_dict: dict[str, Tensor],
        strict: bool = True,
        assign: bool = False,
    ):  # type: ignore[override]
        remapped = remap_legacy_whisper_vq_state_dict(state_dict)
        try:
            return super().load_state_dict(remapped, strict=strict, assign=assign)
        except TypeError:
            return super().load_state_dict(remapped, strict=strict)

    def forward(
        self,
        input_features: Tensor,
        attention_mask: Tensor | None = None,
        **_: Any,
    ) -> QuantizedBaseModelOutput:
        batch_size, _, _ = input_features.shape
        hidden_states = F.gelu(self.conv1(input_features))
        hidden_states = F.gelu(self.conv2(hidden_states))
        hidden_states = hidden_states.permute(0, 2, 1)
        seq_len = int(hidden_states.shape[1])

        pos_embed = self._position_embeddings(seq_len, hidden_states.device, hidden_states.dtype)
        hidden_states = hidden_states + pos_embed

        valid_mask = self._downsample_attention_mask(attention_mask, batch_size, seq_len, hidden_states.device)
        hidden_states = hidden_states.masked_fill(~valid_mask.unsqueeze(-1), 0)

        quantized_token_ids = None
        for idx, layer in enumerate(self.layers):
            layer_mask = self._make_layer_attention_mask(valid_mask, hidden_states.dtype)
            layer_outputs = layer(
                hidden_states,
                attention_mask=layer_mask,
                layer_head_mask=None,
                output_attentions=False,
            )
            hidden_states = layer_outputs[0] if isinstance(layer_outputs, tuple) else layer_outputs
            hidden_states = hidden_states.masked_fill(~valid_mask.unsqueeze(-1), 0)

            if idx + 1 == self._pooling_position and self.pooling_layer is not None:
                hidden_states, valid_mask = self._apply_pooling(hidden_states, valid_mask)

            if idx + 1 == self._quantize_position and self.codebook is not None:
                hidden_states, quantized_token_ids = self._apply_vq(hidden_states, valid_mask)

        if self.layer_norm is not None:
            hidden_states = self.layer_norm(hidden_states)
            hidden_states = hidden_states.masked_fill(~valid_mask.unsqueeze(-1), 0)

        return QuantizedBaseModelOutput(
            last_hidden_state=hidden_states,
            quantized_token_ids=quantized_token_ids,
        )

    def _position_embeddings(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        ep = self.embed_positions
        if isinstance(ep, nn.Embedding):
            positions = torch.arange(seq_len, device=device)
            return ep(positions).to(dtype=dtype).unsqueeze(0)
        return ep[:seq_len].to(device=device, dtype=dtype).unsqueeze(0)

    def _downsample_attention_mask(
        self,
        attention_mask: Tensor | None,
        batch_size: int,
        seq_len: int,
        device: torch.device,
    ) -> Tensor:
        if attention_mask is None:
            return torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
        conv_stride = int(self.conv1.stride[0]) * int(self.conv2.stride[0])
        downsampled = attention_mask[:, ::conv_stride].to(device=device, dtype=torch.bool)
        if downsampled.shape[1] < seq_len:
            pad = seq_len - int(downsampled.shape[1])
            downsampled = F.pad(downsampled, (0, pad), value=False)
        return downsampled[:, :seq_len]

    @staticmethod
    def _make_layer_attention_mask(valid_mask: Tensor, dtype: torch.dtype) -> Tensor:
        min_value = torch.finfo(dtype).min
        additive = torch.zeros(valid_mask.shape, dtype=dtype, device=valid_mask.device)
        additive = additive.masked_fill(~valid_mask, min_value)
        return additive[:, None, None, :]

    def _apply_pooling(self, hidden_states: Tensor, valid_mask: Tensor) -> tuple[Tensor, Tensor]:
        k = int(self.config.pooling_kernel_size)  # type: ignore[attr-defined]
        pooled_segments: list[Tensor] = []
        pooled_masks: list[Tensor] = []
        for batch_idx in range(int(hidden_states.shape[0])):
            length = int(valid_mask[batch_idx].sum().item())
            segment = hidden_states[batch_idx, :length]
            if segment.numel() == 0:
                pooled = hidden_states.new_zeros((0, hidden_states.shape[-1]))
            else:
                segment_t = segment.unsqueeze(0).transpose(1, 2)
                if int(segment_t.shape[-1]) % k != 0:
                    segment_t = F.pad(segment_t, (0, k - int(segment_t.shape[-1]) % k))
                pooled = self.pooling_layer(segment_t).transpose(1, 2).squeeze(0)  # type: ignore[misc]
            pooled_segments.append(pooled)
            pooled_masks.append(torch.ones(pooled.shape[0], dtype=torch.bool, device=hidden_states.device))

        max_len = max((int(seg.shape[0]) for seg in pooled_segments), default=0)
        output = hidden_states.new_zeros((hidden_states.shape[0], max_len, hidden_states.shape[-1]))
        output_mask = torch.zeros(hidden_states.shape[0], max_len, dtype=torch.bool, device=hidden_states.device)
        for i, segment in enumerate(pooled_segments):
            if segment.shape[0] > 0:
                output[i, : segment.shape[0]] = segment
                output_mask[i, : segment.shape[0]] = pooled_masks[i]
        return output, output_mask

    def _apply_vq(self, hidden_states: Tensor, valid_mask: Tensor) -> tuple[Tensor, Tensor | None]:
        output = hidden_states.clone()
        token_ids = torch.full(
            (hidden_states.shape[0], hidden_states.shape[1]),
            -1,
            dtype=torch.long,
            device=hidden_states.device,
        )
        for batch_idx in range(int(hidden_states.shape[0])):
            length = int(valid_mask[batch_idx].sum().item())
            if length <= 0:
                continue
            segment = hidden_states[batch_idx : batch_idx + 1, :length]
            quantized, indices, _ = vector_quantize(segment, self.codebook.weight)  # type: ignore[union-attr]
            if self.embed_positions2 is not None:
                quantized = quantized + self.embed_positions2.weight[:length].to(dtype=quantized.dtype).unsqueeze(0)
            output[batch_idx, :length] = quantized.squeeze(0)
            token_ids[batch_idx, :length] = indices.reshape(-1)
        return output.masked_fill(~valid_mask.unsqueeze(-1), 0), token_ids
