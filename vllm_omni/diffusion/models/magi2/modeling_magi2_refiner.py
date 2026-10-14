# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Copyright (c) 2026 SandAI. All Rights Reserved.

"""Native MAGI-2 1080p refiner diffusion transformer.

The architecture and its dtype chain are adapted from SandAI's Apache-2.0
MAGI-2 refiner implementation.  This vLLM-Omni version removes the external
``inference`` package, MagiCompiler and the SandAI process manager while
keeping the released checkpoint's module names and tensor layouts.

The refiner is a dense, timestep-free DiT over one block-major token sequence
(video, then text).  Adapters and the output head run in fp32; the layers keep
bf16 GEMMs, fp32 norms, RoPE and gating, and a residual stream that is bf16
until the first MLP add and fp32 afterwards.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from safetensors import safe_open

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.layer import Attention

from .attention import apply_rotary_emb
from .configuration_magi2 import MAGI2_REFINER_CONFIG, Magi2RefinerConfig
from .layers import ElementWiseFourierEmbed, Magi2GroupedLinear, ModalityDispatcher, MultiModalityRMSNorm, swiglu7
from .preview_data_proxy import Modality
from .window_attention import Magi2WindowAttentionKernel, Magi2WindowAttentionPlan

CHECKPOINT_INDEX_FILE = "model.safetensors.index.json"
# Modality experts of the multimodal layers, in checkpoint order.
_NUM_MODALITIES = 3  # video, audio, text
_SAFETENSORS_DTYPES = {"BF16": torch.bfloat16, "F32": torch.float32}


class Magi2RefinerAttention(nn.Module):
    """Gated window self-attention with per-head q/k RMSNorm and partial RoPE."""

    def __init__(self, config: Magi2RefinerConfig, *, num_modality: int) -> None:
        super().__init__()
        self.config = config
        self.head_dim = config.head_dim
        self.num_heads_q = config.num_heads_q
        self.num_heads_kv = config.num_heads_kv
        self.q_size = self.num_heads_q * self.head_dim
        self.kv_size = self.num_heads_kv * self.head_dim
        self.pre_norm = MultiModalityRMSNorm(
            config.hidden_size,
            eps=config.norm_eps,
            num_modality=num_modality,
            out_dtype=config.params_dtype,
        )
        self.linear_qkv = Magi2GroupedLinear(
            config.hidden_size,
            self.q_size + 2 * self.kv_size,
            num_experts=num_modality,
            dtype=config.params_dtype,
        )
        self.linear_g = Magi2GroupedLinear(
            config.hidden_size,
            self.num_heads_q,
            num_experts=num_modality,
            dtype=config.params_dtype,
        )
        self.q_norm = MultiModalityRMSNorm(
            self.head_dim,
            eps=config.norm_eps,
            num_modality=num_modality,
            out_dtype=torch.float32,
        )
        self.k_norm = MultiModalityRMSNorm(
            self.head_dim,
            eps=config.norm_eps,
            num_modality=num_modality,
            out_dtype=torch.float32,
        )
        self.linear_proj = Magi2GroupedLinear(
            self.q_size,
            config.hidden_size,
            num_experts=num_modality,
            dtype=config.params_dtype,
        )
        self.window_attention = Attention(
            num_heads=self.num_heads_q,
            num_kv_heads=self.num_heads_kv,
            head_size=self.head_dim,
            causal=False,
            softmax_scale=self.head_dim**-0.5,
            qkv_layout="THD",
            skip_sequence_parallel=True,
            disable_kv_quant=True,
            custom_attention=Magi2WindowAttentionKernel(self.head_dim**-0.5),
        )

    def project(
        self,
        hidden_states: torch.Tensor,
        rope: torch.Tensor,
        modality_dispatcher: ModalityDispatcher,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the sequence-ordered bf16 ``q``, ``k``, ``v`` and the modality-ordered gate logits."""

        normalized = self.pre_norm(hidden_states, modality_dispatcher)
        qkv = self.linear_qkv(normalized, modality_dispatcher)
        gate = self.linear_g(normalized, modality_dispatcher)
        q, k, v = torch.split(qkv, (self.q_size, self.kv_size, self.kv_size), dim=-1)
        # The norms see the bf16 GEMM outputs; upcasting them first changes nothing.
        q = self.q_norm(q.view(-1, self.num_heads_q, self.head_dim), modality_dispatcher)
        k = self.k_norm(k.view(-1, self.num_heads_kv, self.head_dim), modality_dispatcher)
        q = modality_dispatcher.inverse_permute(q)
        k = modality_dispatcher.inverse_permute(k)
        v = modality_dispatcher.inverse_permute(v.reshape(-1, self.num_heads_kv, self.head_dim))
        sin, cos = rope.tensor_split(2, dim=-1)
        q = apply_rotary_emb(q, cos, sin).to(self.config.params_dtype)
        k = apply_rotary_emb(k, cos, sin).to(self.config.params_dtype)
        return q, k, v.to(self.config.params_dtype), gate

    def output(
        self,
        attention: torch.Tensor,
        gate: torch.Tensor,
        modality_dispatcher: ModalityDispatcher,
    ) -> torch.Tensor:
        # The bf16 attention output is gated in fp32 and rounded once for the projection.
        output = modality_dispatcher.permute(attention) * torch.sigmoid(gate.float()).unsqueeze(-1)
        output = output.reshape(-1, self.q_size).to(self.config.params_dtype)
        return self.linear_proj(output, modality_dispatcher)

    def forward(
        self,
        hidden_states: torch.Tensor,
        rope: torch.Tensor,
        modality_dispatcher: ModalityDispatcher,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        """Attend over modality-ordered ``hidden_states``; ``rope`` is in sequence order."""

        q, k, v, gate = self.project(hidden_states, rope, modality_dispatcher)
        attention = self.window_attention(q, k, v, attn_metadata)
        return self.output(attention, gate, modality_dispatcher)


class Magi2RefinerMLP(nn.Module):
    """Interleaved SwiGLU7 MLP; returns fp32."""

    def __init__(self, config: Magi2RefinerConfig, *, num_modality: int) -> None:
        super().__init__()
        self.config = config
        self.pre_norm = MultiModalityRMSNorm(
            config.hidden_size,
            eps=config.norm_eps,
            num_modality=num_modality,
            out_dtype=config.params_dtype,
        )
        self.up_gate_proj = Magi2GroupedLinear(
            config.hidden_size,
            2 * config.intermediate_size,
            num_experts=num_modality,
            dtype=config.params_dtype,
        )
        self.down_proj = Magi2GroupedLinear(
            config.intermediate_size,
            config.hidden_size,
            num_experts=num_modality,
            dtype=config.params_dtype,
        )

    def forward(self, hidden_states: torch.Tensor, modality_dispatcher: ModalityDispatcher) -> torch.Tensor:
        hidden_states = self.pre_norm(hidden_states, modality_dispatcher)
        hidden_states = self.up_gate_proj(hidden_states, modality_dispatcher)
        # SwiGLU7 computes in fp32 from the bf16 GEMM output and rounds once to bf16.
        hidden_states = swiglu7(
            hidden_states,
            self.config.swiglu7_alpha,
            self.config.swiglu7_limit,
            out_dtype=self.config.params_dtype,
        )
        return self.down_proj(hidden_states, modality_dispatcher).to(torch.float32)


class Magi2RefinerLayer(nn.Module):
    """Pre-norm residual layer: ``h + attention(h)``, then ``h + mlp(h)``."""

    # The layer is split into compile regions around the eager window attention.
    region_methods = ("_attention_input", "_attention_output")

    def __init__(self, config: Magi2RefinerConfig, layer_index: int) -> None:
        super().__init__()
        num_modality = config.num_modalities(layer_index)
        self.attention = Magi2RefinerAttention(config, num_modality=num_modality)
        self.mlp = Magi2RefinerMLP(config, num_modality=num_modality)

    def _attention_input(
        self,
        hidden_states: torch.Tensor,
        rope: torch.Tensor,
        modality_dispatcher: ModalityDispatcher,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.attention.project(hidden_states, rope, modality_dispatcher)

    def _attention_output(
        self,
        hidden_states: torch.Tensor,
        attention: torch.Tensor,
        gate: torch.Tensor,
        modality_dispatcher: ModalityDispatcher,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attention.output(attention, gate, modality_dispatcher)
        return hidden_states + self.mlp(hidden_states, modality_dispatcher)

    def forward(
        self,
        hidden_states: torch.Tensor,
        rope: torch.Tensor,
        modality_dispatcher: ModalityDispatcher,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        q, k, v, gate = self._attention_input(hidden_states, rope, modality_dispatcher)
        attention = self.attention.window_attention(q, k, v, attn_metadata)
        return self._attention_output(hidden_states, attention, gate, modality_dispatcher)


class Magi2RefinerBlock(nn.Module):
    def __init__(self, config: Magi2RefinerConfig) -> None:
        super().__init__()
        self.layers = nn.ModuleList(Magi2RefinerLayer(config, index) for index in range(config.num_layers))

    def forward(
        self,
        hidden_states: torch.Tensor,
        rope: torch.Tensor,
        modality_dispatcher: ModalityDispatcher,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states, rope, modality_dispatcher, attn_metadata)
        return hidden_states


class Magi2RefinerPreAdapter(nn.Module):
    """fp32 per-modality input embedders and the element-wise Fourier RoPE."""

    def __init__(self, config: Magi2RefinerConfig) -> None:
        super().__init__()
        self.config = config
        self.video_embedder = nn.Linear(config.video_in_channels, config.hidden_size, dtype=torch.float32)
        self.text_embedder = nn.Linear(config.text_in_channels, config.hidden_size, dtype=torch.float32)
        # Loaded with the checkpoint; the refiner never receives audio tokens.
        self.audio_embedder = nn.Linear(config.audio_in_channels, config.hidden_size, dtype=torch.float32)
        self.rope = ElementWiseFourierEmbed(config.head_dim, temperature=config.rope_temperature)

    def forward(
        self,
        video: torch.Tensor,
        text: torch.Tensor,
        video_indices: torch.Tensor,
        text_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Embed video and text tokens into their sequence positions in fp32."""

        num_tokens = video_indices.numel() + text_indices.numel()
        output = torch.empty(num_tokens, self.config.hidden_size, dtype=torch.float32, device=video.device)
        output.index_copy_(0, text_indices, self.text_embedder(text.float()))
        output.index_copy_(0, video_indices, self.video_embedder(video.float()))
        return output


class Magi2RefinerPostAdapter(nn.Module):
    """fp32 final norm and projection back to latent channels."""

    def __init__(self, config: Magi2RefinerConfig) -> None:
        super().__init__()
        self.final_norm_video = MultiModalityRMSNorm(config.hidden_size, eps=config.norm_eps)
        self.final_norm_audio = MultiModalityRMSNorm(config.hidden_size, eps=config.norm_eps)
        self.final_linear_video = nn.Linear(
            config.hidden_size, config.video_in_channels, bias=False, dtype=torch.float32
        )
        self.final_linear_audio = nn.Linear(
            config.hidden_size, config.audio_in_channels, bias=False, dtype=torch.float32
        )

    def forward(self, hidden_states: torch.Tensor, video_indices: torch.Tensor) -> torch.Tensor:
        video = self.final_norm_video(hidden_states.index_select(0, video_indices).float())
        return self.final_linear_video(video)


class Magi2RefinerTransformer(nn.Module):
    """Native refiner DiT with the released checkpoint hierarchy."""

    def __init__(self, config: Magi2RefinerConfig | None = None) -> None:
        super().__init__()
        self.config = config or MAGI2_REFINER_CONFIG
        self.config.validate()
        self.pre_adapter = Magi2RefinerPreAdapter(self.config)
        self.block = Magi2RefinerBlock(self.config)
        self.post_adapter = Magi2RefinerPostAdapter(self.config)

    @property
    def layers(self) -> nn.ModuleList:
        return self.block.layers

    def compile_regions(self, **compile_kwargs: Any) -> None:
        """Compile the dense compute around the eager window attention of every layer."""

        for layer in self.block.layers:
            for name in layer.region_methods:
                setattr(layer, name, torch.compile(getattr(layer, name), **compile_kwargs))

    def forward(
        self,
        video: torch.Tensor,
        text: torch.Tensor,
        coords: torch.Tensor,
        modality: torch.Tensor,
        window_plan: Magi2WindowAttentionPlan,
        split_sizes: Sequence[int] | None = None,
    ) -> torch.Tensor:
        """Predict the velocity of one Ulysses rank's video tokens.

        The inputs are the rank's contiguous slice of the block-major sequence
        that ``window_plan`` describes: ``video`` ``[Nv, Cv]`` and ``text``
        ``[Nt, Ct]`` hold the features of its video and text tokens, and
        ``coords`` ``[n, 9]`` and ``modality`` ``[n]`` describe all ``n``
        tokens in order.  ``split_sizes`` holds every rank's token count and
        defaults to a single rank.  Returns the fp32 ``[Nv, Cv]`` velocity in
        the order of ``video``.
        """

        num_tokens = modality.shape[0]
        if modality.ndim != 1 or coords.shape != (num_tokens, 9):
            raise ValueError(
                f"expected modality [n] and coords [n, 9], got {tuple(modality.shape)} and {tuple(coords.shape)}"
            )
        modality_dispatcher = ModalityDispatcher(modality, _NUM_MODALITIES)
        group_sizes = modality_dispatcher.group_size_cpu
        if len(group_sizes) != _NUM_MODALITIES or group_sizes[int(Modality.AUDIO)]:
            raise ValueError("MAGI-2 refiner tokens must be video or text")
        num_video, num_text = group_sizes[int(Modality.VIDEO)], group_sizes[int(Modality.TEXT)]
        if video.shape[0] != num_video or text.shape[0] != num_text:
            raise ValueError(
                f"modality has {num_video} video and {num_text} text tokens, "
                f"got {video.shape[0]} video and {text.shape[0]} text features"
            )
        split_sizes = [num_tokens] if split_sizes is None else [int(size) for size in split_sizes]
        attn_metadata = AttentionMetadata(
            extra={"magi2_window_plan": window_plan, "magi2_split_sizes": split_sizes},
        )
        video_indices = torch.nonzero(modality == int(Modality.VIDEO)).flatten()
        text_indices = torch.nonzero(modality == int(Modality.TEXT)).flatten()

        rope = self.pre_adapter.rope(coords)
        hidden_states = self.pre_adapter(video, text, video_indices, text_indices)
        hidden_states = modality_dispatcher.permute(hidden_states.to(self.config.params_dtype))
        hidden_states = self.block(hidden_states, rope, modality_dispatcher, attn_metadata)
        hidden_states = modality_dispatcher.inverse_permute(hidden_states)
        return self.post_adapter(hidden_states, video_indices)

    def load_checkpoint(self, checkpoint_dir: str | os.PathLike[str]) -> set[str]:
        """Strictly load the released refiner checkpoint.

        ``checkpoint_dir`` is the refiner directory holding
        ``model.safetensors.index.json`` and its shards.  The index must list
        exactly this module's tensors, every shard must hold exactly the
        tensors the index assigns to it, and every tensor must match its
        parameter's shape and dtype.  All of this is checked from the shard
        headers before any parameter is written.
        """

        root = Path(checkpoint_dir)
        index_path = root / CHECKPOINT_INDEX_FILE
        if not index_path.is_file():
            raise FileNotFoundError(f"MAGI-2 refiner checkpoint index {index_path} does not exist")
        with open(index_path, encoding="utf-8") as index_file:
            weight_map = json.load(index_file).get("weight_map")
        if not isinstance(weight_map, dict):
            raise ValueError(f"{index_path} has no weight_map")

        targets = self.state_dict(keep_vars=True)
        missing = sorted(set(targets) - set(weight_map))
        unexpected = sorted(set(weight_map) - set(targets))
        if missing or unexpected:
            raise ValueError(
                f"MAGI-2 refiner checkpoint keys do not match the model: "
                f"{len(missing)} missing {missing[:8]}, {len(unexpected)} unexpected {unexpected[:8]}"
            )

        shards: dict[str, list[str]] = {}
        for name, shard in weight_map.items():
            if not isinstance(shard, str) or Path(shard).name != shard or shard in ("", ".", ".."):
                raise ValueError(f"MAGI-2 refiner shard {shard!r} for {name!r} is not a file name in {root}")
            shards.setdefault(shard, []).append(name)
        for shard, names in shards.items():
            with safe_open(root / shard, framework="pt", device="cpu") as weights:
                stored = set(weights.keys())
                absent = sorted(set(names) - stored)
                extra = sorted(stored - set(names))
                if absent or extra:
                    raise ValueError(
                        f"MAGI-2 refiner shard {shard} does not match the index: "
                        f"missing {absent[:8]}, unexpected {extra[:8]}"
                    )
                for name in names:
                    header = weights.get_slice(name)
                    shape = tuple(header.get_shape())
                    dtype = _SAFETENSORS_DTYPES.get(header.get_dtype())
                    target = targets[name]
                    if shape != tuple(target.shape) or dtype != target.dtype:
                        raise ValueError(
                            f"MAGI-2 refiner weight {name!r} is {header.get_dtype()}{list(shape)}, "
                            f"expected {target.dtype}{list(target.shape)}"
                        )

        with torch.no_grad():
            for shard, names in shards.items():
                with safe_open(root / shard, framework="pt", device="cpu") as weights:
                    for name in names:
                        targets[name].copy_(weights.get_tensor(name))
        return set(weight_map)


__all__ = [
    "CHECKPOINT_INDEX_FILE",
    "Magi2RefinerAttention",
    "Magi2RefinerBlock",
    "Magi2RefinerLayer",
    "Magi2RefinerMLP",
    "Magi2RefinerPostAdapter",
    "Magi2RefinerPreAdapter",
    "Magi2RefinerTransformer",
]
