# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Copyright (c) 2026 SandAI. All Rights Reserved.

"""Native MAGI-2 Preview diffusion transformer.

The architecture and mHC/MoE equations are adapted from SandAI's Apache-2.0
MAGI-2 Preview implementation.  This modified vLLM-Omni version removes the
external ``inference`` package, MagiCompiler, and SandAI process manager while
preserving the released checkpoint module names and tensor layouts.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from enum import IntEnum
from functools import partial

import torch
import torch.nn as nn

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.layer import Attention

from .attention import Magi2PackedAttentionKernel, VarlenHandler, apply_rotary_emb
from .configuration_magi2 import Magi2PreviewConfig
from .layers import (
    ElementWiseFourierEmbed,
    MHCHandler,
    ModalityDispatcher,
    MultiModalityRMSNorm,
    make_grouped_linear,
    swiglu7,
)
from .mh_moe import Magi2MultiHeadMoE, Magi2MultiHeadMoEConfig
from .parallel import Magi2SequenceDispatcher


class Modality(IntEnum):
    VIDEO = 0
    AUDIO = 1
    TEXT = 2
    TIME = 3


class Magi2Attention(nn.Module):
    def __init__(
        self,
        config: Magi2PreviewConfig,
        *,
        num_modality: int,
    ) -> None:
        super().__init__()
        self.config = config
        self.num_modality = num_modality
        self.head_dim = config.head_dim
        global_q_size = config.num_heads_q * self.head_dim
        global_kv_size = config.num_heads_kv * self.head_dim

        self.pre_norm = MultiModalityRMSNorm(
            config.hidden_size,
            num_modality=num_modality,
        )
        self.linear_g = make_grouped_linear(
            config.hidden_size,
            config.num_heads_q,
            num_experts=num_modality,
            bias=False,
            dtype=config.params_dtype,
            parallel_mode="column",
        )
        self.linear_qkv = make_grouped_linear(
            config.hidden_size,
            global_q_size + 2 * global_kv_size,
            num_experts=num_modality,
            bias=False,
            dtype=config.params_dtype,
            parallel_mode="column",
            qkv_splits=(global_q_size, global_kv_size, global_kv_size),
        )
        self.linear_proj = make_grouped_linear(
            global_q_size,
            config.hidden_size,
            num_experts=num_modality,
            bias=False,
            dtype=config.params_dtype,
            parallel_mode="row",
        )
        self.tp_group = self.linear_qkv.tp_group
        self.num_heads_q = config.num_heads_q // self.tp_group.world_size
        self.num_heads_kv = config.num_heads_kv // self.tp_group.world_size
        self.q_size = self.num_heads_q * self.head_dim
        self.kv_size = self.num_heads_kv * self.head_dim
        self.sinks = nn.Parameter(torch.empty(config.attention_sink_tokens, self.num_heads_q, dtype=torch.float32))
        if self.tp_group.world_size > 1:
            self.sinks.checkpoint_weight_transform = self._shard_sinks
        self.q_norm = MultiModalityRMSNorm(
            self.head_dim,
            num_modality=num_modality,
            out_dtype=torch.float32,
        )
        self.k_norm = MultiModalityRMSNorm(
            self.head_dim,
            num_modality=num_modality,
            out_dtype=torch.float32,
        )
        # MAGI-2's packed CFG sequences, learned sink logits, softcap, and
        # uneven Ulysses splits require a model kernel. Route it through the
        # framework Attention layer so compile/dispatch ownership remains
        # shared while the model kernel owns its specialized communication.
        self.packed_attention = Attention(
            num_heads=self.num_heads_q,
            num_kv_heads=self.num_heads_kv,
            head_size=self.head_dim,
            causal=False,
            softmax_scale=self.head_dim**-0.5,
            qkv_layout="THD",
            skip_sequence_parallel=True,
            disable_kv_quant=True,
            custom_attention=Magi2PackedAttentionKernel(config.attention_softcap),
        )

    def _shard_sinks(self, checkpoint_tensor: torch.Tensor) -> torch.Tensor:
        if tuple(checkpoint_tensor.shape) == tuple(self.sinks.shape):
            return checkpoint_tensor
        expected = (self.config.attention_sink_tokens, self.config.num_heads_q)
        if tuple(checkpoint_tensor.shape) != expected:
            raise ValueError(f"attention sinks have shape {tuple(checkpoint_tensor.shape)}, expected {expected}")
        start = self.tp_group.rank * self.num_heads_q
        return checkpoint_tensor[:, start : start + self.num_heads_q]

    def forward(
        self,
        hidden_states: torch.Tensor,
        rope: torch.Tensor,
        varlen_handler: VarlenHandler,
        modality_dispatcher: ModalityDispatcher,
        cp_split_sizes: list[int] | torch.Tensor,
    ) -> torch.Tensor:
        normalized = self.pre_norm(hidden_states, modality_dispatcher)
        gates = self.linear_g(normalized, modality_dispatcher)
        qkv = self.linear_qkv(normalized, modality_dispatcher)
        q, k, v = torch.split(qkv, (self.q_size, self.kv_size, self.kv_size), dim=-1)
        q = q.view(-1, self.num_heads_q, self.head_dim)
        k = k.view(-1, self.num_heads_kv, self.head_dim)
        v = v.view(-1, self.num_heads_kv, self.head_dim)
        gates = gates.view(-1, self.num_heads_q, 1)

        q = self.q_norm(q, modality_dispatcher)
        k = self.k_norm(k, modality_dispatcher)
        q = modality_dispatcher.inverse_permute(q).unsqueeze(0)
        k = modality_dispatcher.inverse_permute(k).unsqueeze(0)
        v = modality_dispatcher.inverse_permute(v).unsqueeze(0)
        sin, cos = rope.tensor_split(2, dim=-1)
        # The released implementation normalizes Q/K in fp32, applies RoPE,
        # then converts Q/K/V to the checkpoint dtype at the attention
        # boundary.  This conversion is also required by FlashAttention.
        q = apply_rotary_emb(q, cos, sin).squeeze(0).to(self.config.params_dtype)
        k = apply_rotary_emb(k, cos, sin).squeeze(0).to(self.config.params_dtype)
        v = v.squeeze(0).to(self.config.params_dtype)

        output = self.packed_attention(
            q,
            k,
            v,
            AttentionMetadata(
                extra={
                    "magi2_varlen": varlen_handler,
                    "magi2_split_sizes": cp_split_sizes,
                    "magi2_sink": self.sinks,
                }
            ),
        )
        output = modality_dispatcher.permute(output)
        output = output * torch.sigmoid(gates)
        output = output.reshape(-1, self.q_size).to(self.config.params_dtype)
        return self.linear_proj(output, modality_dispatcher)


class Magi2MLP(nn.Module):
    def __init__(
        self,
        config: Magi2PreviewConfig,
        *,
        num_modality: int,
    ) -> None:
        super().__init__()
        intermediate_size = int(config.hidden_size * config.intermediate_factor * 2 / 3) // 128 * 128
        # Tiny unit-test configurations can be narrower than the release's
        # 128-channel rounding unit while retaining the same formula family.
        if intermediate_size == 0:
            intermediate_size = max(1, int(config.hidden_size * config.intermediate_factor * 2 / 3))
        self.intermediate_size = intermediate_size
        self.pre_norm = MultiModalityRMSNorm(config.hidden_size, num_modality=num_modality)
        self.up_gate_proj = make_grouped_linear(
            config.hidden_size,
            2 * intermediate_size,
            num_experts=num_modality,
            bias=False,
            dtype=config.params_dtype,
            parallel_mode="column",
        )
        self.down_proj = make_grouped_linear(
            intermediate_size,
            config.hidden_size,
            num_experts=num_modality,
            bias=False,
            dtype=config.params_dtype,
            parallel_mode="row",
        )

    def forward(self, hidden_states: torch.Tensor, dispatcher: ModalityDispatcher) -> torch.Tensor:
        hidden_states = self.pre_norm(hidden_states, dispatcher)
        hidden_states = self.up_gate_proj(hidden_states, dispatcher)
        hidden_states = swiglu7(hidden_states)
        return self.down_proj(hidden_states, dispatcher)


class Magi2MultiHeadMoELayer(nn.Module):
    def __init__(self, config: Magi2PreviewConfig) -> None:
        super().__init__()
        moe = config.moe
        self.config = config
        self.pre_norm = MultiModalityRMSNorm(config.hidden_size, num_modality=3)
        self.split_linear = make_grouped_linear(
            config.hidden_size,
            config.hidden_size,
            bias=False,
            dtype=config.params_dtype,
            parallel_mode="column",
        )
        self.moe_mlp = Magi2MultiHeadMoE(
            Magi2MultiHeadMoEConfig(
                hidden_size=config.hidden_size,
                num_heads=moe.num_heads,
                num_experts=moe.num_experts,
                top_k=moe.top_k,
                expert_intermediate_size=moe.expert_intermediate_size,
                params_dtype=config.params_dtype,
                score_func=moe.score_function,  # type: ignore[arg-type]
                route_norm=moe.normalize_routing_weights,
                route_scale=moe.routing_scale,
            )
        )
        self.merge_linear = make_grouped_linear(
            config.hidden_size,
            config.hidden_size,
            bias=False,
            dtype=config.params_dtype,
            parallel_mode="row",
        )
        self.shared_expert_fc1 = make_grouped_linear(
            config.hidden_size,
            2 * moe.shared_expert_intermediate_size,
            bias=False,
            dtype=config.params_dtype,
            parallel_mode="column",
        )
        self.shared_expert_fc2 = make_grouped_linear(
            moe.shared_expert_intermediate_size,
            config.hidden_size,
            bias=False,
            dtype=config.params_dtype,
            parallel_mode="row",
        )
        self.modality_specific_shared_expert_fc1 = make_grouped_linear(
            config.hidden_size,
            2 * moe.modality_shared_expert_intermediate_size,
            num_experts=3,
            bias=False,
            dtype=config.params_dtype,
            parallel_mode="column",
        )
        self.modality_specific_shared_expert_fc2 = make_grouped_linear(
            moe.modality_shared_expert_intermediate_size,
            config.hidden_size,
            num_experts=3,
            bias=False,
            dtype=config.params_dtype,
            parallel_mode="row",
        )
        tp_size = self.split_linear.tp_group.world_size
        self.local_shared_expert_intermediate_size = moe.shared_expert_intermediate_size // tp_size
        self.local_modality_shared_expert_intermediate_size = moe.modality_shared_expert_intermediate_size // tp_size

    def _shared_experts(
        self,
        normalized: torch.Tensor,
        dispatcher: ModalityDispatcher,
    ) -> torch.Tensor:
        shared = self.shared_expert_fc1(normalized)
        modality = self.modality_specific_shared_expert_fc1(normalized, dispatcher)
        activated = swiglu7(torch.cat((shared, modality), dim=-1))
        shared, modality = activated.split(
            (
                self.local_shared_expert_intermediate_size,
                self.local_modality_shared_expert_intermediate_size,
            ),
            dim=-1,
        )
        return self.shared_expert_fc2(shared) + self.modality_specific_shared_expert_fc2(
            modality.contiguous(), dispatcher
        )

    def forward(self, hidden_states: torch.Tensor, dispatcher: ModalityDispatcher) -> torch.Tensor:
        normalized = self.pre_norm(hidden_states, dispatcher)
        routed = self.split_linear(normalized)
        routed = self.moe_mlp(routed)
        routed = self.merge_linear(routed)
        return routed + self._shared_experts(normalized, dispatcher)


class Magi2PreAdapter(nn.Module):
    def __init__(self, config: Magi2PreviewConfig) -> None:
        super().__init__()
        self.config = config
        self.adapter_dim = config.virtual_width
        self.video_embedder = nn.Linear(
            config.video_in_channels,
            self.adapter_dim,
            bias=True,
            dtype=torch.float32,
        )
        self.text_embedder = nn.Linear(
            config.text_in_channels,
            self.adapter_dim,
            bias=True,
            dtype=torch.float32,
        )
        self.audio_embedder = nn.Linear(
            config.audio_in_channels,
            self.adapter_dim,
            bias=True,
            dtype=torch.float32,
        )
        self.rope = ElementWiseFourierEmbed(config.head_dim, learnable=False)

    def forward(
        self,
        packed: torch.Tensor,
        coords_mapping: torch.Tensor,
        video_indices: torch.Tensor,
        audio_indices: torch.Tensor,
        text_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rope = self.rope(coords_mapping)
        output = torch.zeros(
            packed.shape[0],
            self.adapter_dim,
            dtype=torch.float32,
            device=packed.device,
        )
        if text_indices.numel():
            output.index_copy_(
                0,
                text_indices,
                self.text_embedder(packed.index_select(0, text_indices)[:, : self.config.text_in_channels].float()),
            )
        if audio_indices.numel():
            output.index_copy_(
                0,
                audio_indices,
                self.audio_embedder(packed.index_select(0, audio_indices)[:, : self.config.audio_in_channels].float()),
            )
        if video_indices.numel():
            output.index_copy_(
                0,
                video_indices,
                self.video_embedder(packed.index_select(0, video_indices)[:, : self.config.video_in_channels].float()),
            )
        return output, rope


class Magi2PostAdapter(nn.Module):
    def __init__(self, config: Magi2PreviewConfig) -> None:
        super().__init__()
        self.config = config
        self.adapter_dim = config.virtual_width
        self.final_norm_video = MultiModalityRMSNorm(self.adapter_dim)
        self.final_norm_audio = MultiModalityRMSNorm(self.adapter_dim)
        self.final_linear_video = nn.Linear(
            self.adapter_dim,
            config.video_in_channels,
            bias=False,
            dtype=torch.float32,
        )
        self.final_linear_audio = nn.Linear(
            self.adapter_dim,
            config.audio_in_channels,
            bias=False,
            dtype=torch.float32,
        )
        self.final_out_dim = max(config.video_in_channels, config.audio_in_channels)

    def forward(
        self,
        hidden_states: torch.Tensor,
        video_indices: torch.Tensor,
        audio_indices: torch.Tensor,
    ) -> torch.Tensor:
        output = torch.zeros(
            hidden_states.shape[0],
            self.final_out_dim,
            dtype=torch.float32,
            device=hidden_states.device,
        )
        if video_indices.numel():
            video = self.final_norm_video(hidden_states.index_select(0, video_indices).float())
            video = self.final_linear_video(video).float()
            output[:, : self.config.video_in_channels].index_copy_(0, video_indices, video)
        if audio_indices.numel():
            audio = self.final_norm_audio(hidden_states.index_select(0, audio_indices).float())
            audio = self.final_linear_audio(audio).float()
            output[:, : self.config.audio_in_channels].index_copy_(0, audio_indices, audio)
        return output


class Magi2TransformerLayer(nn.Module):
    def __init__(self, config: Magi2PreviewConfig, layer_index: int) -> None:
        super().__init__()
        num_modality = 3 if layer_index in config.multimodal_layers else 1
        self.config = config
        self.attention = Magi2Attention(config, num_modality=num_modality)
        self.mlp: nn.Module
        if layer_index in config.moe.layers:
            self.mlp = Magi2MultiHeadMoELayer(config)
        else:
            self.mlp = Magi2MLP(config, num_modality=num_modality)
        self._init_mhc(num_modality)

    def _init_mhc(self, num_modality: int) -> None:
        streams = self.config.mhc.num_streams
        hidden = self.config.hidden_size
        alpha = self.config.mhc.alpha_init
        for branch in ("attn", "mlp"):
            for coefficient in ("pre", "post", "res"):
                self.register_parameter(
                    f"mhc_alpha_{coefficient}_{branch}",
                    nn.Parameter(torch.full((1,), alpha, dtype=torch.float32)),
                )
            self.register_parameter(
                f"mhc_bias_pre_{branch}",
                nn.Parameter(torch.empty(streams, dtype=torch.float32)),
            )
            self.register_parameter(
                f"mhc_bias_post_{branch}",
                nn.Parameter(torch.empty(streams, dtype=torch.float32)),
            )
            self.register_parameter(
                f"mhc_bias_res_{branch}",
                nn.Parameter(torch.empty(streams, streams, dtype=torch.float32)),
            )
            self.register_parameter(
                f"mhc_phi_fused_{branch}",
                nn.Parameter(
                    torch.empty(
                        streams * hidden,
                        streams + streams + streams * streams,
                        dtype=torch.float32,
                    )
                ),
            )
        self.mhc_norm = MultiModalityRMSNorm(
            streams * hidden,
            num_modality=num_modality,
            out_dtype=torch.float32,
        )
        self.mhc_handler = MHCHandler(
            streams,
            hidden,
            sinkhorn_iterations=self.config.mhc.sinkhorn_iterations,
            sinkhorn_epsilon=self.config.mhc.sinkhorn_epsilon,
        )

    def _branch_logits(
        self,
        streams: torch.Tensor,
        branch: str,
        dispatcher: ModalityDispatcher,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.mhc_handler.compute_logits(
            self.mhc_handler.flatten(streams),
            partial(self.mhc_norm, modality_dispatcher=dispatcher),
            getattr(self, f"mhc_phi_fused_{branch}"),
        )

    def _branch_input(
        self,
        streams: torch.Tensor,
        branch: str,
        logits: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        pre_logits, _, _ = logits
        return self.mhc_handler.apply_pre(
            streams,
            (
                getattr(self, f"mhc_alpha_pre_{branch}"),
                getattr(self, f"mhc_bias_pre_{branch}"),
                pre_logits,
            ),
            out_dtype=streams.dtype,
        )

    def _connect(
        self,
        streams: torch.Tensor,
        output: torch.Tensor,
        branch: str,
        logits: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        _, post_logits, residual_logits = logits
        post, residual = self.mhc_handler.compute_post_residual(
            (
                getattr(self, f"mhc_alpha_post_{branch}"),
                getattr(self, f"mhc_bias_post_{branch}"),
                post_logits,
            ),
            (
                getattr(self, f"mhc_alpha_res_{branch}"),
                getattr(self, f"mhc_bias_res_{branch}"),
                residual_logits,
            ),
            out_dtype=streams.dtype,
        )
        return self.mhc_handler.hyper_connect(streams, output, post, residual)

    def forward(
        self,
        hidden_states: torch.Tensor,
        rope: torch.Tensor,
        varlen_handler: VarlenHandler,
        modality_dispatcher: ModalityDispatcher,
        cp_split_sizes: list[int] | torch.Tensor,
    ) -> torch.Tensor:
        streams = hidden_states.reshape(hidden_states.shape[0], self.config.mhc.num_streams, self.config.hidden_size)
        attention_logits = self._branch_logits(streams, "attn", modality_dispatcher)
        attention_input = self._branch_input(streams, "attn", attention_logits)
        attention_output = self.attention(
            attention_input,
            rope,
            varlen_handler,
            modality_dispatcher,
            cp_split_sizes,
        )
        streams = self._connect(streams, attention_output, "attn", attention_logits)
        mlp_logits = self._branch_logits(streams, "mlp", modality_dispatcher)
        mlp_input = self._branch_input(streams, "mlp", mlp_logits)
        mlp_output = self.mlp(mlp_input, modality_dispatcher)
        streams = self._connect(streams, mlp_output, "mlp", mlp_logits)
        return streams.reshape(streams.shape[0], -1)


class Magi2TransformerBlock(nn.Module):
    def __init__(self, config: Magi2PreviewConfig) -> None:
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(Magi2TransformerLayer(config, index) for index in range(config.num_layers))

    def __iter__(self):
        """Expose checkpoint-nested layers to the layerwise offload backends."""

        return iter(self.layers)

    def forward(
        self,
        hidden_states: torch.Tensor,
        rope: torch.Tensor,
        varlen_handler: VarlenHandler,
        modality_dispatcher: ModalityDispatcher,
        cp_split_sizes: list[int] | torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = hidden_states.to(self.config.params_dtype)
        hidden_states = modality_dispatcher.permute(hidden_states)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                rope,
                varlen_handler,
                modality_dispatcher,
                cp_split_sizes,
            )
        return modality_dispatcher.inverse_permute(hidden_states)


def _is_magi2_transformer_layer(_name: str, module: nn.Module) -> bool:
    """Shard one Preview layer at a time under the shared FSDP2/HSDP path."""

    return isinstance(module, Magi2TransformerLayer)


class Magi2PreviewTransformer(nn.Module):
    """Native preview DiT with the released checkpoint hierarchy."""

    # ``block`` must remain the registered child name because it is part of the
    # released checkpoint hierarchy.  The ``layers`` property below exposes its
    # ModuleList through the standard layerwise-offload contract without
    # changing the registered module names.
    _layerwise_offload_blocks_attrs = ["layers"]
    _hsdp_shard_conditions = [_is_magi2_transformer_layer]
    _hsdp_preserve_parameter_dtypes = True
    _EP_SHARDED_SUFFIXES = (
        ".moe_mlp.gate",
        ".moe_mlp.W_gate",
        ".moe_mlp.W_up",
        ".moe_mlp.W_down",
        ".moe_mlp.router.expert_bias",
        ".moe_mlp.router.expert_bias_ema",
    )

    def __init__(self, config: Magi2PreviewConfig | None = None) -> None:
        super().__init__()
        self.config = config or Magi2PreviewConfig()
        self.config.validate()
        self.pre_adapter = Magi2PreAdapter(self.config)
        self.post_adapter = Magi2PostAdapter(self.config)
        self.block = Magi2TransformerBlock(self.config)
        # SP doubles as MAGI's MoE-head parallel axis. These modules therefore
        # contain different checkpoint slices on each SP rank and must remain
        # rank-local while HSDP shards the replicated parameters around them.
        self._hsdp_ignored_modules = [
            f"block.layers.{index}.mlp.moe_mlp"
            for index, layer in enumerate(self.block.layers)
            if isinstance(layer.mlp, Magi2MultiHeadMoELayer)
        ]

    @property
    def layers(self) -> nn.ModuleList:
        """Expose Preview layers to the shared offload block discovery API."""

        return self.block.layers

    def forward(
        self,
        x: torch.Tensor,
        coords_mapping: torch.Tensor,
        modality_mapping: torch.Tensor,
        varlen_handler: VarlenHandler,
        time_token_sequence: torch.Tensor | None = None,
    ) -> torch.Tensor:
        dispatcher = Magi2SequenceDispatcher()
        x = dispatcher.dispatch(x)
        coords_mapping = dispatcher.dispatch(coords_mapping)
        modality_mapping = dispatcher.dispatch(modality_mapping)
        if time_token_sequence is not None:
            time_token_sequence = dispatcher.dispatch(time_token_sequence)
        assert dispatcher.split_sizes is not None
        cp_split_sizes = dispatcher.split_sizes

        time_mask = modality_mapping == int(Modality.TIME)
        modality_mapping = torch.where(time_mask, int(Modality.TEXT), modality_mapping)
        modality_dispatcher = ModalityDispatcher(modality_mapping, 3)
        video_indices = torch.nonzero(modality_mapping == int(Modality.VIDEO)).flatten()
        audio_indices = torch.nonzero(modality_mapping == int(Modality.AUDIO)).flatten()
        text_indices = torch.nonzero(modality_mapping == int(Modality.TEXT)).flatten()

        hidden_states, rope = self.pre_adapter(
            x,
            coords_mapping,
            video_indices,
            audio_indices,
            text_indices,
        )
        if time_token_sequence is not None and time_token_sequence.shape[-1] > 0:
            hidden_states[:, : time_token_sequence.shape[-1]] = time_token_sequence.to(hidden_states.dtype)
        hidden_states = self.block(
            hidden_states,
            rope,
            varlen_handler,
            modality_dispatcher,
            cp_split_sizes,
        )
        output = self.post_adapter(hidden_states, video_indices, audio_indices)
        return dispatcher.undispatch(output)

    def _moe_for_weight(self, name: str) -> Magi2MultiHeadMoE | None:
        if ".moe_mlp." not in name:
            return None
        module_name = name.split(".moe_mlp.", 1)[0] + ".moe_mlp"
        module = dict(self.named_modules()).get(module_name)
        return module if isinstance(module, Magi2MultiHeadMoE) else None

    def validate_loaded_weights(self, loaded_names: set[str]) -> None:
        """Fail closed when the mmap loader misses a Preview tensor."""

        loaded_names = {name.removeprefix("transformer.") for name in loaded_names}
        loaded_names = {f"block.{name}" if name.startswith("layers.") else name for name in loaded_names}
        expected = {name for name, _ in self.named_parameters()}
        missing = expected - loaded_names
        if missing:
            raise ValueError(f"MAGI-2 Preview mmap loading is missing {len(missing)} weights: {sorted(missing)[:8]}")

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Strictly load and TP/MoE-slice the released Preview checkpoint.

        This is the ordinary loader used by resident and rank-local DLO
        deployments.  DLO+AllGather uses the pipeline's mmap mapping and the
        same per-parameter transforms before orthogonal DP sharding.
        """

        targets = self.state_dict(keep_vars=True)
        loaded: set[str] = set()
        for raw_name, checkpoint_tensor in weights:
            name = raw_name.removeprefix("transformer.")
            if name not in targets:
                raise ValueError(f"unexpected MAGI-2 preview checkpoint weight {raw_name!r}")
            target = targets[name]
            checkpoint_transform = getattr(target, "checkpoint_weight_transform", None)
            moe = self._moe_for_weight(name)
            if callable(checkpoint_transform):
                checkpoint_tensor = checkpoint_transform(checkpoint_tensor)
            elif moe is not None and name.endswith(self._EP_SHARDED_SUFFIXES):
                checkpoint_tensor = moe.ep_slice(checkpoint_tensor)
            if tuple(target.shape) != tuple(checkpoint_tensor.shape):
                raise ValueError(
                    f"MAGI-2 weight {name!r} has shape {tuple(checkpoint_tensor.shape)}, expected {tuple(target.shape)}"
                )
            with torch.no_grad():
                target.copy_(checkpoint_tensor.to(device=target.device, dtype=target.dtype))
            loaded.add(name)

        if (os.environ.get("MAGI2_ROUTER_BIAS_SOURCE") or "ema").strip().lower() != "main":
            for module_name, module in self.named_modules():
                if not isinstance(module, Magi2MultiHeadMoE):
                    continue
                module.router.expert_bias.copy_(module.router.expert_bias_ema)
                loaded.add(f"{module_name}.router.expert_bias")

        missing = set(targets) - loaded
        if missing:
            raise ValueError(f"MAGI-2 preview checkpoint is missing {len(missing)} weights: {sorted(missing)[:8]}")
        return loaded


__all__ = [
    "Magi2PreviewTransformer",
    "Modality",
    "VarlenHandler",
]
