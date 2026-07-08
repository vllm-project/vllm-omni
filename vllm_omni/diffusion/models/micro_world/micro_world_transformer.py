# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Modifications Copyright(C) 2025 Advanced Micro Devices, Inc. All rights reserved.
"""
Optimized transformer models for AMD Micro-World (T2W and I2W variants).

These extend the vLLM-optimized WanTransformer3DModel with action-injection
mechanisms:
- MicroWorldControlNetTransformer: ControlNet-style parallel action branch (T2W)
- MicroWorldAdaLNTransformer: AdaLN modulation of timestep embeddings (I2W)

Ported from: https://github.com/AMD-AGI/Micro-World
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from vllm_omni.diffusion.forward_context import get_forward_context
from vllm_omni.diffusion.models.wan2_2.wan2_2_transformer import (
    WanTransformer3DModel,
    WanTransformerBlock,
)

from .action_module import ActionModule

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# ControlNet variant (T2W) — parallel action branch with skip connections
# ---------------------------------------------------------------------------


class BaseWanTransformerBlock(WanTransformerBlock):
    """Standard transformer block extended with optional hint skip connections.

    When ``block_id`` is not None, the output of the block is augmented by a
    hint tensor (from the parallel action branch) scaled by ``context_scale``.
    """

    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        eps: float = 1e-6,
        added_kv_proj_dim: int | None = None,
        cross_attn_norm: bool = False,
        block_id: int | None = None,
    ):
        super().__init__(dim, ffn_dim, num_heads, eps, added_kv_proj_dim, cross_attn_norm)
        self.block_id = block_id

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor],
        hidden_states_mask: torch.Tensor | None = None,
        hints: list[torch.Tensor] | None = None,
        context_scale: float = 1.0,
    ) -> torch.Tensor:
        hidden_states = super().forward(hidden_states, encoder_hidden_states, temb, rotary_emb, hidden_states_mask)
        if hints is not None and self.block_id is not None:
            hidden_states = hidden_states + hints[self.block_id] * context_scale
        return hidden_states


class WanActionAttentionBlock(WanTransformerBlock):
    """Transformer block for the parallel ControlNet action branch.

    Block 0 projects raw action features via ``before_proj``. Each block emits
    a skip via ``after_proj``; skips accumulate in a stacked tensor.
    """

    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        action_dim: int,
        eps: float = 1e-6,
        added_kv_proj_dim: int | None = None,
        cross_attn_norm: bool = False,
        block_id: int = 0,
    ):
        super().__init__(dim, ffn_dim, num_heads, eps, added_kv_proj_dim, cross_attn_norm)
        self.block_id = block_id

        if block_id == 0:
            self.before_proj = nn.Linear(action_dim, dim)
            nn.init.zeros_(self.before_proj.weight)
            nn.init.zeros_(self.before_proj.bias)

        self.after_proj = nn.Linear(dim, dim)
        nn.init.zeros_(self.after_proj.weight)
        nn.init.zeros_(self.after_proj.bias)

    def forward(
        self,
        conditions: torch.Tensor,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor],
        hidden_states_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.block_id == 0:
            c = self.before_proj(conditions) + hidden_states
            all_c = []
        else:
            all_c = list(torch.unbind(conditions))
            c = all_c.pop(-1)

        c = super().forward(c, encoder_hidden_states, temb, rotary_emb, hidden_states_mask)

        c_skip = self.after_proj(c)

        all_c.append(c_skip)
        all_c.append(c)
        stacked = torch.stack(all_c)

        return stacked


class MicroWorldControlNetTransformer(WanTransformer3DModel):
    """Wan transformer with ControlNet-style action injection for T2W generation."""

    def __init__(
        self,
        action_dim: int = 1536,
        mouse_dim: int = 2,
        keyboard_dim: int = 7,
        action_layers: list[int] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        num_layers = kwargs.get("num_layers", 40)
        inner_dim = kwargs.get("num_attention_heads", 40) * kwargs.get("attention_head_dim", 128)
        ffn_dim = kwargs.get("ffn_dim", 13824)
        num_heads = kwargs.get("num_attention_heads", 40)
        eps = kwargs.get("eps", 1e-6)
        added_kv_proj_dim = kwargs.get("added_kv_proj_dim", None)
        cross_attn_norm = kwargs.get("cross_attn_norm", True)

        # Determine which main blocks receive action hints
        if action_layers is None:
            action_layers = list(range(0, num_layers, 2))
        self.action_layers = action_layers
        self.action_layers_mapping = {layer_idx: n for n, layer_idx in enumerate(action_layers)}

        # Replace base blocks with BaseWanTransformerBlock (hint-aware)
        self.blocks = nn.ModuleList(
            [
                BaseWanTransformerBlock(
                    dim=inner_dim,
                    ffn_dim=ffn_dim,
                    num_heads=num_heads,
                    eps=eps,
                    added_kv_proj_dim=added_kv_proj_dim,
                    cross_attn_norm=cross_attn_norm,
                    block_id=self.action_layers_mapping.get(i),
                )
                for i in range(num_layers)
            ]
        )

        # Parallel action branch blocks
        self.action_blocks = nn.ModuleList(
            [
                WanActionAttentionBlock(
                    dim=inner_dim,
                    ffn_dim=ffn_dim,
                    num_heads=num_heads,
                    action_dim=action_dim,
                    eps=eps,
                    added_kv_proj_dim=added_kv_proj_dim,
                    cross_attn_norm=cross_attn_norm,
                    block_id=i,
                )
                for i in range(len(action_layers))
            ]
        )

        # Action preprocessor
        self.action_preprocess = ActionModule(
            mouse_dim=mouse_dim,
            keyboard_dim=keyboard_dim,
            action_dim=action_dim,
            window_size=3,
            temporal_ratio=4,
            flatten_spatial=True,
        )

    def forward_action(
        self,
        action_features: torch.Tensor,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor],
        hidden_states_mask: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        """Run the parallel action branch and collect skip connections.

        Returns:
            List of skip connection tensors, one per action block.
        """
        conditions = action_features
        for block in self.action_blocks:
            conditions = block(
                conditions,
                hidden_states,
                encoder_hidden_states,
                temb,
                rotary_emb,
                hidden_states_mask,
            )
        # Unstack: all elements except the last (which is the running hidden state)
        hints = list(torch.unbind(conditions))[:-1]
        return hints

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: torch.Tensor | None = None,
        mouse_actions: torch.Tensor | None = None,
        keyboard_actions: torch.Tensor | None = None,
        action_context_scale: float = 1.0,
        return_dict: bool = True,
        attention_kwargs: dict[str, Any] | None = None,
    ) -> torch.Tensor | Transformer2DModelOutput:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        # WanRotaryPosEmbed emits full head_dim via repeat_interleave(2); the
        # pair-rotation kernel expects half head_dim, so take the even slice.
        freqs_cos, freqs_sin = self.rope(hidden_states)
        rotary_emb = (
            freqs_cos[..., 0::2].to(dtype=hidden_states.dtype),
            freqs_sin[..., 0::2].to(dtype=hidden_states.dtype),
        )

        # Patch embedding and flatten to sequence
        hidden_states = self.patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        # Handle timestep shape
        if timestep.ndim == 2:
            ts_seq_len = timestep.shape[1]
            timestep = timestep.flatten()
        else:
            ts_seq_len = None

        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
            timestep, encoder_hidden_states, encoder_hidden_states_image, timestep_seq_len=ts_seq_len
        )
        timestep_proj = self.timestep_proj_prepare(timestep_proj, ts_seq_len)

        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

        # SP attention mask
        hidden_states_mask = None
        config = get_forward_context().omni_diffusion_config
        parallel_config = config.parallel_config
        if parallel_config is not None and parallel_config.sequence_parallel_size > 1:
            ctx = get_forward_context()
            if ctx.sp_original_seq_len is not None and ctx.sp_padding_size > 0:
                padded_seq_len = ctx.sp_original_seq_len + ctx.sp_padding_size
                hidden_states_mask = torch.ones(
                    batch_size, padded_seq_len, dtype=torch.bool, device=hidden_states.device
                )
                hidden_states_mask[:, ctx.sp_original_seq_len :] = False
        if hidden_states_mask is not None and hidden_states_mask.all():
            hidden_states_mask = None

        # Compute action features and skip connections
        hints = None
        if mouse_actions is not None and keyboard_actions is not None:
            grid_sizes = (post_patch_num_frames, post_patch_height, post_patch_width)
            action_features = self.action_preprocess(mouse_actions, keyboard_actions, grid_sizes)
            hints = self.forward_action(
                action_features,
                hidden_states,
                encoder_hidden_states,
                timestep_proj,
                rotary_emb,
                hidden_states_mask,
            )

        # Main transformer blocks with action hints
        for block in self.blocks:
            hidden_states = block(
                hidden_states,
                encoder_hidden_states,
                timestep_proj,
                rotary_emb,
                hidden_states_mask,
                hints=hints,
                context_scale=action_context_scale,
            )

        # Output norm, projection & unpatchify
        shift, scale = self.output_scale_shift_prepare(temb)
        shift = shift.to(hidden_states.device)
        scale = scale.to(hidden_states.device)
        if shift.ndim == 2:
            shift = shift.unsqueeze(1)
            scale = scale.unsqueeze(1)

        hidden_states = self.norm_out(hidden_states, scale, shift).type_as(hidden_states)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = hidden_states.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights with Wan2.1-to-vllm name mapping for both base and action blocks."""
        tp_rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()

        # QKV fusion mappings — apply to both blocks.N and action_blocks.N
        stacked_params_mapping = [
            (".attn1.to_qkv", ".attn1.to_q", "q"),
            (".attn1.to_qkv", ".attn1.to_k", "k"),
            (".attn1.to_qkv", ".attn1.to_v", "v"),
        ]
        self.stacked_params_mapping = stacked_params_mapping

        # Wan2.1 original → vllm-omni name remapping (top-level modules)
        # Order matters: more specific patterns first
        weight_name_remapping = [
            ("head.head.", "proj_out."),
            ("head.modulation", "output_scale_shift_prepare.scale_shift_table"),
            ("time_embedding.0.", "condition_embedder.time_embedder.linear_1."),
            ("time_embedding.2.", "condition_embedder.time_embedder.linear_2."),
            ("time_projection.1.", "condition_embedder.time_proj."),
            ("text_embedding.0.", "condition_embedder.text_embedder.linear_1."),
            ("text_embedding.2.", "condition_embedder.text_embedder.linear_2."),
            # I2V image embedder: reference MLPProj → vllm-omni norm1+ff+norm2.
            ("img_emb.proj.0.", "condition_embedder.image_embedder.norm1."),
            ("img_emb.proj.1.", "condition_embedder.image_embedder.ff.net.0.proj."),
            ("img_emb.proj.3.", "condition_embedder.image_embedder.ff.net.2."),
            ("img_emb.proj.4.", "condition_embedder.image_embedder.norm2."),
        ]

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            original_name = name

            for old, new in weight_name_remapping:
                if old in name:
                    name = name.replace(old, new)
                    break

            name = name.replace(".self_attn.q.", ".attn1.to_q.")
            name = name.replace(".self_attn.k.", ".attn1.to_k.")
            name = name.replace(".self_attn.v.", ".attn1.to_v.")
            name = name.replace(".self_attn.o.", ".attn1.to_out.")
            name = name.replace(".self_attn.norm_q.", ".attn1.norm_q.")
            name = name.replace(".self_attn.norm_k.", ".attn1.norm_k.")
            name = name.replace(".cross_attn.q.", ".attn2.to_q.")
            name = name.replace(".cross_attn.k.", ".attn2.to_k.")
            name = name.replace(".cross_attn.v.", ".attn2.to_v.")
            name = name.replace(".cross_attn.o.", ".attn2.to_out.")
            name = name.replace(".cross_attn.norm_q.", ".attn2.norm_q.")
            name = name.replace(".cross_attn.norm_k.", ".attn2.norm_k.")
            # I2V image cross-attn branch uses add_k_proj/add_v_proj in vllm-omni.
            name = name.replace(".cross_attn.k_img.", ".attn2.add_k_proj.")
            name = name.replace(".cross_attn.v_img.", ".attn2.add_v_proj.")
            name = name.replace(".cross_attn.norm_k_img.", ".attn2.norm_added_k.")
            if ".modulation." in name and "action_preprocess" not in name:
                name = name.replace(".modulation.", ".scale_shift_table.")
            if name.endswith(".modulation") and "action_preprocess" not in name:
                name = name[: -len(".modulation")] + ".scale_shift_table"
            name = name.replace(".norm3.", ".norm2.")

            for old, new in weight_name_remapping:
                name = name.replace(old, new)

            lookup_name = name

            fused = False
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                lookup_name = name.replace(weight_name, param_name)
                if lookup_name not in params_dict:
                    continue
                param = params_dict[lookup_name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                fused = True
                break

            if not fused:
                if ".ffn.0." in lookup_name:
                    lookup_name = lookup_name.replace(".ffn.0.", ".ffn.net_0.proj.")
                elif ".ffn.2." in lookup_name:
                    lookup_name = lookup_name.replace(".ffn.2.", ".ffn.net_2.")
                if ".ffn.net.0." in lookup_name:
                    lookup_name = lookup_name.replace(".ffn.net.0.", ".ffn.net_0.proj.")
                elif ".ffn.net.2." in lookup_name:
                    lookup_name = lookup_name.replace(".ffn.net.2.", ".ffn.net_2.")
                if ".to_out.0." in lookup_name:
                    lookup_name = lookup_name.replace(".to_out.0.", ".to_out.")

                if lookup_name not in params_dict:
                    logger.warning(f"Skipping weight {original_name} -> {lookup_name}")
                    continue

                param = params_dict[lookup_name]

                # Shard RMSNorm weights for TP
                if tp_size > 1 and any(
                    norm_name in lookup_name
                    for norm_name in [
                        ".attn1.norm_q.",
                        ".attn1.norm_k.",
                        ".attn2.norm_q.",
                        ".attn2.norm_k.",
                        ".attn2.norm_added_k.",
                    ]
                ):
                    shard_size = loaded_weight.shape[0] // tp_size
                    loaded_weight = loaded_weight[tp_rank * shard_size : (tp_rank + 1) * shard_size]

                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)

            loaded_params.add(original_name)
            loaded_params.add(lookup_name)

        return loaded_params


# ---------------------------------------------------------------------------
# AdaLN variant (I2W) — action features modulate timestep embeddings
# ---------------------------------------------------------------------------


class AdaLNWanTransformerBlock(WanTransformerBlock):
    """Transformer block with AdaLN action modulation.

    Action features are converted to 6 modulation parameters via a learned
    projection, spatially broadcast, and added to the timestep embedding before
    the standard block processing.
    """

    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        action_dim: int = 1536,
        eps: float = 1e-6,
        added_kv_proj_dim: int | None = None,
        cross_attn_norm: bool = False,
    ):
        super().__init__(dim, ffn_dim, num_heads, eps, added_kv_proj_dim, cross_attn_norm)
        self.action_adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(action_dim, 6 * dim, bias=True),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor],
        hidden_states_mask: torch.Tensor | None = None,
        action_feat: torch.Tensor | None = None,
        grid_sizes: tuple[int, int, int] | None = None,
    ) -> torch.Tensor:
        if action_feat is not None and grid_sizes is not None:
            t, h, w = grid_sizes
            modulation = self.action_adaLN_modulation(action_feat)
            modulation = modulation.view(modulation.shape[0], t, 6, -1)
            modulation = modulation.unsqueeze(2).unsqueeze(3)
            modulation = modulation.expand(-1, -1, h, w, -1, -1)
            modulation = modulation.reshape(modulation.shape[0], t * h * w, 6, -1)
            if temb.ndim == 3:
                temb = temb.unsqueeze(1).expand(-1, t * h * w, -1, -1)
            temb = temb + modulation

        return super().forward(hidden_states, encoder_hidden_states, temb, rotary_emb, hidden_states_mask)


class MicroWorldAdaLNTransformer(WanTransformer3DModel):
    """Wan transformer with AdaLN action modulation for I2W generation."""

    def __init__(
        self,
        action_dim: int = 1536,
        mouse_dim: int = 2,
        keyboard_dim: int = 7,
        **kwargs,
    ):
        super().__init__(**kwargs)

        num_layers = kwargs.get("num_layers", 40)
        inner_dim = kwargs.get("num_attention_heads", 40) * kwargs.get("attention_head_dim", 128)
        ffn_dim = kwargs.get("ffn_dim", 13824)
        num_heads = kwargs.get("num_attention_heads", 40)
        eps = kwargs.get("eps", 1e-6)
        added_kv_proj_dim = kwargs.get("added_kv_proj_dim", None)
        cross_attn_norm = kwargs.get("cross_attn_norm", True)

        # Replace base blocks with AdaLN-aware blocks
        self.blocks = nn.ModuleList(
            [
                AdaLNWanTransformerBlock(
                    dim=inner_dim,
                    ffn_dim=ffn_dim,
                    num_heads=num_heads,
                    action_dim=action_dim,
                    eps=eps,
                    added_kv_proj_dim=added_kv_proj_dim,
                    cross_attn_norm=cross_attn_norm,
                )
                for _ in range(num_layers)
            ]
        )

        # Action preprocessor (no spatial flatten for AdaLN)
        self.action_preprocess = ActionModule(
            mouse_dim=mouse_dim,
            keyboard_dim=keyboard_dim,
            action_dim=action_dim,
            window_size=3,
            temporal_ratio=4,
            flatten_spatial=False,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: torch.Tensor | None = None,
        mouse_actions: torch.Tensor | None = None,
        keyboard_actions: torch.Tensor | None = None,
        return_dict: bool = True,
        attention_kwargs: dict[str, Any] | None = None,
    ) -> torch.Tensor | Transformer2DModelOutput:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w
        grid_sizes = (post_patch_num_frames, post_patch_height, post_patch_width)

        # WanRotaryPosEmbed emits full head_dim via repeat_interleave(2); the
        # pair-rotation kernel expects half head_dim, so take the even slice.
        freqs_cos, freqs_sin = self.rope(hidden_states)
        rotary_emb = (
            freqs_cos[..., 0::2].to(dtype=hidden_states.dtype),
            freqs_sin[..., 0::2].to(dtype=hidden_states.dtype),
        )

        # Patch embedding and flatten to sequence
        hidden_states = self.patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        # Handle timestep shape
        if timestep.ndim == 2:
            ts_seq_len = timestep.shape[1]
            timestep = timestep.flatten()
        else:
            ts_seq_len = None

        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
            timestep, encoder_hidden_states, encoder_hidden_states_image, timestep_seq_len=ts_seq_len
        )
        timestep_proj = self.timestep_proj_prepare(timestep_proj, ts_seq_len)

        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

        # SP attention mask
        hidden_states_mask = None
        config = get_forward_context().omni_diffusion_config
        parallel_config = config.parallel_config
        if parallel_config is not None and parallel_config.sequence_parallel_size > 1:
            ctx = get_forward_context()
            if ctx.sp_original_seq_len is not None and ctx.sp_padding_size > 0:
                padded_seq_len = ctx.sp_original_seq_len + ctx.sp_padding_size
                hidden_states_mask = torch.ones(
                    batch_size, padded_seq_len, dtype=torch.bool, device=hidden_states.device
                )
                hidden_states_mask[:, ctx.sp_original_seq_len :] = False
        if hidden_states_mask is not None and hidden_states_mask.all():
            hidden_states_mask = None

        # Compute action features
        action_feat = None
        if mouse_actions is not None and keyboard_actions is not None:
            action_feat = self.action_preprocess(mouse_actions, keyboard_actions, grid_sizes)

        # Transformer blocks with AdaLN action modulation
        for block in self.blocks:
            hidden_states = block(
                hidden_states,
                encoder_hidden_states,
                timestep_proj,
                rotary_emb,
                hidden_states_mask,
                action_feat=action_feat,
                grid_sizes=grid_sizes,
            )

        # Output norm, projection & unpatchify
        shift, scale = self.output_scale_shift_prepare(temb)
        shift = shift.to(hidden_states.device)
        scale = scale.to(hidden_states.device)
        if shift.ndim == 2:
            shift = shift.unsqueeze(1)
            scale = scale.unsqueeze(1)

        hidden_states = self.norm_out(hidden_states, scale, shift).type_as(hidden_states)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = hidden_states.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights with Wan2.1-to-vllm name mapping including AdaLN layers."""
        tp_rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()

        stacked_params_mapping = [
            (".attn1.to_qkv", ".attn1.to_q", "q"),
            (".attn1.to_qkv", ".attn1.to_k", "k"),
            (".attn1.to_qkv", ".attn1.to_v", "v"),
        ]
        self.stacked_params_mapping = stacked_params_mapping

        weight_name_remapping = {
            # Top-level Wan2.1 → vllm-omni rename. Note: do NOT add a bare
            # ``scale_shift_table`` substring rule here — every per-block
            # ``blocks.N.scale_shift_table`` would also get clobbered.
            "head.head.": "proj_out.",
            "head.modulation": "output_scale_shift_prepare.scale_shift_table",
            "time_embedding.0.": "condition_embedder.time_embedder.linear_1.",
            "time_embedding.2.": "condition_embedder.time_embedder.linear_2.",
            "time_projection.1.": "condition_embedder.time_proj.",
            "text_embedding.0.": "condition_embedder.text_embedder.linear_1.",
            "text_embedding.2.": "condition_embedder.text_embedder.linear_2.",
            # Image embedder (I2V): reference Wan ``MLPProj`` is a Sequential
            # of [LayerNorm, Linear, GELU, Linear, LayerNorm]; vllm-omni uses
            # ``norm1`` + diffusers ``FeedForward`` + ``norm2``.
            "img_emb.proj.0.": "condition_embedder.image_embedder.norm1.",
            "img_emb.proj.1.": "condition_embedder.image_embedder.ff.net.0.proj.",
            "img_emb.proj.3.": "condition_embedder.image_embedder.ff.net.2.",
            "img_emb.proj.4.": "condition_embedder.image_embedder.norm2.",
        }

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            original_name = name

            # Wan2.1 → vllm-omni name remapping
            name = name.replace(".self_attn.q.", ".attn1.to_q.")
            name = name.replace(".self_attn.k.", ".attn1.to_k.")
            name = name.replace(".self_attn.v.", ".attn1.to_v.")
            name = name.replace(".self_attn.o.", ".attn1.to_out.")
            name = name.replace(".self_attn.norm_q.", ".attn1.norm_q.")
            name = name.replace(".self_attn.norm_k.", ".attn1.norm_k.")
            name = name.replace(".cross_attn.q.", ".attn2.to_q.")
            name = name.replace(".cross_attn.k.", ".attn2.to_k.")
            name = name.replace(".cross_attn.v.", ".attn2.to_v.")
            name = name.replace(".cross_attn.o.", ".attn2.to_out.")
            name = name.replace(".cross_attn.norm_q.", ".attn2.norm_q.")
            name = name.replace(".cross_attn.norm_k.", ".attn2.norm_k.")
            # I2V image cross-attn branch uses add_k_proj/add_v_proj in vllm-omni.
            name = name.replace(".cross_attn.k_img.", ".attn2.add_k_proj.")
            name = name.replace(".cross_attn.v_img.", ".attn2.add_v_proj.")
            name = name.replace(".cross_attn.norm_k_img.", ".attn2.norm_added_k.")

            # Top-level remaps must run before the per-block modulation fallback.
            for old, new in weight_name_remapping.items():
                name = name.replace(old, new)

            if ".modulation." in name and "action_preprocess" not in name and "action_adaLN" not in name:
                name = name.replace(".modulation.", ".scale_shift_table.")
            if name.endswith(".modulation") and "action_preprocess" not in name and "action_adaLN" not in name:
                name = name[: -len(".modulation")] + ".scale_shift_table"
            name = name.replace(".norm3.", ".norm2.")

            lookup_name = name

            # Handle QKV fusion
            fused = False
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                lookup_name = name.replace(weight_name, param_name)
                if lookup_name not in params_dict:
                    continue
                param = params_dict[lookup_name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                fused = True
                break

            if not fused:
                if ".ffn.0." in lookup_name:
                    lookup_name = lookup_name.replace(".ffn.0.", ".ffn.net_0.proj.")
                elif ".ffn.2." in lookup_name:
                    lookup_name = lookup_name.replace(".ffn.2.", ".ffn.net_2.")
                if ".ffn.net.0." in lookup_name:
                    lookup_name = lookup_name.replace(".ffn.net.0.", ".ffn.net_0.proj.")
                elif ".ffn.net.2." in lookup_name:
                    lookup_name = lookup_name.replace(".ffn.net.2.", ".ffn.net_2.")
                if ".to_out.0." in lookup_name:
                    lookup_name = lookup_name.replace(".to_out.0.", ".to_out.")

                if lookup_name not in params_dict:
                    logger.warning(f"Skipping weight {original_name} -> {lookup_name}")
                    continue

                param = params_dict[lookup_name]

                if tp_size > 1 and any(
                    norm_name in lookup_name
                    for norm_name in [
                        ".attn1.norm_q.",
                        ".attn1.norm_k.",
                        ".attn2.norm_q.",
                        ".attn2.norm_k.",
                        ".attn2.norm_added_k.",
                    ]
                ):
                    shard_size = loaded_weight.shape[0] // tp_size
                    loaded_weight = loaded_weight[tp_rank * shard_size : (tp_rank + 1) * shard_size]

                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)

            loaded_params.add(original_name)
            loaded_params.add(lookup_name)

        return loaded_params
