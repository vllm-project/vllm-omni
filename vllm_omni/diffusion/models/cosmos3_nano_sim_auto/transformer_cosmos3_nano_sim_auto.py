# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Causal rolling multiview Cosmos3 transformer for Cosmos3-Nano-Sim-Auto.

The weights are the Cosmos3 MoT generator (plus LiDAR projections for joint
checkpoints). What changes is the GEN attention: every current token attends
one softmax over the keys the replay predicate admits from three streams --
the current chunk, the KV ring of committed chunks and the per-camera caption
documents -- executed as log-sum-exp-merged maskless passes planned on the host
(:mod:`.causal_plan`). Clean refresh forwards additionally write their K/V into
the paged pool slots the pipeline allocated for the chunk.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from vllm.distributed import get_tensor_model_parallel_world_size

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.models.cosmos3.multiview_maskless_attention import maskless_attention_streams
from vllm_omni.diffusion.models.cosmos3.multiview_packing import patchify_sensor, unpatchify_sensor
from vllm_omni.diffusion.models.cosmos3.transformer_cosmos3 import (
    Cosmos3CrossAttention,
    Cosmos3GenDecoderLayer,
    Cosmos3VFMTransformer,
    _apply_rotary_pos_emb,
    _tf_config_get,
    compute_mrope_position_ids_vision,
)

from .config import Cosmos3NanoSimAutoManifest
from .layout import StreamItem


@dataclass(frozen=True)
class CausalMasklessRuntime:
    """Flattened causal plan plus the worker's pass kernel, as the custom op consumes it."""

    plan: list[torch.Tensor]
    num_passes: int
    kernel: str


@dataclass
class Cosmos3NanoSimAutoTransformerOutput:
    #: Velocity (or clean) predictions of the target items, in item order, each ``[1, C, V*T, H, W]``.
    targets: list[torch.Tensor]
    #: Current-chunk K/V per layer (post-norm, post-RoPE K; raw V) when collected for the dense oracle.
    current_kv: list[tuple[torch.Tensor, torch.Tensor]]


class Cosmos3NanoSimAutoJointAttention(Cosmos3CrossAttention):
    """One softmax over [current | ring history | captions] for every current GEN token."""

    def forward(  # type: ignore[override]
        self,
        hidden_states: torch.Tensor,
        *,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        und_k: torch.Tensor,
        und_v: torch.Tensor,
        hist_k: torch.Tensor,
        hist_v: torch.Tensor,
        runtime: CausalMasklessRuntime,
        write_slots: torch.Tensor | None = None,
        key_pool: torch.Tensor | None = None,
        value_pool: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if hidden_states.shape[0] != 1:
            raise ValueError(f"Cosmos3-Nano-Sim-Auto attention supports batch_size=1, got {hidden_states.shape[0]}.")
        if und_k.shape != und_v.shape or und_k.ndim != 4 or und_k.shape[0] != 1:
            raise ValueError("Cosmos3-Nano-Sim-Auto caption K/V must be matching [1, U, H_kv, D] tensors.")
        if hist_k.shape != hist_v.shape or hist_k.ndim != 3:
            raise ValueError("Cosmos3-Nano-Sim-Auto history K/V must be matching [tokens, H_kv, D] tensors.")
        batch, seq_len, _ = hidden_states.shape
        q = self.to_q(hidden_states).view(batch, seq_len, self.num_heads_local, self.head_dim)
        k = self.to_k(hidden_states).view(batch, seq_len, self.num_kv_heads_local, self.head_dim)
        v = self.to_v(hidden_states).view(batch, seq_len, self.num_kv_heads_local, self.head_dim)
        if self.qk_norm:
            q = F.rms_norm(q, (self.head_dim,), self.norm_q.weight, eps=self.norm_q.variance_epsilon)
            k = F.rms_norm(k, (self.head_dim,), self.norm_k.weight, eps=self.norm_k.variance_epsilon)
        q, k = _apply_rotary_pos_emb(q, k, freqs_cos, freqs_sin)
        if write_slots is not None:
            # Clean refresh: publish this chunk's K/V (post-norm/RoPE K, raw V)
            # into the pool pages the pipeline allocated. Padding slots of the
            # frame unit are zero-filled by the pipeline and never read.
            if key_pool is None or value_pool is None:
                raise ValueError("Committing forwards need the layer's flat key/value pools.")
            if write_slots.numel() != seq_len:
                raise ValueError(f"write_slots has {write_slots.numel()} entries for {seq_len} current tokens.")
            key_pool.index_copy_(0, write_slots, k[0].to(key_pool.dtype))
            value_pool.index_copy_(0, write_slots, v[0].to(value_pool.dtype))
        out = maskless_attention_streams(
            q,
            [k[0], hist_k.to(k.dtype), und_k[0].to(k.dtype)],
            [v[0], hist_v.to(v.dtype), und_v[0].to(v.dtype)],
            runtime.plan,
            runtime.num_passes,
            runtime.kernel,
        )
        return self.to_out(out.reshape(batch, seq_len, -1)), k, v


class Cosmos3NanoSimAutoGenDecoderLayer(Cosmos3GenDecoderLayer):
    """Cosmos3 GEN layer whose attention reads the causal ring; weight names are unchanged."""

    def __init__(
        self,
        *,
        layer_idx: int | None = None,
        hidden_size: int,
        intermediate_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        rms_norm_eps: float,
        quant_config=None,
        mlp_cls,
        cross_attention_cls=None,
        qk_norm: bool = True,
        prefix: str = "",
    ) -> None:
        del cross_attention_cls  # replaced below; the base default is constructed and discarded
        super().__init__(
            layer_idx=layer_idx,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            rms_norm_eps=rms_norm_eps,
            quant_config=quant_config,
            mlp_cls=mlp_cls,
            qk_norm=qk_norm,
            prefix=prefix,
        )
        self.cross_attention = Cosmos3NanoSimAutoJointAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            rms_norm_eps=rms_norm_eps,
            quant_config=quant_config,
            qk_norm=qk_norm,
            prefix=f"{prefix}.cross_attention",
        )

    def forward(self, hidden_states: torch.Tensor, **attention_kwargs: Any):  # type: ignore[override]
        residual = hidden_states
        attention_output, current_k, current_v = self.cross_attention(
            self.input_layernorm(hidden_states), **attention_kwargs
        )
        hidden_states = residual + attention_output
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, current_k, current_v


class Cosmos3NanoSimAutoTransformer(Cosmos3VFMTransformer):
    """Cosmos3 MoT generator with a chunk-causal multiview KV ring and optional LiDAR stream."""

    _hsdp_forward_methods = ("encode_und_kv",)
    _inductor_cudagraphs = False
    _gen_layer_cls = Cosmos3NanoSimAutoGenDecoderLayer
    _repeated_blocks = ["Cosmos3NanoSimAutoGenDecoderLayer"]

    def _language_model_kwargs(self) -> dict[str, Any]:
        return {"use_und_k_norm_for_gen": bool(self.use_und_k_norm_for_gen)}

    @staticmethod
    def _validate_supported_config(model_config: Any) -> None:
        expected_values = {
            "qk_norm_for_diffusion": True,
            "qk_norm_for_text": True,
            "position_embedding_type": "unified_3d_mrope",
            "unified_3d_mrope_reset_spatial_ids": True,
            "joint_attn_implementation": "multiview",
            "video_temporal_causal": True,
        }
        for key, expected in expected_values.items():
            actual = _tf_config_get(model_config, key, expected)
            if actual != expected:
                raise ValueError(
                    f"Unsupported Cosmos3-Nano-Sim-Auto transformer config: {key}={actual!r}; expected {expected!r}."
                )

    def validate_loaded_weights(self, loaded: set[str]) -> None:
        loaded = {name.removeprefix("transformer.") for name in loaded}
        if any(name.endswith("rig_view_embed.weight") for name in loaded):
            raise ValueError(
                "Cosmos3-Nano-Sim-Auto checkpoints have no rig view embedding; this artifact belongs to the "
                "bidirectional Cosmos3-Nano-Transfer-Auto pipeline."
            )
        required = {name for name, _ in self.named_parameters()}
        missing = sorted(f"transformer.{name}" for name in required - loaded)
        if missing:
            preview = ", ".join(missing[:12])
            suffix = "" if len(missing) <= 12 else f" (and {len(missing) - 12} more)"
            raise ValueError(
                f"Cosmos3-Nano-Sim-Auto checkpoint is missing required transformer weights: {preview}{suffix}"
            )

    def __init__(
        self,
        od_config: OmniDiffusionConfig,
        *,
        temporal_compression_factor: int | None = None,
        sound_gen: bool = False,
        sound_dim: int | None = None,
        sound_latent_fps: float | None = None,
    ) -> None:
        self.manifest = Cosmos3NanoSimAutoManifest.from_od_config(od_config)
        if sound_gen:
            raise ValueError("Cosmos3-Nano-Sim-Auto does not support joint sound generation.")
        super().__init__(
            od_config,
            temporal_compression_factor=temporal_compression_factor or self.manifest.temporal_compression_factor,
            sound_gen=False,
            sound_dim=sound_dim,
            sound_latent_fps=sound_latent_fps,
        )
        if self.action_gen:
            raise ValueError("Cosmos3-Nano-Sim-Auto checkpoints carry no action pathway; action_gen must be off.")
        if self.temporal_compression_factor != self.manifest.temporal_compression_factor:
            raise ValueError(
                "Cosmos3-Nano-Sim-Auto temporal compression differs between VAE and manifest: "
                f"{self.temporal_compression_factor} != {self.manifest.temporal_compression_factor}."
            )
        if (
            self.latent_patch_size != self.manifest.latent_patch_size
            or self.latent_channel_size != self.manifest.latent_channels
        ):
            raise ValueError("Cosmos3-Nano-Sim-Auto transformer patch/latent geometry disagrees with the manifest.")
        self.temporal_modality_margin = self.manifest.temporal_modality_margin
        self.base_fps = self.manifest.base_fps
        self.enable_fps_modulation = self.manifest.enable_fps_modulation
        self.lidar_patch_hw: tuple[int, int] | None = None
        if self.manifest.lidar is not None:
            assert self.manifest.lidar_latent_patch_size_hw is not None
            self.lidar_patch_hw = self.manifest.lidar_latent_patch_size_hw
            width = self.lidar_patch_hw[0] * self.lidar_patch_hw[1] * int(self.manifest.lidar["latent_channels"])
            self.lidar_proj_in = torch.nn.Linear(width, self.hidden_size)
            self.lidar_proj_out = torch.nn.Linear(self.hidden_size, width)

    @property
    def num_kv_heads_local(self) -> int:
        return self.num_key_value_heads // get_tensor_model_parallel_world_size()

    @staticmethod
    def pad_text_kv(
        layer_kv: list[tuple[torch.Tensor, torch.Tensor]], *, max_len: int
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        padded: list[tuple[torch.Tensor, torch.Tensor]] = []
        for key, value in layer_kv:
            if key.shape != value.shape or key.shape[0] != 1:
                raise ValueError(f"Caption K/V must be matching batch-one tensors, got {key.shape} and {value.shape}.")
            if key.shape[1] > max_len:
                raise ValueError(f"Caption K/V has {key.shape[1]} tokens but text_cache_max_len={max_len}.")
            pad_len = max_len - key.shape[1]
            if pad_len:
                key = torch.cat([key, key.new_zeros(1, pad_len, *key.shape[2:])], dim=1)
                value = torch.cat([value, value.new_zeros(1, pad_len, *value.shape[2:])], dim=1)
            padded.append((key, value))
        return padded

    # -- geometry helpers ----------------------------------------------------

    def _item_patch(self, item: StreamItem) -> tuple[int, int]:
        if item.is_lidar:
            if self.lidar_patch_hw is None:
                raise ValueError("Joint requests require a checkpoint with LiDAR projections.")
            return self.lidar_patch_hw
        return (self.latent_patch_size, self.latent_patch_size)

    def _item_fps_and_compression(self, item: StreamItem) -> tuple[float, int]:
        if item.is_lidar:
            assert self.manifest.lidar is not None
            compression = int(self.manifest.lidar["temporal_compression_factor"])
        else:
            compression = self.temporal_compression_factor
        return compression / item.seconds_per_frame, compression

    def chunk_position_ids(self, items: Sequence[StreamItem], *, text_extent: int) -> torch.Tensor:
        """Absolute 3D mRoPE ids of the packed chunk, ``[3, S]``.

        Text documents are rewound to ``0..len-1``, so media starts at
        ``text_extent + temporal_modality_margin`` (``text_extent`` = longest framed
        caption). RGB latent ``f`` sits at ``+f`` and LiDAR sweep ``s`` at ``+0.75 s``
        through FPS modulation with the camera compression as the base (4 at 30 fps);
        camera-major items repeat the same temporal ids for every view, and
        spatial ids restart at zero per frame.
        """
        offset = text_extent + self.temporal_modality_margin
        parts = []
        for item in items:
            frames = item.frames
            if any(later != earlier + 1 for earlier, later in zip(frames[:-1], frames[1:])):
                raise ValueError(f"Chunk items must hold contiguous frames, got {frames}.")
            fps, compression = self._item_fps_and_compression(item)
            positions, _ = compute_mrope_position_ids_vision(
                grid_t=len(frames),
                grid_h=item.grid.grid_height,
                grid_w=item.grid.grid_width,
                temporal_offset=offset,
                fps=fps,
                base_fps=self.base_fps,
                temporal_compression_factor=compression,
                base_temporal_compression_factor=self.temporal_compression_factor,
                enable_fps_modulation=self.enable_fps_modulation,
                start_frame_offset=frames[0],
            )
            parts.append(positions.repeat(1, len(item.view_ids)))
        return torch.cat(parts, dim=1)

    def chunk_rope(
        self, items: Sequence[StreamItem], *, text_extent: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        positions = self.chunk_position_ids(items, text_extent=text_extent).unsqueeze(1).to(device)
        dummy = torch.empty(0, device=device, dtype=dtype)
        cos, sin = self.language_model.rotary_emb(dummy, position_ids=positions)
        return cos.unsqueeze(2), sin.unsqueeze(2)

    # -- embedding -------------------------------------------------------

    def embed_items(
        self,
        items: Sequence[StreamItem],
        latents: Sequence[torch.Tensor],
        *,
        timestep: torch.Tensor | None,
    ) -> torch.Tensor:
        """Project every item's latents and add the timestep to generated target frames only.

        ``timestep`` is ``None`` for clean refresh forwards: controls, observed
        condition frames and committed-clean targets never receive a time embed.
        """
        if len(items) != len(latents):
            raise ValueError(f"{len(items)} items but {len(latents)} latent tensors.")
        time_embedding = None
        if timestep is not None:
            time_embedding = self._embed_timestep(timestep.reshape(1), self.proj_in.weight.dtype).view(1, 1, -1)
        embeddings = []
        for item, latent in zip(items, latents, strict=True):
            expected_frames = len(item.view_ids) * len(item.frames)
            if latent.ndim != 5 or latent.shape[0] != 1 or latent.shape[2] != expected_frames:
                raise ValueError(
                    f"Item {item.kind} latents must be [1, C, {expected_frames}, H, W] (camera-major), "
                    f"got {tuple(latent.shape)}."
                )
            project = self.lidar_proj_in if item.is_lidar else self.proj_in
            hidden = project(patchify_sensor(latent.to(self.proj_in.weight.dtype), self._item_patch(item)))
            if time_embedding is not None and not item.is_control:
                generated = torch.tensor(
                    [frame not in item.condition_frames for frame in item.frames],
                    dtype=torch.bool,
                    device=hidden.device,
                )
                mask = generated.repeat(len(item.view_ids)).repeat_interleave(item.tokens_per_frame)
                hidden = hidden + time_embedding.to(hidden.dtype) * mask.view(1, -1, 1).to(hidden.dtype)
            embeddings.append(hidden)
        return torch.cat(embeddings, dim=1)

    # -- forward ----------------------------------------------------------

    def forward(  # type: ignore[override]
        self,
        *,
        items: Sequence[StreamItem],
        latents: Sequence[torch.Tensor],
        timestep: torch.Tensor | None,
        text_extent: int,
        und_kv: list[tuple[torch.Tensor, torch.Tensor]],
        history_kv: list[tuple[torch.Tensor, torch.Tensor]] | None,
        runtime: CausalMasklessRuntime,
        write_slots: torch.Tensor | None = None,
        kv_pools: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        collect_kv: bool = False,
    ) -> Cosmos3NanoSimAutoTransformerOutput:
        """Denoise (``timestep`` set) or clean-refresh (``timestep`` None) one chunk.

        ``und_kv`` holds the per-layer compact caption K/V ``[1, U, H_kv, D]``;
        ``history_kv`` the per-layer ring keys ``[tokens, H_kv, D]`` (flat pool
        views or the dense ring), or ``None`` for chunk 0. ``write_slots`` with
        ``kv_pools`` publishes the refresh K/V into the paged pool.
        """
        if self._inductor_cudagraphs:
            torch.compiler.cudagraph_mark_step_begin()
        if len(und_kv) != self.num_hidden_layers:
            raise ValueError(f"Expected {self.num_hidden_layers} caption KV layers, got {len(und_kv)}.")
        if history_kv is not None and len(history_kv) != self.num_hidden_layers:
            raise ValueError(f"Expected {self.num_hidden_layers} history KV layers, got {len(history_kv)}.")
        if (write_slots is None) != (kv_pools is None):
            raise ValueError("write_slots and kv_pools must be given together.")
        if kv_pools is not None and len(kv_pools) != self.num_hidden_layers:
            raise ValueError(f"Expected {self.num_hidden_layers} KV pools, got {len(kv_pools)}.")
        if timestep is not None and timestep.numel() != 1:
            raise ValueError(f"timestep must hold one value, got {tuple(timestep.shape)}.")

        hidden = self.embed_items(items, latents, timestep=timestep)
        freqs_cos, freqs_sin = self.chunk_rope(items, text_extent=text_extent, device=hidden.device, dtype=hidden.dtype)
        empty_history = und_kv[0][0].new_empty(0, self.num_kv_heads_local, self.head_dim)
        current_kv: list[tuple[torch.Tensor, torch.Tensor]] = []
        with self._offload_context("generator"):
            for layer_idx, layer in enumerate(self.gen_layers):
                und_k, und_v = und_kv[layer_idx]
                hist_k, hist_v = history_kv[layer_idx] if history_kv is not None else (empty_history, empty_history)
                key_pool, value_pool = kv_pools[layer_idx] if kv_pools is not None else (None, None)
                hidden, k, v = layer(
                    hidden,
                    freqs_cos=freqs_cos,
                    freqs_sin=freqs_sin,
                    und_k=und_k,
                    und_v=und_v,
                    hist_k=hist_k,
                    hist_v=hist_v,
                    runtime=runtime,
                    write_slots=write_slots,
                    key_pool=key_pool,
                    value_pool=value_pool,
                )
                if collect_kv:
                    current_kv.append((k, v))
            hidden = self.norm_moe_gen(hidden)
            outputs = []
            for item, latent, part in zip(
                items, latents, hidden.split([item.num_tokens for item in items], dim=1), strict=True
            ):
                if item.is_control:
                    continue
                projected = (self.lidar_proj_out if item.is_lidar else self.proj_out)(part)
                outputs.append(unpatchify_sensor(projected, tuple(latent.shape[1:]), self._item_patch(item)))
        return Cosmos3NanoSimAutoTransformerOutput(targets=outputs, current_kv=current_kv)


__all__ = [
    "CausalMasklessRuntime",
    "Cosmos3NanoSimAutoGenDecoderLayer",
    "Cosmos3NanoSimAutoJointAttention",
    "Cosmos3NanoSimAutoTransformer",
    "Cosmos3NanoSimAutoTransformerOutput",
]
