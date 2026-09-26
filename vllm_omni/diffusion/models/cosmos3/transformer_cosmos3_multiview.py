# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cosmos3 transformer variant with sparse and maskless multiview attention."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import torch
from torch import nn
from vllm.distributed import tensor_model_parallel_all_reduce

from .multiview_attention import multiview_attention
from .multiview_flex_attention import (
    MaskItem,
    MultiviewAttentionContext,
    MultiviewLayout,
)
from .multiview_maskless_attention import build_maskless_plan, load_maskless_runtime, make_merge_scratch
from .multiview_packing import pack_state, packed_position_ids, patchify_sensor, unpack_state, unpatchify_sensor
from .multiview_parallel import multiview_ulysses_attention
from .transformer_cosmos3 import (
    COSMOS3_MULTIVIEW_BACKBONE_TYPE,
    Cosmos3CrossAttention,
    Cosmos3GenDecoderLayer,
    Cosmos3VFMTransformer,
    _get_ulysses_state,
    _is_sp_active,
    _tf_config_get,
)


class Cosmos3MultiviewCrossAttention(Cosmos3CrossAttention):
    """Dispatch the checkpoint's attention semantics through a model-local context."""

    def _forward_multiview(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        k_und: torch.Tensor,
        v_und: torch.Tensor,
        multiview_layout: Any,
    ) -> torch.Tensor:
        if not isinstance(multiview_layout, MultiviewAttentionContext):
            raise TypeError(
                "Cosmos3 multiview cross-attention expected MultiviewAttentionContext, "
                f"got {type(multiview_layout).__name__}."
            )
        if _is_sp_active():
            size, rank, group = _get_ulysses_state()
            if group is None:
                raise RuntimeError("Cosmos3 multiview CP is active without an initialized Ulysses group.")
            output = multiview_ulysses_attention(
                q, k, v, k_und, v_und, multiview_layout, group=group, rank=rank, world_size=size
            )
        else:
            output = multiview_attention(q, k, v, k_und, v_und, multiview_layout)
        return output.reshape(q.shape[0], q.shape[1], -1)


class Cosmos3MultiviewGenDecoderLayer(Cosmos3GenDecoderLayer):
    """Bound post-attention norm/MLP activations for long multiview sequences."""

    # Bound the total token rows across the batch in each norm/MLP call.
    _mlp_chunk_size = 65536

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Assemble local chunk outputs before doing a single TP reduction.
        self.mlp.down_proj.reduce_results = False

    def _forward_mlp_chunk(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.post_attention_layernorm(hidden_states))

    def _add_residual(self, output: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        # Attention and MLP projections produce fresh outputs. Reuse those
        # buffers, preserving the caller's input and the normal autograd path.
        if not torch.is_grad_enabled() and output.dtype == residual.dtype:
            return output.add_(residual)
        return super()._add_residual(output, residual)

    @torch.compiler.disable(recursive=False)
    def _forward_mlp_chunked(self, hidden_states: torch.Tensor, chunk_size: int) -> torch.Tensor:
        # Keep the loop out of the graph so compilation cannot unroll it and
        # retain intermediates across chunks. Child calls may still compile.
        batch, sequence_length, hidden_size = hidden_states.shape
        # Flatten before the compiled call so its strides do not depend on the
        # full sequence length. Any copy for a batched slice is chunk-sized.
        chunk = self._forward_mlp_chunk(hidden_states[:, :chunk_size].reshape(-1, hidden_size))
        output = chunk.new_empty(batch, sequence_length, hidden_size)
        output[:, :chunk_size] = chunk.view(batch, chunk_size, hidden_size)
        del chunk
        for start in range(chunk_size, sequence_length, chunk_size):
            end = min(start + chunk_size, sequence_length)
            output[:, start:end] = self._forward_mlp_chunk(hidden_states[:, start:end].reshape(-1, hidden_size)).view(
                batch, end - start, hidden_size
            )
        return output

    def _forward_mlp(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch, sequence_length, _ = hidden_states.shape
        chunk_size = max(1, self._mlp_chunk_size // batch)
        if sequence_length <= chunk_size:
            output = self._forward_mlp_chunk(hidden_states)
        else:
            output = self._forward_mlp_chunked(hidden_states, chunk_size)
        if self.mlp.down_proj.tp_size > 1:
            output = tensor_model_parallel_all_reduce(output)
        return output


class Cosmos3MultiviewVFMTransformer(Cosmos3VFMTransformer):
    """Cosmos3 Nano weights with request-local multiview block-mask caching."""

    _gen_layer_cls = Cosmos3MultiviewGenDecoderLayer
    _repeated_blocks = ["Cosmos3MultiviewGenDecoderLayer"]
    _cross_attention_cls = Cosmos3MultiviewCrossAttention

    # At hidden size 4096, each FP32 RMSNorm temporary is at most 128 MiB
    # for the single-sample video requests served by this pipeline.
    _output_projection_chunk_size = 8192

    def _project_video_tokens(self, hidden_video: torch.Tensor) -> torch.Tensor:
        """Normalize/project video tokens without full-sequence FP32 temporaries.

        Keep the existing RMSNorm arithmetic, including its FP32 intermediates.
        Only the smaller projected latent tokens are retained between chunks;
        collecting normalized chunks would recreate the large hidden tensor.
        """
        batch, sequence_length, _ = hidden_video.shape
        chunk_size = max(1, self._output_projection_chunk_size // batch)
        projected = self.proj_out(self.norm_moe_gen(hidden_video[:, :chunk_size]))
        if sequence_length <= chunk_size:
            return projected

        # Allocate from the projection result to preserve its dtype under autocast.
        output = projected.new_empty(batch, sequence_length, projected.shape[-1])
        output[:, :chunk_size] = projected
        for start in range(chunk_size, sequence_length, chunk_size):
            end = min(start + chunk_size, sequence_length)
            output[:, start:end] = self.proj_out(self.norm_moe_gen(hidden_video[:, start:end]))
        return output

    @staticmethod
    def _validate_supported_config(model_config: Any) -> None:
        Cosmos3VFMTransformer._validate_supported_config(model_config)
        backbone_type = _tf_config_get(model_config, "backbone_type", None)
        if backbone_type != COSMOS3_MULTIVIEW_BACKBONE_TYPE:
            raise ValueError(
                "Cosmos3MultiviewVFMTransformer requires transformer/config.json "
                f"backbone_type={COSMOS3_MULTIVIEW_BACKBONE_TYPE!r}, got {backbone_type!r}."
            )

    def __init__(self, *args, **kwargs) -> None:
        od_config = kwargs.get("od_config", args[0] if args else None)
        deployment = _tf_config_get(od_config.tf_model_config, "multiview", {})
        self._maskless_fa_version = (
            load_maskless_runtime() if _tf_config_get(deployment, "backend", "triton") == "maskless" else 0
        )
        self._maskless_gqa_ratio = _tf_config_get(
            od_config.tf_model_config, "num_attention_heads", 32
        ) // _tf_config_get(od_config.tf_model_config, "num_key_value_heads", 8)
        super().__init__(*args, **kwargs)
        self.lidar_config = _tf_config_get(deployment, "lidar", None)
        if self.lidar_config is not None:
            from .lidar import validate_lidar_config

            self.lidar_config = dict(self.lidar_config)
            validate_lidar_config(self.lidar_config)
            width = self.latent_patch_size**2 * self.lidar_config["latent_channels"]
            self.lidar_proj_in = nn.Linear(width, self.hidden_size)
            self.lidar_proj_out = nn.Linear(self.hidden_size, width)
        self._multiview_mask_cache: dict[tuple[Any, ...], Any] = {}
        # Padded q/k/v packing buffers, keyed by shape/dtype/device. Held on the
        # transformer rather than the per-forward context so the ~2.5 GiB of
        # packed tensors are zeroed once per request instead of once per layer.
        self._multiview_buffer_cache: dict[tuple[Any, ...], torch.Tensor] = {}

    def reset_cache(self) -> None:
        super().reset_cache()
        self._multiview_mask_cache.clear()
        self._multiview_buffer_cache.clear()

    def validate_loaded_weights(self, loaded: set[str]) -> None:
        super().validate_loaded_weights(loaded)
        if self.lidar_config is not None:
            required = {
                f"lidar_proj_{direction}.{parameter}" for direction in ("in", "out") for parameter in ("weight", "bias")
            }
            missing = [name for name in sorted(required) if not any(key.endswith(name) for key in loaded)]
            if missing:
                raise ValueError(f"Incomplete joint checkpoint: missing LiDAR projection weights {missing}.")

    def _embed_packed_streams(
        self,
        items: tuple[MaskItem, ...],
        streams: list[torch.Tensor],
        timestep: torch.Tensor,
        camera: torch.Tensor,
        noisy_frame_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        # Both targets get the same timestep. Controls never receive it;
        # camera condition frames receive zero.
        time = self._embed_timestep(timestep, camera.dtype).unsqueeze(1)
        embeddings = []
        for item, latent in zip(items, streams, strict=True):
            project = self.lidar_proj_in if item.is_lidar else self.proj_in
            hidden = project(patchify_sensor(latent.to(camera), self.latent_patch_size))
            if not item.is_control:
                # Projection outputs are fresh, unaliased tensors, so the
                # timestep update can mutate them in place.
                stream_time = time.to(hidden)
                if not item.is_lidar and noisy_frame_mask is not None:
                    mask = (
                        noisy_frame_mask[:, 0, :, 0, 0]
                        .repeat_interleave(item.token_shape[1] * item.token_shape[2], dim=1)
                        .unsqueeze(-1)
                        .to(hidden)
                    )
                    # addcmul_ also avoids materializing the broadcast
                    # ``stream_time * mask`` tensor.
                    hidden.addcmul_(stream_time, mask)
                else:
                    hidden.add_(stream_time)
            embeddings.append(hidden)
        return torch.cat(embeddings, dim=1)

    def _forward_packed(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        text_ids: torch.Tensor,
        text_mask: torch.Tensor,
        multiview_layout: MultiviewLayout,
        packed_shapes: tuple[tuple[int, ...], ...],
        caption_lengths: tuple[int, ...],
        control_latents=None,
        lidar_control_latents: torch.Tensor | None = None,
        noisy_frame_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Pack sensor-specific embeddings around the base transformer's shared GEN execution.

        Captions arrive compacted from the pipeline with host-side lengths;
        no text-mask reduction or device-to-host synchronization is needed here.
        """
        if kwargs.get("action_latents") is not None or kwargs.get("sound_latents") is not None:
            raise ValueError("Multiview generation cannot be combined with action or sound.")
        if multiview_layout.backend == "maskless" and hidden_states.shape[0] != 1:
            raise ValueError("Maskless multiview transformer requires B == 1 (including each CFG branch).")
        targets = unpack_state(hidden_states, packed_shapes)
        camera = targets[0]
        has_control = control_latents is not None and len(control_latents) > 0
        items = tuple(item for item in multiview_layout.items if has_control or not item.is_control)
        lengths = caption_lengths
        if not lengths or any(length <= 0 for length in lengths) or sum(lengths) != text_ids.shape[1]:
            raise ValueError("Packed caption lengths must cover the compacted text tokens with nonempty segments.")
        layout = replace(multiview_layout, items=items, caption_lengths=lengths if len(lengths) > 1 else ())
        if self.cached_kv is None or self.cached_freqs_gen is None:
            dummy = camera.new_empty(0)
            rotary = self.language_model.rotary_emb
            if self.cached_kv is None:
                caption_caches = []
                with self._offload_context("reasoner"):
                    for ids, length in zip(text_ids.split(lengths, dim=1), lengths, strict=True):
                        # A separate causal UND call per camera prevents text
                        # leakage and resets all three text position origins.
                        pos = torch.arange(length, device=ids.device).reshape(1, 1, length).expand(3, 1, -1)
                        cos, sin = rotary(dummy, position_ids=pos)
                        caption_caches.append(self.language_model(ids, (cos.unsqueeze(2), sin.unsqueeze(2))))
                self.cached_kv = [
                    (
                        torch.cat([cache[layer][0] for cache in caption_caches], dim=1),
                        torch.cat([cache[layer][1] for cache in caption_caches], dim=1),
                    )
                    for layer in range(len(caption_caches[0]))
                ]
                del caption_caches
            # No subsequent sample/modality follows this request, so the
            # reference-compatible endpoint cursor is intentionally unused.
            positions, _ = packed_position_ids(
                items,
                text_origin=max(lengths) + self.temporal_modality_margin,
                base_fps=self.base_fps,
                camera_compression=self.temporal_compression_factor,
                lidar_compression=1 if self.lidar_config is None else self.lidar_config["temporal_compression_factor"],
                enable_fps_modulation=self.enable_fps_modulation,
                align_views=kwargs.get("temporal_position_period") is not None,
            )
            cos, sin = rotary(dummy, position_ids=positions.unsqueeze(1).to(camera.device))
            self.cached_freqs_gen = (cos.unsqueeze(2), sin.unsqueeze(2))

        context = MultiviewAttentionContext(layout, self._multiview_mask_cache, self._multiview_buffer_cache)
        if layout.backend == "maskless":
            cp = _get_ulysses_state()[0] if _is_sp_active() else 1
            key = self.cached_kv[0][0]
            kv_heads, head_dim = key.shape[2] // cp, key.shape[3]
            query_heads = kv_heads * self._maskless_gqa_ratio
            cache_key = (
                "maskless",
                layout,
                text_ids.data_ptr(),
                text_ids.shape[1],
                key.device,
                key.dtype,
                query_heads,
                kv_heads,
                head_dim,
            )
            if cache_key not in self._multiview_mask_cache:
                plan = build_maskless_plan(layout, text_ids.shape[1], key.device, query_heads, kv_heads, head_dim)
                self._multiview_mask_cache[cache_key] = plan
            scratch_key = ("maskless_merge", query_heads, head_dim, key.dtype, key.device)
            if (*scratch_key, 0) not in self._multiview_buffer_cache:
                for index, buffer in enumerate(make_merge_scratch(query_heads, head_dim, key.dtype, key.device)):
                    self._multiview_buffer_cache[(*scratch_key, index)] = buffer
            scratch = [self._multiview_buffer_cache[(*scratch_key, index)] for index in range(6)]
            # Compact prompt-dependent dimensions cross the compiled GEN boundary
            # as dynamic tensors; caption lengths/maxima are plan tensor data.
            for k_und, v_und in self.cached_kv:
                torch._dynamo.mark_dynamic(k_und, 1)
                torch._dynamo.mark_dynamic(v_und, 1)
            context = replace(
                context,
                maskless_plan=(self._multiview_mask_cache[cache_key], scratch),
                fa_version=self._maskless_fa_version,
            )

        with self._offload_context("generator"):
            streams = []
            if has_control:
                streams.append(control_latents[0])
            streams.append(camera)
            if len(targets) == 2:
                if self.lidar_config is None:
                    raise ValueError("Joint requests require a checkpoint with LiDAR projections.")
                if has_control:
                    if lidar_control_latents is None:
                        raise ValueError("Joint transfer requires LiDAR control latents.")
                    streams.append(lidar_control_latents)
                streams.append(targets[1])
            if len(streams) != len(items):
                raise ValueError("Packed stream boundaries do not match the camera/LiDAR inputs.")
            # Pass a temporary: retaining the packed embedding in this frame
            # would keep it alive after SP sharding and throughout every layer.
            hidden = self._run_gen_layers(
                self._embed_packed_streams(items, streams, timestep, camera, noisy_frame_mask),
                multiview_layout=context,
            )
            outputs = []
            for item, latent, part in zip(
                items, streams, hidden.split([item.num_tokens for item in items], dim=1), strict=True
            ):
                if item.is_control:
                    continue
                if item.is_lidar:
                    projected = self.lidar_proj_out(self.norm_moe_gen(part))
                else:
                    projected = self._project_video_tokens(part)
                outputs.append(unpatchify_sensor(projected, tuple(latent.shape[1:]), self.latent_patch_size))
            return pack_state(outputs)

    def forward(
        self,
        *args,
        multiview_layout: MultiviewLayout | None = None,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        if multiview_layout is None:
            return super().forward(*args, **kwargs)
        return self._forward_packed(*args, multiview_layout=multiview_layout, **kwargs)
