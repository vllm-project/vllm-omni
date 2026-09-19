# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Offline vision encoder graphs; audio and stateful duplex remain eager."""

from collections.abc import Hashable
from typing import Any, cast

import torch
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask
from vllm.config import VllmConfig
from vllm.model_executor.models.interfaces import SupportsEncoderCudaGraph, supports_encoder_cudagraph
from vllm.v1.worker import encoder_cudagraph_defs
from vllm.v1.worker.encoder_cudagraph_defs import (
    EncoderCudaGraphCaptureInputs,
    EncoderCudaGraphConfig,
    EncoderCudaGraphReplayBuffers,
    EncoderItemSpec,
)

from vllm_omni.platforms import current_omni_platform

_AXIS_KEY = getattr(encoder_cudagraph_defs, "ENCODER_CUDAGRAPH_AXIS_KEYS_KWARG", "encoder_cudagraph_axis_keys")
_LAYOUT_KEY = "minicpmo_encoder_layout"
# Reuse selected-batch metadata across the manager's spec and replay calls.
_PARSE_KEY = "minicpmo_encoder_parse"
_SPECS_KEY = "minicpmo_encoder_specs"
_PATCH_CAPS = (1024, 1152, 2048)


def bind_minicpmo_encoder_cudagraph(model: torch.nn.Module, thinker: torch.nn.Module) -> None:
    """Expose the Thinker's protocol on the actual Stage-0 serving instance.

    Protocol checks use static attribute lookup, so __getattr__ forwarding
    would silently leave the manager disabled. Binding concrete attributes
    also avoids advertising this interface on Talker or unsupported pins.
    """
    if not supports_encoder_cudagraph(thinker):
        return
    if thinker.vpm is None:
        return
    names = (
        "supports_encoder_cudagraph",
        "get_encoder_cudagraph_config",
        "get_input_modality",
        "get_max_frames_per_video",
        "get_encoder_cudagraph_budget_range",
        "get_encoder_cudagraph_item_specs",
        "select_encoder_cudagraph_items",
        "postprocess_encoder_output",
        "prepare_encoder_cudagraph_capture_inputs",
        "prepare_encoder_cudagraph_replay_buffers",
        "encoder_cudagraph_forward",
        "encoder_eager_forward",
    )
    for name in names:
        setattr(model, name, getattr(thinker, name))


def _ceiling(value: int, tiers: tuple[int, ...]) -> int:
    # Out-of-range values stay out of range rather than being clamped: the layout
    # key has to describe the input that actually arrives, and the caller decides
    # between the eager path and a named failure.
    return next((tier for tier in tiers if value <= tier), value)


class _MiniCPMO45EncoderCudaGraphMixin(SupportsEncoderCudaGraph):
    def get_input_modality(self, mm_kwargs: dict[str, Any]) -> str:
        if "audio_features" in mm_kwargs:
            return "audio"
        return "video" if "video_pixel_values" in mm_kwargs else "image"

    def get_encoder_cudagraph_config(self) -> EncoderCudaGraphConfig:
        axes = []
        modalities = []
        if self.vpm is not None and not self.vpm._use_flash_attention_2:
            modalities.extend(["image", "video"])
            # Capture the largest intermediates first so smaller layouts can
            # reuse the graph pool instead of growing fragmented segments.
            axes.extend(("vision", patches) for patches in reversed(_PATCH_CAPS))
        # Audio remains eager: even padded BF16 convolution changes values
        # at rounding boundaries, which can change greedy transcript tokens.
        return EncoderCudaGraphConfig(
            modalities=modalities,
            buffer_keys=[
                "pixels",
                "position_ids",
                "vision_mask",
                "resampler_positions",
                "resampler_mask",
            ],
            out_hidden_size=int(self.config.hidden_size),
            max_frames_per_video=self.get_max_frames_per_video(),
            # Only reachable when the mixin is used directly: serving goes
            # through ``bind_minicpmo_encoder_cudagraph``, which leaves the
            # protocol unbound when there is no vision tower, so the runner
            # factory never builds a manager off this empty tuple.
            capture_axes=(tuple(axes),) if axes else (),
        )

    def get_max_frames_per_video(self) -> int:
        return int(self.multimodal_config.media_io_kwargs.get("video", {}).get("num_frames", 128))

    def get_encoder_cudagraph_budget_range(self, vllm_config: VllmConfig) -> tuple[int, int]:
        # A bounded default avoids multiplying every model-length budget by
        # every layout tier. Larger items remain eligible for eager fallback.
        limit = min(vllm_config.scheduler_config.max_num_batched_tokens, vllm_config.model_config.max_model_len)
        max_budget = max(1, min(4 * int(self.config.query_num), limit))
        # The manager infers max_batch_size = max_budget // min_budget, so a
        # floor equal to the ceiling pins it to a single item per graph. Use
        # one query group (the smallest per-item output) as the floor: with the
        # default 4x ceiling that yields max_budget // min_budget == 4, while
        # still never exceeding the ceiling or dropping to zero.
        min_budget = min(max(1, int(self.config.query_num)), max_budget)
        return min_budget, max_budget

    def _vision_layout(self, slices: int, patches: int) -> tuple[str, int]:
        """Reject in-budget layouts the manager cannot replay."""
        budgets = self.vllm_config.compilation_config.encoder_cudagraph_token_budgets
        max_budget = max(budgets) if budgets else self.get_encoder_cudagraph_budget_range(self.vllm_config)[1]
        extent = _ceiling(patches, _PATCH_CAPS)
        if slices * int(self.config.query_num) <= max_budget and extent not in _PATCH_CAPS:
            raise ValueError(
                f"No captured vision layout for {slices} slices of {patches} patches; "
                f"captured patch caps: {_PATCH_CAPS}. Disable cudagraph_mm_encoder for this input."
            )
        return ("vision", extent)

    def _encoder_data(self, mm_kwargs: dict[str, Any]):
        cached = mm_kwargs.get(_PARSE_KEY)
        if cached is not None:
            return cached
        modality = self.get_input_modality(mm_kwargs)
        kwargs = (
            {key.removeprefix("video_"): value for key, value in mm_kwargs.items()}
            if modality == "video"
            else mm_kwargs
        )
        if len(kwargs["pixel_values"]) == 0:
            return modality, [], torch.empty((0, 2), dtype=torch.int32), []
        data = self._parse_and_validate_vision_input(modality, **kwargs)
        counts = data["num_slices"].tolist()
        return modality, data["pixel_values"], data["tgt_sizes"], counts

    def get_encoder_cudagraph_item_specs(self, mm_kwargs: dict[str, Any]) -> list[EncoderItemSpec]:
        if _SPECS_KEY in mm_kwargs:
            return cast(list[EncoderItemSpec], mm_kwargs[_SPECS_KEY])
        modality, features, metadata, counts = self._encoder_data(mm_kwargs)
        patch_counts = metadata.prod(-1).split(counts)
        return [
            EncoderItemSpec(input_size=int(group.sum()), output_tokens=count * int(self.config.query_num))
            for count, group in zip(counts, patch_counts)
        ]

    def select_encoder_cudagraph_items(self, mm_kwargs: dict[str, Any], indices: list[int]) -> dict[str, Any]:
        modality, features, metadata, counts = self._encoder_data(mm_kwargs)
        offsets = [0]
        for count in counts:
            offsets.append(offsets[-1] + count)
        selected_counts = [counts[index] for index in indices]
        slices = sum(selected_counts)
        pixel_key = "video_pixel_values" if modality == "video" else "pixel_values"
        size_key = "video_tgt_sizes" if modality == "video" else "tgt_sizes"
        groups = metadata.split(counts)
        sizes = [groups[index] for index in indices]
        pixel_groups = [features[offsets[index] : offsets[index + 1]] for index in indices]
        selected = {
            pixel_key: pixel_groups,
            size_key: sizes,
        }
        patches = int(torch.cat(sizes).prod(-1).max()) if sizes else 0
        layout = self._vision_layout(slices, patches)
        selected[_LAYOUT_KEY] = layout
        selected[_AXIS_KEY] = (layout,)
        selected[_PARSE_KEY] = (
            modality,
            [feature for group in pixel_groups for feature in group],
            torch.cat(sizes) if sizes else torch.empty((0, 2), dtype=torch.int32),
            selected_counts,
        )
        selected[_SPECS_KEY] = [
            EncoderItemSpec(input_size=int(size.prod(-1).sum()), output_tokens=count * int(self.config.query_num))
            for size, count in zip(sizes, selected_counts)
        ]
        return selected

    def prepare_encoder_cudagraph_capture_inputs(
        self,
        token_budget: int,
        max_batch_size: int,
        max_frames_per_batch: int,
        device: torch.device,
        dtype: torch.dtype,
        path: str = "default",
        axis_keys: tuple[Hashable, ...] | None = None,
    ) -> EncoderCudaGraphCaptureInputs:
        kind, extent = axis_keys[0]
        capacity = max(1, token_budget // int(self.config.query_num))
        patch = int(self.vpm.embeddings.patch_size)
        values = {
            "pixels": torch.zeros(capacity, 3, patch, extent * patch, device=device, dtype=dtype),
            "position_ids": torch.zeros(capacity, extent, device=device, dtype=torch.long),
            "vision_mask": torch.zeros(capacity, 1, extent, extent, device=device, dtype=dtype),
            "resampler_positions": torch.zeros(capacity, extent, self.resampler.embed_dim, device=device, dtype=dtype),
            "resampler_mask": torch.zeros(capacity, extent, device=device, dtype=torch.bool),
        }
        return EncoderCudaGraphCaptureInputs(values=values)

    def prepare_encoder_cudagraph_replay_buffers(
        self,
        mm_kwargs: dict[str, Any],
        max_batch_size: int,
        max_frames_per_batch: int,
        path: str = "default",
    ) -> EncoderCudaGraphReplayBuffers:
        kind, extent = mm_kwargs[_LAYOUT_KEY]
        modality, features, metadata, counts = self._encoder_data(mm_kwargs)
        device, dtype = next(self.vpm.parameters()).device, next(self.vpm.parameters()).dtype
        patch = int(self.vpm.embeddings.patch_size)
        pixels = torch.zeros(len(features), 3, patch, extent * patch, device=device, dtype=dtype)
        for index, feature in enumerate(features):
            pixels[index, ..., : feature.shape[-1]] = feature
        sizes = metadata.to(device)
        mask = torch.arange(extent, device=device)[None, :] < sizes.prod(-1)[:, None]
        position_ids = self.vpm.embeddings._create_position_ids(mask[:, None, :], sizes, device=device)
        positions, padding_mask = self.resampler.prepare_metadata(sizes, device=device, dtype=dtype)
        positions = torch.nn.functional.pad(positions.permute(1, 0, 2), (0, 0, 0, extent - positions.shape[0]))
        padding_mask = torch.nn.functional.pad(padding_mask, (0, extent - padding_mask.shape[1]), value=True)
        values = {
            "pixels": pixels,
            "position_ids": position_ids,
            "vision_mask": _prepare_4d_attention_mask(mask, dtype),
            "resampler_positions": positions,
            "resampler_mask": padding_mask,
        }
        return EncoderCudaGraphReplayBuffers(values=values)

    def encoder_cudagraph_forward(self, values: dict[str, torch.Tensor], path: str = "default") -> torch.Tensor:
        hidden = self.vpm(
            values["pixels"], position_ids=values["position_ids"], encoder_attention_mask=values["vision_mask"]
        ).last_hidden_state
        return self.resampler(
            hidden, pos_embed=values["resampler_positions"].permute(1, 0, 2), key_padding_mask=values["resampler_mask"]
        ).flatten(0, 1)

    def encoder_eager_forward(self, mm_kwargs: dict[str, Any], path: str = "default") -> torch.Tensor:
        return torch.cat(self.get_multimodal_embeddings(**mm_kwargs))


# Do not advertise the protocol on the released pin: it cannot dispatch the
# layout axis, so enabling the flag there must keep these encoders eager.
class _NoEncoderCudaGraph:
    pass


def _select_encoder_cudagraph_mixin():
    if "capture_axes" in EncoderCudaGraphConfig.__dataclass_fields__ and current_omni_platform.is_cuda():
        return _MiniCPMO45EncoderCudaGraphMixin
    return _NoEncoderCudaGraph


MiniCPMO45EncoderCudaGraphMixin = _select_encoder_cudagraph_mixin()
