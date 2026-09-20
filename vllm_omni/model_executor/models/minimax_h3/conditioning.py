# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""MiniMax H3 text-conditioning contract."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from typing import Any

import torch

MINIMAX_H3_TEXT_CONDITIONING_SCHEMA = "minimax_h3.text_conditioning/v1"
MINIMAX_H3_TEXT_HIDDEN_SIZE = 5120
MINIMAX_H3_PRESENTATION_TASK_KEY = "_minimax_h3_presentation_task"
MINIMAX_H3_CONDITION_LABELS_KEY = "_minimax_h3_condition_labels"


def _validate_contiguous_strided_layout(name: str, tensor: torch.Tensor) -> None:
    if tensor.layout != torch.strided:
        raise ValueError(
            f"{MINIMAX_H3_TEXT_CONDITIONING_SCHEMA}: {name} must use contiguous strided layout, got {tensor.layout}"
        )
    if not tensor.is_contiguous():
        raise ValueError(
            f"{MINIMAX_H3_TEXT_CONDITIONING_SCHEMA}: {name} must use contiguous "
            f"strided layout, got stride={tuple(tensor.stride())}"
        )


@dataclass(frozen=True)
class MiniMaxH3TextConditioning:
    """``minimax_h3.text_conditioning/v1`` semantic payload."""

    hidden_states: torch.Tensor
    token_tags: torch.Tensor

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
    ) -> MiniMaxH3TextConditioning:
        """Validate the semantic payload consumed by the diffusion stage."""
        hidden_states = payload.get("hidden_states")
        token_tags = payload.get("token_tags")
        if not isinstance(hidden_states, torch.Tensor) or not isinstance(token_tags, torch.Tensor):
            raise ValueError(
                f"{MINIMAX_H3_TEXT_CONDITIONING_SCHEMA}: conditioning requires hidden_states and token_tags tensors"
            )
        if hidden_states.ndim != 2 or hidden_states.shape[-1] != MINIMAX_H3_TEXT_HIDDEN_SIZE:
            raise ValueError(
                f"{MINIMAX_H3_TEXT_CONDITIONING_SCHEMA}: hidden_states must have shape "
                f"[tokens, {MINIMAX_H3_TEXT_HIDDEN_SIZE}], got {tuple(hidden_states.shape)}"
            )
        if hidden_states.dtype != torch.bfloat16:
            raise ValueError(
                f"{MINIMAX_H3_TEXT_CONDITIONING_SCHEMA}: hidden_states must have dtype "
                f"torch.bfloat16, got {hidden_states.dtype}"
            )
        _validate_contiguous_strided_layout("hidden_states", hidden_states)
        if token_tags.ndim != 1 or token_tags.shape[0] != hidden_states.shape[0]:
            raise ValueError(
                f"{MINIMAX_H3_TEXT_CONDITIONING_SCHEMA}: token_tags must align with hidden_states, got "
                f"token_tags={tuple(token_tags.shape)} and hidden_states={tuple(hidden_states.shape)}"
            )
        if token_tags.dtype != torch.int64:
            raise ValueError(
                f"{MINIMAX_H3_TEXT_CONDITIONING_SCHEMA}: token_tags must have dtype torch.int64, got {token_tags.dtype}"
            )
        _validate_contiguous_strided_layout("token_tags", token_tags)
        if not torch.all((token_tags == 0) | (token_tags == 1)):
            raise ValueError(
                f"{MINIMAX_H3_TEXT_CONDITIONING_SCHEMA}: text-encoder token_tags must contain only 0 and 1"
            )
        return cls(hidden_states=hidden_states, token_tags=token_tags)

    @classmethod
    def from_omni_payload(
        cls,
        payload: Mapping[str, Any],
    ) -> MiniMaxH3TextConditioning:
        """Validate and adapt the existing ``OmniPayload`` stage-wire view."""
        hidden_states = payload.get("hidden_states")
        if not isinstance(hidden_states, Mapping):
            raise ValueError(f"{MINIMAX_H3_TEXT_CONDITIONING_SCHEMA}: text encoder returned no hidden_states payload")
        hidden = hidden_states.get("output")
        if not isinstance(hidden, torch.Tensor):
            raise ValueError(
                f"{MINIMAX_H3_TEXT_CONDITIONING_SCHEMA}: text encoder returned no hidden_states.output tensor"
            )

        meta = payload.get("meta")
        if not isinstance(meta, Mapping):
            raise ValueError(f"{MINIMAX_H3_TEXT_CONDITIONING_SCHEMA}: text encoder returned no conditioning metadata")
        token_role_ids = meta.get("token_role_ids")
        if not isinstance(token_role_ids, torch.Tensor):
            raise ValueError(f"{MINIMAX_H3_TEXT_CONDITIONING_SCHEMA}: text encoder returned no token_role_ids tensor")
        if token_role_ids.ndim != 2 or token_role_ids.shape[-1] != 1:
            raise ValueError(
                f"{MINIMAX_H3_TEXT_CONDITIONING_SCHEMA}: stage-wire token_role_ids must have shape "
                f"[tokens, 1], got {tuple(token_role_ids.shape)}"
            )
        if token_role_ids.dtype != torch.int64:
            raise ValueError(
                f"{MINIMAX_H3_TEXT_CONDITIONING_SCHEMA}: stage-wire token_role_ids must have dtype "
                f"torch.int64, got {token_role_ids.dtype}"
            )
        _validate_contiguous_strided_layout("stage-wire token_role_ids", token_role_ids)

        return cls.from_payload(
            {
                "hidden_states": hidden,
                "token_tags": token_role_ids.squeeze(-1),
            }
        )

    def to_payload(self) -> dict[str, torch.Tensor]:
        return {
            "hidden_states": self.hidden_states,
            "token_tags": self.token_tags,
        }


MINIMAX_H3_ENCODER_REQUEST_KEY = "minimax_h3_encoder_request"
MINIMAX_H3_ENCODER_LAYOUT_KEY = "minimax_h3_encoder_layout"
STAGE_SCHEMA_VERSION = 2
_ENCODER_WIRE_SCHEMA_ID = 1

_TASK_TO_CODE = {"t2va": 1, "fl2va": 2, "ref2va": 3}
_CODE_TO_TASK = {value: key for key, value in _TASK_TO_CODE.items()}
_REF_KIND_TO_CODE = {"image": 1, "video": 2, "video_audio": 3, "audio": 4}
_CODE_TO_REF_KIND = {value: key for key, value in _REF_KIND_TO_CODE.items()}
_INTEGER_DTYPES = frozenset(
    {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }
)
_VIDEO_CONDITION_WIDTH = 96
_AUDIO_CONDITION_WIDTH = 32
_AUDIO_CONDITION_CHANNELS = 2


def _validate_edit_mask_structure(
    mask: torch.Tensor,
    *,
    name: str,
    shape: tuple[int, ...],
) -> None:
    """Validate only the internal tensor contract after encoder-boundary parsing."""
    if mask.dtype != torch.float32 or tuple(mask.shape) != shape:
        raise ValueError(f"MiniMax H3 {name} edit mask must be FP32 with shape {shape}")
    _validate_contiguous_strided_layout(f"{name} edit mask", mask)


def _validate_edit_tensors(
    clean_rows: torch.Tensor | None,
    raw_mask: torch.Tensor | None,
    *,
    name: str,
    width: int,
    mask_shape: tuple[int, ...],
) -> int:
    values = (clean_rows, raw_mask)
    if all(value is None for value in values):
        return 0
    if clean_rows is None or raw_mask is None:
        raise ValueError(f"MiniMax H3 {name} edit requires clean rows and a raw mask tensor")
    if clean_rows.dtype != torch.float32 or clean_rows.ndim != 2 or clean_rows.shape[1] != width:
        raise ValueError(f"MiniMax H3 {name} edit rows must be FP32 with shape [rows, {width}]")
    row_count = int(clean_rows.shape[0])
    if row_count <= 0:
        raise ValueError(f"MiniMax H3 {name} edit rows must not be empty")
    _validate_edit_mask_structure(raw_mask, name=name, shape=mask_shape)
    if raw_mask.device != clean_rows.device:
        raise ValueError(f"MiniMax H3 {name} edit tensors must share one device")
    _validate_contiguous_strided_layout(f"{name} edit rows", clean_rows)
    return row_count


def _packed_vector_rows(value: torch.Tensor, *, width: int) -> torch.Tensor:
    flat = value.reshape(-1)
    padding = (-int(flat.numel())) % width
    if padding:
        flat = torch.cat((flat, flat.new_zeros(padding)))
    return flat.reshape(-1, width)


def _pack_condition_slot(
    condition: torch.Tensor | None,
    clean_rows: torch.Tensor | None,
    raw_mask: torch.Tensor | None,
    *,
    width: int,
) -> torch.Tensor:
    if clean_rows is None:
        return condition if condition is not None else torch.empty((0,), dtype=torch.float32)
    if raw_mask is None:
        raise ValueError("MiniMax H3 edit rows require a raw mask tensor")
    values = [clean_rows, _packed_vector_rows(raw_mask, width=width)]
    if condition is not None:
        values.insert(0, condition)
    return torch.cat(values).contiguous()


def _unpack_condition_slot(
    value: torch.Tensor,
    *,
    condition_rows: int,
    edit_rows: int,
    mask_shape: tuple[int, ...],
    width: int,
    name: str,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    mask_values = math.prod(mask_shape) if mask_shape else int(edit_rows > 0)
    mask_wire_rows = (mask_values + width - 1) // width
    expected_rows = condition_rows + edit_rows + mask_wire_rows
    if expected_rows == 0:
        if value.dtype != torch.float32 or tuple(value.shape) != (0,):
            raise ValueError(f"MiniMax H3 empty {name} condition slot must be FP32 with shape [0]")
        return None, None, None
    if value.dtype != torch.float32 or value.ndim != 2 or tuple(value.shape) != (expected_rows, width):
        raise ValueError(
            f"MiniMax H3 packed {name} condition must be FP32 with shape "
            f"[{expected_rows}, {width}], got {value.dtype} {tuple(value.shape)}"
        )
    cursor = 0
    condition = value[:condition_rows].contiguous() if condition_rows else None
    cursor += condition_rows
    if edit_rows == 0:
        return condition, None, None
    clean_rows = value[cursor : cursor + edit_rows].contiguous()
    cursor += edit_rows
    raw_mask = value[cursor : cursor + mask_wire_rows].reshape(-1)[:mask_values].clone()
    if mask_shape:
        raw_mask = raw_mask.reshape(mask_shape)
    else:
        raw_mask = raw_mask.reshape(())
    return condition, clean_rows, raw_mask


def _wire_layout(payload: Mapping[str, Any]) -> torch.Tensor:
    private_metadata = payload.get("kv_metadata")
    if not isinstance(private_metadata, Mapping):
        raise ValueError("MiniMax H3 wire payload requires private layout metadata")
    layout = private_metadata.get(MINIMAX_H3_ENCODER_LAYOUT_KEY)
    if isinstance(layout, torch.Tensor):
        return layout
    raise ValueError("MiniMax H3 wire payload requires one packed encoder layout tensor")


def _validate_condition_tensors(
    visual: torch.Tensor | None,
    visual_shapes: Sequence[Sequence[int]],
    audio: torch.Tensor | None,
    audio_lengths: Sequence[int],
) -> None:
    if (visual is None) != (not visual_shapes):
        raise ValueError("MiniMax H3 visual condition and shapes must be present together")
    if (audio is None) != (not audio_lengths):
        raise ValueError("MiniMax H3 audio condition and lengths must be present together")

    parsed_shapes = [tuple(int(item) for item in shape) for shape in visual_shapes]
    if any(
        len(shape) != 3 or any(item <= 0 for item in shape) or shape[1] % 2 or shape[2] % 2 for shape in parsed_shapes
    ):
        raise ValueError("MiniMax H3 visual condition shapes must be positive [T, H, W] triplets with even H/W")
    if visual is not None:
        expected_rows = sum(t * (h // 2) * (w // 2) for t, h, w in parsed_shapes)
        if (
            visual.dtype != torch.float32
            or visual.ndim != 2
            or tuple(visual.shape)
            != (
                expected_rows,
                _VIDEO_CONDITION_WIDTH,
            )
        ):
            raise ValueError(
                "MiniMax H3 visual condition must be FP32 with shape "
                f"[{expected_rows}, {_VIDEO_CONDITION_WIDTH}], got {visual.dtype} {tuple(visual.shape)}"
            )

    parsed_lengths = [int(length) for length in audio_lengths]
    if any(length <= 0 for length in parsed_lengths):
        raise ValueError("MiniMax H3 audio condition lengths must be positive")
    if audio is not None:
        expected_rows = _AUDIO_CONDITION_CHANNELS * sum(parsed_lengths)
        if (
            audio.dtype != torch.float32
            or audio.ndim != 2
            or tuple(audio.shape)
            != (
                expected_rows,
                _AUDIO_CONDITION_WIDTH,
            )
        ):
            raise ValueError(
                "MiniMax H3 audio condition must be FP32 with shape "
                f"[{expected_rows}, {_AUDIO_CONDITION_WIDTH}], got {audio.dtype} {tuple(audio.shape)}"
            )


def _ref_block_rows(ref_blocks: Sequence[Mapping[str, Any]]) -> list[list[int]]:
    rows: list[list[int]] = []
    for block in ref_blocks:
        kind = str(block.get("kind") or "")
        kind_code = _REF_KIND_TO_CODE.get(kind)
        if kind_code is None:
            raise ValueError(f"unsupported MiniMax H3 ref kind {kind!r}")
        rows.append(
            [
                kind_code,
                int(block.get("ref_audio_t", 0)),
                int(block.get("latent_t", 0)),
                int(block.get("latent_h", 0)),
                int(block.get("latent_w", 0)),
            ]
        )
    return rows


def _decode_ref_blocks(value: torch.Tensor) -> tuple[dict[str, Any], ...]:
    if value.ndim != 2 or value.shape[1] != 5:
        raise ValueError("MiniMax H3 ref_blocks metadata must have shape [N, 5]")
    blocks: list[dict[str, Any]] = []
    for kind_code, ref_audio_t, ref_t, ref_h, ref_w in value.detach().cpu().tolist():
        kind = _CODE_TO_REF_KIND.get(int(kind_code))
        if kind is None:
            raise ValueError(f"unsupported MiniMax H3 ref kind code {kind_code!r}")
        block: dict[str, Any] = {"kind": kind}
        for key, item in (
            ("ref_audio_t", ref_audio_t),
            ("latent_t", ref_t),
            ("latent_h", ref_h),
            ("latent_w", ref_w),
        ):
            if item:
                block[key] = int(item)
        blocks.append(block)
    return tuple(blocks)


@dataclass(frozen=True)
class MiniMaxH3EncoderMediaInput:
    task: str
    height: int
    width: int
    num_frames: int
    latent_t: int
    audio_t: int
    images: tuple[torch.Tensor, ...] = ()
    videos: tuple[torch.Tensor, ...] = ()
    video_audios: tuple[tuple[torch.Tensor, int] | None, ...] = ()
    audios: tuple[tuple[torch.Tensor, int], ...] = ()
    keyframe_frame_indices: tuple[int, ...] = ()
    video_edit: torch.Tensor | None = None
    video_edit_mask: torch.Tensor | None = None
    audio_edit: tuple[torch.Tensor, int] | None = None
    audio_edit_mask: torch.Tensor | None = None

    @classmethod
    def from_mm_tensors(
        cls,
        values: Sequence[torch.Tensor],
        metadata: Mapping[str, Any],
    ) -> MiniMaxH3EncoderMediaInput:
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise ValueError("MiniMax H3 encoder media must be a tensor sequence")
        tensors = list(values)
        if any(not isinstance(item, torch.Tensor) for item in tensors):
            raise ValueError("MiniMax H3 encoder media input must contain only tensors")
        task = str(metadata["task"])
        if task not in _TASK_TO_CODE:
            raise ValueError(f"unsupported MiniMax H3 task {task!r}")
        height = int(metadata["height"])
        width = int(metadata["width"])
        num_frames = int(metadata["num_frames"])
        latent_t = int(metadata["latent_t"])
        audio_t = int(metadata["audio_t"])
        image_count = int(metadata.get("image_count", 0))
        video_count = int(metadata.get("video_count", 0))
        audio_count = int(metadata.get("audio_count", 0))
        if min(height, width, num_frames, latent_t, audio_t) <= 0:
            raise ValueError("MiniMax H3 encoder media dimensions must be positive")
        if min(image_count, video_count, audio_count) < 0:
            raise ValueError("MiniMax H3 encoder media counts must be non-negative")

        audio_flags = tuple(bool(value) for value in metadata.get("video_audio_flags", ()))
        video_audio_sample_rates = tuple(int(value) for value in metadata.get("video_audio_sample_rates", ()))
        audio_sample_rates = tuple(int(value) for value in metadata.get("audio_sample_rates", ()))
        if len(audio_flags) != video_count or len(video_audio_sample_rates) != video_count:
            raise ValueError("MiniMax H3 video audio flags must align with videos")
        if len(audio_sample_rates) != audio_count:
            raise ValueError("MiniMax H3 audio sample rates must align with audio inputs")
        if any((rate > 0) != has_audio for rate, has_audio in zip(video_audio_sample_rates, audio_flags, strict=True)):
            raise ValueError("MiniMax H3 embedded audio sample rates must align with audio flags")
        if any(rate <= 0 for rate in audio_sample_rates):
            raise ValueError("MiniMax H3 audio sample rates must be positive")

        cursor = 0
        images = tuple(tensors[cursor : cursor + image_count])
        cursor += image_count
        videos = tuple(tensors[cursor : cursor + video_count])
        cursor += video_count
        if len(images) != image_count or any(image.ndim != 3 or image.shape[-1] != 3 for image in images):
            raise ValueError("MiniMax H3 encoder images must have shape [H, W, 3]")
        if len(videos) != video_count or any(video.ndim != 4 or video.shape[-1] != 3 for video in videos):
            raise ValueError("MiniMax H3 encoder videos must have shape [T, H, W, 3]")

        video_audios: list[tuple[torch.Tensor, int] | None] = []
        for has_audio, sample_rate in zip(audio_flags, video_audio_sample_rates, strict=True):
            if not has_audio:
                video_audios.append(None)
                continue
            if cursor >= len(tensors):
                raise ValueError("MiniMax H3 encoder media input is truncated")
            waveform = tensors[cursor]
            cursor += 1
            if waveform.ndim not in (1, 2):
                raise ValueError("MiniMax H3 embedded audio must have shape [samples] or [channels, samples]")
            video_audios.append((waveform, sample_rate))

        audios: list[tuple[torch.Tensor, int]] = []
        for sample_rate in audio_sample_rates:
            if cursor >= len(tensors):
                raise ValueError("MiniMax H3 encoder media input is truncated")
            waveform = tensors[cursor]
            cursor += 1
            if waveform.ndim not in (1, 2):
                raise ValueError("MiniMax H3 audio must have shape [samples] or [channels, samples]")
            audios.append((waveform, sample_rate))

        video_edit = None
        video_edit_mask = None
        if bool(metadata.get("has_video_edit", False)):
            if cursor + 2 > len(tensors):
                raise ValueError("MiniMax H3 encoder video edit input is truncated")
            video_edit, video_edit_mask = tensors[cursor : cursor + 2]
            cursor += 2
            if tuple(video_edit.shape) != (num_frames, height, width, 3):
                raise ValueError(
                    "MiniMax H3 encoder video edit must have shape "
                    f"[{num_frames}, {height}, {width}, 3], got {tuple(video_edit.shape)}"
                )
            _validate_edit_mask_structure(
                video_edit_mask,
                name="video",
                shape=(latent_t, height // 16, width // 16),
            )

        audio_edit = None
        audio_edit_mask = None
        audio_edit_sample_rate = int(metadata.get("audio_edit_sample_rate", 0))
        if audio_edit_sample_rate:
            if cursor + 2 > len(tensors):
                raise ValueError("MiniMax H3 encoder audio edit input is truncated")
            waveform, audio_edit_mask = tensors[cursor : cursor + 2]
            cursor += 2
            if waveform.ndim not in (1, 2) or audio_edit_sample_rate <= 0:
                raise ValueError("MiniMax H3 encoder audio edit requires a waveform and positive sample rate")
            _validate_edit_mask_structure(audio_edit_mask, name="audio", shape=(2, audio_t))
            audio_edit = (waveform, audio_edit_sample_rate)
        if cursor != len(tensors):
            raise ValueError(f"MiniMax H3 encoder media input has {len(tensors) - cursor} trailing tensors")
        return cls(
            task=task,
            height=height,
            width=width,
            num_frames=num_frames,
            latent_t=latent_t,
            audio_t=audio_t,
            images=images,
            videos=videos,
            video_audios=tuple(video_audios),
            audios=tuple(audios),
            keyframe_frame_indices=tuple(int(value) for value in metadata.get("keyframe_frame_indices", ())),
            video_edit=video_edit,
            video_edit_mask=video_edit_mask,
            audio_edit=audio_edit,
            audio_edit_mask=audio_edit_mask,
        )

    def to_mm_tensors(self) -> list[torch.Tensor]:
        task_code = _TASK_TO_CODE.get(self.task)
        if task_code is None:
            raise ValueError(f"unsupported MiniMax H3 task {self.task!r}")
        if min(self.height, self.width, self.num_frames, self.latent_t, self.audio_t) <= 0:
            raise ValueError("MiniMax H3 encoder media dimensions must be positive")
        if len(self.video_audios) != len(self.videos):
            raise ValueError("MiniMax H3 video audio slots must align with videos")
        if any(image.ndim != 3 or image.shape[-1] != 3 for image in self.images):
            raise ValueError("MiniMax H3 encoder images must have shape [H, W, 3]")
        if any(video.ndim != 4 or video.shape[-1] != 3 for video in self.videos):
            raise ValueError("MiniMax H3 encoder videos must have shape [T, H, W, 3]")
        audio_items = [item for item in self.video_audios if item is not None] + list(self.audios)
        if any(waveform.ndim not in (1, 2) or int(sample_rate) <= 0 for waveform, sample_rate in audio_items):
            raise ValueError("MiniMax H3 encoder audio must have a waveform and positive sample rate")
        video_edit_values = (self.video_edit, self.video_edit_mask)
        if any(value is not None for value in video_edit_values):
            if self.video_edit is None or self.video_edit_mask is None:
                raise ValueError("MiniMax H3 encoder video edit requires a source and mask")
            if tuple(self.video_edit.shape) != (self.num_frames, self.height, self.width, 3):
                raise ValueError(
                    "MiniMax H3 encoder video edit must have shape "
                    f"[{self.num_frames}, {self.height}, {self.width}, 3], got {tuple(self.video_edit.shape)}"
                )
            _validate_edit_mask_structure(
                self.video_edit_mask,
                name="video",
                shape=(self.latent_t, self.height // 16, self.width // 16),
            )
        audio_edit_values = (self.audio_edit, self.audio_edit_mask)
        if any(value is not None for value in audio_edit_values):
            if self.audio_edit is None or self.audio_edit_mask is None:
                raise ValueError("MiniMax H3 encoder audio edit requires a source and mask")
            waveform, sample_rate = self.audio_edit
            if waveform.ndim not in (1, 2) or int(sample_rate) <= 0:
                raise ValueError("MiniMax H3 encoder audio edit requires a waveform and positive sample rate")
            _validate_edit_mask_structure(self.audio_edit_mask, name="audio", shape=(2, self.audio_t))
        tensors = [*self.images, *self.videos]
        for item in self.video_audios:
            if item is None:
                continue
            waveform, _sample_rate = item
            tensors.append(waveform)
        tensors.extend(waveform for waveform, _sample_rate in self.audios)
        if self.video_edit is not None:
            if self.video_edit_mask is None:
                raise ValueError("MiniMax H3 encoder video edit requires a mask")
            tensors.extend((self.video_edit, self.video_edit_mask))
        if self.audio_edit is not None:
            if self.audio_edit_mask is None:
                raise ValueError("MiniMax H3 encoder audio edit requires a mask")
            tensors.extend((self.audio_edit[0], self.audio_edit_mask))
        return tensors

    def to_metadata(self) -> dict[str, Any]:
        metadata = {
            "task": self.task,
            "height": self.height,
            "width": self.width,
            "num_frames": self.num_frames,
            "latent_t": self.latent_t,
            "audio_t": self.audio_t,
            "image_count": len(self.images),
            "video_count": len(self.videos),
            "audio_count": len(self.audios),
            "video_audio_flags": [item is not None for item in self.video_audios],
            "video_audio_sample_rates": [item[1] if item is not None else 0 for item in self.video_audios],
            "audio_sample_rates": [sample_rate for _waveform, sample_rate in self.audios],
            "keyframe_frame_indices": list(self.keyframe_frame_indices),
        }
        if self.video_edit is not None:
            metadata["has_video_edit"] = True
        if self.audio_edit is not None:
            metadata["audio_edit_sample_rate"] = int(self.audio_edit[1])
        return metadata


@dataclass(frozen=True)
class MiniMaxH3EncoderMediaConditioning:
    task: str
    height: int
    width: int
    num_frames: int
    latent_t: int
    audio_t: int
    visual_condition: torch.Tensor | None = None
    visual_condition_shapes: tuple[tuple[int, int, int], ...] = ()
    audio_condition: torch.Tensor | None = None
    audio_condition_lengths: tuple[int, ...] = ()
    ref_blocks: tuple[dict[str, Any], ...] = ()
    keyframe_frame_indices: tuple[int, ...] = ()
    video_edit_clean_rows: torch.Tensor | None = None
    video_edit_mask: torch.Tensor | None = None
    audio_edit_clean_rows: torch.Tensor | None = None
    audio_edit_mask: torch.Tensor | None = None
    audio_edit_source_t: int = 0

    def to_omni_components(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        task_code = _TASK_TO_CODE.get(self.task)
        if task_code is None:
            raise ValueError(f"unsupported MiniMax H3 task {self.task!r}")
        if min(self.height, self.width, self.num_frames, self.latent_t, self.audio_t) <= 0:
            raise ValueError("MiniMax H3 encoder media dimensions must be positive")
        _validate_condition_tensors(
            self.visual_condition,
            self.visual_condition_shapes,
            self.audio_condition,
            self.audio_condition_lengths,
        )
        video_mask_shape = (self.latent_t, self.height // 16, self.width // 16)
        audio_mask_shape = (2, self.audio_t)
        video_edit_rows = _validate_edit_tensors(
            self.video_edit_clean_rows,
            self.video_edit_mask,
            name="video",
            width=_VIDEO_CONDITION_WIDTH,
            mask_shape=video_mask_shape,
        )
        audio_edit_rows = _validate_edit_tensors(
            self.audio_edit_clean_rows,
            self.audio_edit_mask,
            name="audio",
            width=_AUDIO_CONDITION_WIDTH,
            mask_shape=audio_mask_shape,
        )
        expected_video_rows = self.latent_t * (self.height // 32) * (self.width // 32)
        if video_edit_rows not in (0, expected_video_rows):
            raise ValueError(f"MiniMax H3 video edit has {video_edit_rows} rows, expected {expected_video_rows}")
        expected_audio_rows = self.audio_t * _AUDIO_CONDITION_CHANNELS
        if audio_edit_rows not in (0, expected_audio_rows):
            raise ValueError(f"MiniMax H3 audio edit has {audio_edit_rows} rows, expected {expected_audio_rows}")
        if audio_edit_rows == 0:
            if self.audio_edit_source_t != 0:
                raise ValueError("MiniMax H3 audio_edit_source_t requires audio edit rows")
        elif not 0 < self.audio_edit_source_t <= self.audio_t:
            raise ValueError(f"MiniMax H3 audio_edit_source_t must be in [1, {self.audio_t}]")
        ref_rows = _ref_block_rows(self.ref_blocks)
        layout = [
            _ENCODER_WIRE_SCHEMA_ID,
            STAGE_SCHEMA_VERSION,
            task_code,
            self.height,
            self.width,
            self.num_frames,
            self.latent_t,
            self.audio_t,
            len(self.visual_condition_shapes),
            len(self.audio_condition_lengths),
            len(ref_rows),
            len(self.keyframe_frame_indices),
            video_edit_rows,
            audio_edit_rows,
            self.audio_edit_source_t,
        ]
        layout.extend(item for shape in self.visual_condition_shapes for item in shape)
        layout.extend(self.audio_condition_lengths)
        layout.extend(item for row in ref_rows for item in row)
        layout.extend(self.keyframe_frame_indices)
        return (
            _pack_condition_slot(
                self.visual_condition,
                self.video_edit_clean_rows,
                self.video_edit_mask,
                width=_VIDEO_CONDITION_WIDTH,
            ),
            _pack_condition_slot(
                self.audio_condition,
                self.audio_edit_clean_rows,
                self.audio_edit_mask,
                width=_AUDIO_CONDITION_WIDTH,
            ),
            torch.tensor(layout, dtype=torch.int64),
        )


@dataclass(frozen=True)
class MiniMaxH3EncoderConditioning:
    hidden_states: torch.Tensor
    token_tags: torch.Tensor
    task: str
    height: int
    width: int
    num_frames: int
    latent_t: int
    audio_t: int
    visual_condition: torch.Tensor | None = None
    visual_condition_shapes: tuple[tuple[int, int, int], ...] = ()
    audio_condition: torch.Tensor | None = None
    audio_condition_lengths: tuple[int, ...] = ()
    ref_blocks: tuple[dict[str, Any], ...] = ()
    keyframe_frame_indices: tuple[int, ...] = ()
    video_edit_clean_rows: torch.Tensor | None = None
    video_edit_mask: torch.Tensor | None = None
    audio_edit_clean_rows: torch.Tensor | None = None
    audio_edit_mask: torch.Tensor | None = None
    audio_edit_source_t: int = 0

    @classmethod
    def from_components(
        cls,
        text: MiniMaxH3TextConditioning,
        media: MiniMaxH3EncoderMediaConditioning,
    ) -> MiniMaxH3EncoderConditioning:
        """Combine local encoder results without a stage-wire round trip."""
        text = MiniMaxH3TextConditioning.from_payload(text.to_payload())
        _validate_condition_tensors(
            media.visual_condition,
            media.visual_condition_shapes,
            media.audio_condition,
            media.audio_condition_lengths,
        )
        return cls(
            hidden_states=text.hidden_states,
            token_tags=text.token_tags,
            **{item.name: getattr(media, item.name) for item in fields(media)},
        )

    @classmethod
    def from_omni_payload(cls, payload: Mapping[str, Any]) -> MiniMaxH3EncoderConditioning:
        hidden_payload = payload.get("hidden_states")
        embed_payload = payload.get("embed")
        meta_payload = payload.get("meta")
        hidden_states = hidden_payload.get("output") if isinstance(hidden_payload, Mapping) else None
        token_tags = meta_payload.get("token_role_ids") if isinstance(meta_payload, Mapping) else None
        visual = embed_payload.get("embedding") if isinstance(embed_payload, Mapping) else None
        audio = embed_payload.get("speech_feat") if isinstance(embed_payload, Mapping) else None
        if not all(isinstance(item, torch.Tensor) for item in (hidden_states, token_tags, visual, audio)):
            raise ValueError("MiniMax H3 encoder wire requires text, visual and audio tensors")
        assert isinstance(hidden_states, torch.Tensor)
        assert isinstance(token_tags, torch.Tensor)
        assert isinstance(visual, torch.Tensor)
        assert isinstance(audio, torch.Tensor)
        if token_tags.ndim == 2 and token_tags.shape[-1] == 1:
            token_tags = token_tags.squeeze(-1)
        text = MiniMaxH3TextConditioning.from_payload({"hidden_states": hidden_states, "token_tags": token_tags})
        for name, value in (("visual", visual), ("audio", audio)):
            if value.numel() == 0 and (value.dtype != torch.float32 or tuple(value.shape) != (0,)):
                raise ValueError(f"MiniMax H3 empty {name} condition slot must be FP32 with shape [0]")
        if not isinstance(meta_payload, Mapping):
            raise ValueError("MiniMax H3 wire payload requires a meta mapping")
        layout = _wire_layout(payload)
        if layout.dtype not in _INTEGER_DTYPES or layout.ndim != 1:
            raise ValueError("MiniMax H3 encoder layout must be a one-dimensional integer tensor")
        values = [int(item) for item in layout.detach().cpu().tolist()]
        if len(values) < 2:
            raise ValueError("MiniMax H3 encoder layout header is truncated")
        schema_id, version = values[:2]
        if schema_id != _ENCODER_WIRE_SCHEMA_ID or version != STAGE_SCHEMA_VERSION:
            raise ValueError(f"unsupported MiniMax H3 encoder wire schema {schema_id}:{version}")
        if len(values) < 15:
            raise ValueError("MiniMax H3 encoder layout header is truncated")
        (
            _schema_id,
            _version,
            task_code,
            height,
            width,
            num_frames,
            latent_t,
            audio_t,
            visual_count,
            audio_count,
            ref_count,
            keyframe_count,
            video_edit_rows,
            audio_edit_rows,
            audio_edit_source_t,
        ) = values[:15]
        task = _CODE_TO_TASK.get(task_code)
        if task is None:
            raise ValueError(f"unsupported MiniMax H3 wire task code {task_code!r}")
        if min(height, width, num_frames, latent_t, audio_t) <= 0:
            raise ValueError("MiniMax H3 encoder wire dimensions must be positive")
        if min(visual_count, audio_count, ref_count, keyframe_count) < 0:
            raise ValueError("MiniMax H3 encoder layout counts must be non-negative")
        if min(video_edit_rows, audio_edit_rows, audio_edit_source_t) < 0:
            raise ValueError("MiniMax H3 encoder edit layout values must be non-negative")
        expected_size = 15 + 3 * visual_count + audio_count + 5 * ref_count + keyframe_count
        if len(values) != expected_size:
            raise ValueError(f"MiniMax H3 encoder layout has {len(values)} integers, expected {expected_size}")
        cursor = 15
        visual_shapes = tuple(
            tuple(values[cursor + 3 * index : cursor + 3 * (index + 1)]) for index in range(visual_count)
        )
        cursor += 3 * visual_count
        audio_lengths = tuple(values[cursor : cursor + audio_count])
        cursor += audio_count
        ref_rows = values[cursor : cursor + 5 * ref_count]
        cursor += 5 * ref_count
        ref_blocks_tensor = torch.tensor(ref_rows, dtype=torch.int64).reshape(-1, 5)
        keyframe_frame_indices = tuple(values[cursor : cursor + keyframe_count])
        video_mask_shape = (latent_t, height // 16, width // 16) if video_edit_rows else ()
        audio_mask_shape = (2, audio_t) if audio_edit_rows else ()
        visual_condition_rows = sum(t * (h // 2) * (w // 2) for t, h, w in visual_shapes)
        audio_condition_rows = _AUDIO_CONDITION_CHANNELS * sum(audio_lengths)
        visual_condition, video_edit_clean_rows, video_edit_mask = _unpack_condition_slot(
            visual,
            condition_rows=visual_condition_rows,
            edit_rows=video_edit_rows,
            mask_shape=video_mask_shape,
            width=_VIDEO_CONDITION_WIDTH,
            name="visual",
        )
        audio_condition, audio_edit_clean_rows, audio_edit_mask = _unpack_condition_slot(
            audio,
            condition_rows=audio_condition_rows,
            edit_rows=audio_edit_rows,
            mask_shape=audio_mask_shape,
            width=_AUDIO_CONDITION_WIDTH,
            name="audio",
        )
        _validate_condition_tensors(
            visual_condition,
            visual_shapes,
            audio_condition,
            audio_lengths,
        )
        _validate_edit_tensors(
            video_edit_clean_rows,
            video_edit_mask,
            name="video",
            width=_VIDEO_CONDITION_WIDTH,
            mask_shape=(latent_t, height // 16, width // 16),
        )
        _validate_edit_tensors(
            audio_edit_clean_rows,
            audio_edit_mask,
            name="audio",
            width=_AUDIO_CONDITION_WIDTH,
            mask_shape=(2, audio_t),
        )
        expected_video_edit_rows = latent_t * (height // 32) * (width // 32)
        if video_edit_rows not in (0, expected_video_edit_rows):
            raise ValueError(f"MiniMax H3 video edit has {video_edit_rows} rows, expected {expected_video_edit_rows}")
        if audio_edit_rows not in (0, audio_t * _AUDIO_CONDITION_CHANNELS):
            raise ValueError(f"MiniMax H3 audio edit has {audio_edit_rows} rows, expected {audio_t * 2}")
        if (audio_edit_rows == 0 and audio_edit_source_t != 0) or (
            audio_edit_rows and not 0 < audio_edit_source_t <= audio_t
        ):
            raise ValueError("MiniMax H3 encoder audio edit source length is inconsistent")
        return cls(
            hidden_states=text.hidden_states,
            token_tags=text.token_tags,
            task=task,
            height=height,
            width=width,
            num_frames=num_frames,
            latent_t=latent_t,
            audio_t=audio_t,
            visual_condition=visual_condition,
            visual_condition_shapes=visual_shapes,
            audio_condition=audio_condition,
            audio_condition_lengths=audio_lengths,
            ref_blocks=_decode_ref_blocks(ref_blocks_tensor),
            keyframe_frame_indices=keyframe_frame_indices,
            video_edit_clean_rows=video_edit_clean_rows,
            video_edit_mask=video_edit_mask,
            audio_edit_clean_rows=audio_edit_clean_rows,
            audio_edit_mask=audio_edit_mask,
            audio_edit_source_t=audio_edit_source_t,
        )

    def to_omni_payload(self) -> dict[str, Any]:
        text = MiniMaxH3TextConditioning.from_payload(
            {
                "hidden_states": self.hidden_states,
                "token_tags": self.token_tags,
            }
        )
        visual, audio, layout = MiniMaxH3EncoderMediaConditioning(
            task=self.task,
            height=self.height,
            width=self.width,
            num_frames=self.num_frames,
            latent_t=self.latent_t,
            audio_t=self.audio_t,
            visual_condition=self.visual_condition,
            visual_condition_shapes=self.visual_condition_shapes,
            audio_condition=self.audio_condition,
            audio_condition_lengths=self.audio_condition_lengths,
            ref_blocks=self.ref_blocks,
            keyframe_frame_indices=self.keyframe_frame_indices,
            video_edit_clean_rows=self.video_edit_clean_rows,
            video_edit_mask=self.video_edit_mask,
            audio_edit_clean_rows=self.audio_edit_clean_rows,
            audio_edit_mask=self.audio_edit_mask,
            audio_edit_source_t=self.audio_edit_source_t,
        ).to_omni_components()
        return {
            "hidden_states": {"output": text.hidden_states},
            "embed": {
                "embedding": visual,
                "speech_feat": audio,
            },
            "meta": {"token_role_ids": text.token_tags},
            "kv_metadata": {MINIMAX_H3_ENCODER_LAYOUT_KEY: layout},
        }


__all__ = [
    "MINIMAX_H3_CONDITION_LABELS_KEY",
    "MINIMAX_H3_ENCODER_LAYOUT_KEY",
    "MINIMAX_H3_ENCODER_REQUEST_KEY",
    "MINIMAX_H3_PRESENTATION_TASK_KEY",
    "MINIMAX_H3_TEXT_CONDITIONING_SCHEMA",
    "MINIMAX_H3_TEXT_HIDDEN_SIZE",
    "STAGE_SCHEMA_VERSION",
    "MiniMaxH3EncoderConditioning",
    "MiniMaxH3EncoderMediaConditioning",
    "MiniMaxH3EncoderMediaInput",
    "MiniMaxH3TextConditioning",
]
