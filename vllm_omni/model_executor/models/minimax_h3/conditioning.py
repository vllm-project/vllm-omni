# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""MiniMax H3 validated text and unified encoder-conditioning contracts."""

from __future__ import annotations

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
STAGE_SCHEMA_VERSION = 1
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
        tensors = [*self.images, *self.videos]
        for item in self.video_audios:
            if item is None:
                continue
            waveform, _sample_rate = item
            tensors.append(waveform)
        tensors.extend(waveform for waveform, _sample_rate in self.audios)
        return tensors

    def to_metadata(self) -> dict[str, Any]:
        return {
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
        empty = torch.empty((0,), dtype=torch.float32)
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
        ]
        layout.extend(item for shape in self.visual_condition_shapes for item in shape)
        layout.extend(self.audio_condition_lengths)
        layout.extend(item for row in ref_rows for item in row)
        layout.extend(self.keyframe_frame_indices)
        return (
            self.visual_condition if self.visual_condition is not None else empty,
            self.audio_condition if self.audio_condition is not None else empty,
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
        if not hidden_states.is_floating_point():
            raise ValueError("MiniMax H3 hidden_states must use a floating-point dtype")
        if token_tags.dtype != torch.int64:
            raise ValueError("MiniMax H3 token_tags must use torch.int64")
        for name, value in (("visual", visual), ("audio", audio)):
            if value.numel() == 0 and (value.dtype != torch.float32 or tuple(value.shape) != (0,)):
                raise ValueError(f"MiniMax H3 empty {name} condition slot must be FP32 with shape [0]")
        if not isinstance(meta_payload, Mapping):
            raise ValueError("MiniMax H3 wire payload requires a meta mapping")
        layout = _wire_layout(payload)
        if layout.dtype not in _INTEGER_DTYPES or layout.ndim != 1:
            raise ValueError("MiniMax H3 encoder layout must be a one-dimensional integer tensor")
        values = [int(item) for item in layout.detach().cpu().tolist()]
        if len(values) < 12:
            raise ValueError("MiniMax H3 encoder layout header is truncated")
        (
            schema_id,
            version,
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
        ) = values[:12]
        if schema_id != _ENCODER_WIRE_SCHEMA_ID or version != STAGE_SCHEMA_VERSION:
            raise ValueError(f"unsupported MiniMax H3 encoder wire schema {schema_id}:{version}")
        task = _CODE_TO_TASK.get(task_code)
        if task is None:
            raise ValueError(f"unsupported MiniMax H3 wire task code {task_code!r}")
        if min(height, width, num_frames, latent_t, audio_t) <= 0:
            raise ValueError("MiniMax H3 encoder wire dimensions must be positive")
        if min(visual_count, audio_count, ref_count, keyframe_count) < 0:
            raise ValueError("MiniMax H3 encoder layout counts must be non-negative")
        expected_size = 12 + 3 * visual_count + audio_count + 5 * ref_count + keyframe_count
        if len(values) != expected_size:
            raise ValueError(f"MiniMax H3 encoder layout has {len(values)} integers, expected {expected_size}")
        cursor = 12
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
        visual_condition = visual if visual.numel() else None
        audio_condition = audio if audio.numel() else None
        _validate_condition_tensors(
            visual_condition,
            visual_shapes,
            audio_condition,
            audio_lengths,
        )
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
        )

    def to_omni_payload(self) -> dict[str, Any]:
        text = MiniMaxH3TextConditioning.from_payload(
            {
                "hidden_states": self.hidden_states,
                "token_tags": self.token_tags,
            }
        )
        if not text.hidden_states.is_floating_point():
            raise ValueError("MiniMax H3 hidden_states must use a floating-point dtype")
        if text.token_tags.dtype != torch.int64:
            raise ValueError("MiniMax H3 token_tags must use torch.int64")
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
    "MINIMAX_H3_TEXT_HIDDEN_SIZE",
    "STAGE_SCHEMA_VERSION",
    "MiniMaxH3EncoderConditioning",
    "MiniMaxH3EncoderMediaConditioning",
    "MiniMaxH3EncoderMediaInput",
    "MiniMaxH3TextConditioning",
]
