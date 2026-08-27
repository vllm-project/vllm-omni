# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""NeuCodec waveform decoder for NeuTTS-Air Stage 1."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger

from vllm_omni.model_executor.models.output_templates import OmniOutput

logger = init_logger(__name__)

DEFAULT_NEUCODEC_REPO = "neuphonic/neucodec"
NEUTTS_SAMPLE_RATE = 24_000
NEUTTS_HOP_LENGTH = 480
NEUTTS_STREAMING_CHUNK_FRAMES = 25


@dataclass
class _StreamAudioState:
    frames: list[torch.Tensor] = field(default_factory=list)
    emitted_samples: int = 0
    stride_samples: int = NEUTTS_STREAMING_CHUNK_FRAMES * NEUTTS_HOP_LENGTH


def _meta_scalar(value: Any, default: Any = None) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return default
        return value.detach().reshape(-1)[0].item()
    if isinstance(value, (list, tuple)):
        if not value:
            return default
        return _meta_scalar(value[0], default)
    return default if value is None else value


def _linear_overlap_add(
    frames: list[torch.Tensor],
    stride: int,
) -> torch.Tensor:
    """Match NeuTTS' triangular weighted overlap-add on one-dimensional audio."""
    if not frames:
        raise ValueError("NeuTTS-Air overlap-add requires at least one frame.")
    if stride <= 0:
        raise ValueError(f"NeuTTS-Air overlap-add stride must be positive, got {stride}.")

    first = frames[0]
    if first.ndim != 1:
        raise ValueError("NeuTTS-Air overlap-add expects one-dimensional audio frames.")

    total_size = max(stride * index + int(frame.numel()) for index, frame in enumerate(frames))
    output = torch.zeros(total_size, dtype=first.dtype, device=first.device)
    sum_weight = torch.zeros_like(output)

    for index, frame in enumerate(frames):
        if frame.ndim != 1:
            raise ValueError("NeuTTS-Air overlap-add expects one-dimensional audio frames.")
        if frame.dtype != first.dtype or frame.device != first.device:
            raise ValueError("NeuTTS-Air overlap-add frames must share dtype and device.")

        frame_length = int(frame.numel())
        if frame_length == 0:
            continue
        positions = torch.linspace(
            0,
            1,
            frame_length + 2,
            dtype=frame.dtype,
            device=frame.device,
        )[1:-1]
        weight = 0.5 - torch.abs(positions - 0.5)
        offset = stride * index
        output[offset : offset + frame_length] += weight * frame
        sum_weight[offset : offset + frame_length] += weight

    if bool(torch.any(sum_weight <= 0).item()):
        raise RuntimeError("NeuTTS-Air overlap-add produced uncovered audio samples.")
    return output / sum_weight


class NeuTTSAirCode2Wav(nn.Module):
    """Decode NeuTTS-Air's single-codebook speech codes into waveforms."""

    input_modalities = "audio"

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        del prefix

        self.vllm_config = vllm_config

        self.have_multimodal_outputs = True
        self.has_preprocess = False
        self.has_postprocess = False
        self.enable_update_additional_information = True
        self.requires_raw_input_tokens = True

        hf_config = vllm_config.model_config.hf_config
        self._codec_repo = str(getattr(hf_config, "neutts_codec_repo", DEFAULT_NEUCODEC_REPO))
        self._codec: nn.Module | None = None
        self._output_sample_rate = NEUTTS_SAMPLE_RATE
        self._stream_states: dict[str, _StreamAudioState] = {}

    def _ensure_codec_loaded(self) -> None:
        if self._codec is not None:
            return

        try:
            from neucodec import NeuCodec
        except (ImportError, ModuleNotFoundError) as exc:
            raise RuntimeError(
                "NeuTTS-Air Stage 1 requires the NeuCodec runtime and its compatible optional dependencies."
            ) from exc

        device = self.vllm_config.device_config.device
        codec = NeuCodec.from_pretrained(self._codec_repo)
        codec = codec.eval().to(device)
        self._codec = codec
        logger.info(
            "Loaded NeuCodec from %s on %s (sample_rate=%d).",
            self._codec_repo,
            device,
            self._output_sample_rate,
        )

    def embed_input_ids(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        """Give the generation runner a stable placeholder embedding shape."""
        if input_ids.numel() == 0:
            return torch.empty((0, 1), device=input_ids.device, dtype=torch.float32)
        return torch.zeros(
            (input_ids.shape[0], 1),
            device=input_ids.device,
            dtype=torch.float32,
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor | OmniOutput,
        sampling_metadata: Any = None,
    ) -> None:
        del hidden_states, sampling_metadata
        return None

    def _split_request_ids(
        self,
        ids: torch.Tensor,
        seq_token_counts: list[int] | None = None,
    ) -> list[torch.Tensor]:
        if seq_token_counts is not None and len(seq_token_counts) > 1:
            boundaries = [0]
            for count in seq_token_counts:
                boundaries.append(boundaries[-1] + count)
            num_ids = ids.numel()
            return [ids[boundaries[i] : min(boundaries[i + 1], num_ids)] for i in range(len(seq_token_counts))]

        if is_forward_context_available():
            slices = get_forward_context().ubatch_slices
            if slices is not None and len(slices) > 1 and not any(hasattr(item, "token_slice") for item in slices):
                boundaries = [0]
                for item in slices:
                    boundaries.append(boundaries[-1] + item)
                return [ids[boundaries[i] : boundaries[i + 1]] for i in range(len(boundaries) - 1)]

        return [ids]

    def _decode_codes(self, one_request_codes: torch.Tensor) -> torch.Tensor:
        self._ensure_codec_loaded()
        assert self._codec is not None

        codes = one_request_codes.reshape(1, 1, -1).to(dtype=torch.long)
        audio = self._codec.decode_code(codes)
        if not isinstance(audio, torch.Tensor) or audio.ndim != 3 or audio.shape[0] != 1 or audio.shape[1] != 1:
            raise RuntimeError("NeuCodec.decode_code() must return a [1, 1, T] tensor.")
        return audio[0, 0].to(dtype=torch.float32).reshape(-1)

    @staticmethod
    def _runtime_info_list(
        model_intermediate_buffer: list[dict[str, Any]] | None,
        runtime_additional_information: list[dict[str, Any]] | None,
        num_requests: int,
    ) -> list[dict[str, Any]]:
        raw_infos = (
            model_intermediate_buffer if isinstance(model_intermediate_buffer, list) else runtime_additional_information
        )
        infos = [info if isinstance(info, dict) else {} for info in (raw_infos or [])]
        if len(infos) < num_requests:
            infos.extend({} for _ in range(num_requests - len(infos)))
        return infos[:num_requests]

    @staticmethod
    def _stream_metadata(
        info: dict[str, Any],
    ) -> tuple[bool, str | None, int, int, int, int, bool]:
        meta = info.get("meta", {}) if isinstance(info, dict) else {}
        if not isinstance(meta, dict):
            meta = {}

        streaming = bool(_meta_scalar(meta.get("codec_streaming"), False))
        request_id_raw = _meta_scalar(
            meta.get("req_id", meta.get("request_id")),
            None,
        )
        request_id = str(request_id_raw) if request_id_raw is not None else None
        left_context = int(_meta_scalar(meta.get("left_context_size"), 0))
        right_holdback = int(_meta_scalar(meta.get("right_holdback_size"), 0))
        processed_frames = int(_meta_scalar(meta.get("num_processed_tokens"), 0))
        chunk_frames = int(
            _meta_scalar(
                meta.get("codec_chunk_frames"),
                NEUTTS_STREAMING_CHUNK_FRAMES,
            )
        )
        finished = bool(
            _meta_scalar(
                meta.get("stream_finished", meta.get("finished")),
                False,
            )
        )

        if min(left_context, right_holdback, processed_frames) < 0:
            raise ValueError("NeuTTS-Air streaming metadata cannot be negative.")
        if chunk_frames <= 0:
            raise ValueError("NeuTTS-Air codec_chunk_frames must be positive.")
        return (
            streaming,
            request_id,
            left_context,
            right_holdback,
            processed_frames,
            chunk_frames,
            finished,
        )

    def _decode_streaming_request(
        self,
        one_request_codes: torch.Tensor,
        *,
        request_id: str,
        left_context_frames: int,
        right_holdback_frames: int,
        processed_frames: int,
        chunk_frames: int,
        finished: bool,
    ) -> torch.Tensor:
        state = self._stream_states.get(request_id)

        if processed_frames == 0:
            if not finished:
                raise RuntimeError("NeuTTS-Air received a non-final streaming chunk with no processed frames.")
            if state is None:
                return torch.zeros((0,), dtype=torch.float32)
            mixed = _linear_overlap_add(state.frames, state.stride_samples)
            delta = mixed[state.emitted_samples :]
            self._stream_states.pop(request_id, None)
            return delta

        decoded = self._decode_codes(one_request_codes)
        left_samples = left_context_frames * NEUTTS_HOP_LENGTH
        right_samples = right_holdback_frames * NEUTTS_HOP_LENGTH
        if left_samples + right_samples >= decoded.numel():
            raise RuntimeError(
                "NeuTTS-Air streaming trim removes the entire decoded waveform: "
                f"decoded={decoded.numel()}, left={left_samples}, right={right_samples}."
            )
        end = int(decoded.numel()) - right_samples if right_samples else None
        cropped = decoded[left_samples:end]

        stride_samples = chunk_frames * NEUTTS_HOP_LENGTH
        if state is None:
            state = _StreamAudioState(stride_samples=stride_samples)
            self._stream_states[request_id] = state
        elif state.stride_samples != stride_samples:
            raise RuntimeError("NeuTTS-Air streaming chunk size changed within one request.")

        state.frames.append(cropped)
        mixed = _linear_overlap_add(state.frames, state.stride_samples)
        output_start = state.emitted_samples
        if finished:
            output_end = int(mixed.numel())
        else:
            output_end = min(
                output_start + processed_frames * NEUTTS_HOP_LENGTH,
                int(mixed.numel()),
            )
        delta = mixed[output_start:output_end]
        state.emitted_samples = output_end

        if finished:
            self._stream_states.pop(request_id, None)
        return delta

    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        intermediate_tensors: Any = None,
        inputs_embeds: torch.Tensor | None = None,
        model_intermediate_buffer: list[dict[str, Any]] | None = None,
        runtime_additional_information: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> OmniOutput:
        """Decode synchronous code sequences or streaming code windows."""
        del positions, intermediate_tensors, inputs_embeds

        sample_rate = torch.tensor(
            self._output_sample_rate,
            dtype=torch.int32,
        )
        empty = torch.zeros((0,), dtype=torch.float32)

        if input_ids is None or input_ids.numel() == 0:
            return OmniOutput(
                text_hidden_states=None,
                multimodal_outputs={
                    "model_outputs": [empty],
                    "sr": [sample_rate],
                },
            )

        ids = input_ids.reshape(-1).to(dtype=torch.long)
        request_codes = self._split_request_ids(
            ids,
            kwargs.get("seq_token_counts"),
        )
        info_list = self._runtime_info_list(
            model_intermediate_buffer,
            runtime_additional_information,
            len(request_codes),
        )

        audio_outputs: list[torch.Tensor] = []
        for index, one_request_codes in enumerate(request_codes):
            if one_request_codes.numel() == 0:
                audio_outputs.append(empty)
                continue

            (
                streaming,
                request_id,
                left_context,
                right_holdback,
                processed_frames,
                chunk_frames,
                finished,
            ) = self._stream_metadata(info_list[index])

            if not streaming:
                audio_outputs.append(self._decode_codes(one_request_codes))
                continue
            if request_id is None:
                raise RuntimeError("NeuTTS-Air streaming Stage 1 requires meta.req_id.")

            audio_outputs.append(
                self._decode_streaming_request(
                    one_request_codes,
                    request_id=request_id,
                    left_context_frames=left_context,
                    right_holdback_frames=right_holdback,
                    processed_frames=processed_frames,
                    chunk_frames=chunk_frames,
                    finished=finished,
                )
            )

        return OmniOutput(
            text_hidden_states=None,
            multimodal_outputs={
                "model_outputs": audio_outputs,
                "sr": [sample_rate] * len(audio_outputs),
            },
        )

    def on_requests_finished(
        self,
        finished_req_ids: Iterable[str],
    ) -> None:
        """Release streaming decoder state for finished or aborted requests."""
        for request_id in finished_req_ids:
            self._stream_states.pop(str(request_id), None)

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> set[str]:
        # NeuCodec loads its own external checkpoint, not Stage 0's Qwen2 weights.
        del weights
        return set()
