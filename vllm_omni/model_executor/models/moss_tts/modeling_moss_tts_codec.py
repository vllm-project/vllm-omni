# Copyright 2026 OpenMOSS and the vLLM-Omni team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
"""MOSS-TTS Stage-1 codec decoder: RVQ codes → 24 kHz waveform."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader import DefaultModelLoader
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.utils.torch_utils import set_default_torch_dtype

from vllm_omni.model_executor.models.moss_tts.audio_tokenizer import (
    MossAudioTokenizerConfig,
    MossAudioTokenizerModel,
)
from vllm_omni.model_executor.models.moss_tts.audio_tokenizer_v2 import (
    MossAudioTokenizerModel as MossAudioTokenizerV2Model,
)
from vllm_omni.model_executor.models.moss_tts.configuration_moss_audio_tokenizer_v2 import (
    MossAudioTokenizerConfig as MossAudioTokenizerV2Config,
)
from vllm_omni.model_executor.models.moss_tts.cuda_graph_streaming_decoder_wrapper import (
    CUDAGraphStreamingDecoderWrapper,
)
from vllm_omni.model_executor.models.moss_tts.moss_codec_cudagraph import (
    MossTTSCUDAGraphCodecWrapper,
)
from vllm_omni.model_executor.models.output_templates import OmniOutput

logger = init_logger(__name__)


class _MossCodecStreamSession:
    """Persistent state pool with compact per-step execution batches."""

    def __init__(
        self,
        codec: nn.Module,
        *,
        state_capacity: int,
        n_vq: int,
        vllm_config: VllmConfig,
        graph_batch_sizes: list[int] | None = None,
        graph_frame_sizes: list[int] | None = None,
    ) -> None:
        self._codec = codec
        self._state_capacity = int(state_capacity)
        self._n_vq = int(n_vq)
        self._device = next(codec.parameters()).device
        self._free_stream_slots = list(reversed(range(self._state_capacity)))
        self._leased_slots: set[int] = set()
        self._closed = False
        self._cudagraph_wrapper: CUDAGraphStreamingDecoderWrapper | None = None
        batch_sizes = sorted({int(size) for size in (graph_batch_sizes or []) if 0 < int(size) <= self._state_capacity})
        frame_sizes = sorted({int(size) for size in (graph_frame_sizes or []) if int(size) > 0})
        scratch_capacity = max(batch_sizes, default=0) if self._device.type == "cuda" else 0
        self._total_state_capacity = self._state_capacity + scratch_capacity
        self._state_slot_ids = torch.arange(
            self._total_state_capacity,
            device=self._device,
            dtype=torch.long,
        )
        self._samples_per_frame = int(codec.downsample_rate)
        initialize_state_pool = getattr(codec, "initialize_decoder_state_pool", None)
        if not callable(initialize_state_pool):
            raise RuntimeError("The streaming codec does not implement a decoder state pool.")
        with torch.no_grad():
            initialize_state_pool(self._state_capacity, scratch_capacity)
        if batch_sizes and frame_sizes and self._device.type == "cuda":
            self._cudagraph_wrapper = CUDAGraphStreamingDecoderWrapper(
                codec,
                state_capacity=self._state_capacity,
                batch_sizes=batch_sizes,
                frame_sizes=frame_sizes,
                num_quantizers=self._n_vq,
                vllm_config=vllm_config,
            )
            self._cudagraph_wrapper.warmup(self._device)
            self.reset_slots(list(range(self._state_capacity + scratch_capacity)))
            if not self._cudagraph_wrapper.is_ready:
                self._cudagraph_wrapper = None

    def acquire(self) -> int | None:
        if not self._free_stream_slots:
            return None
        slot = self._free_stream_slots.pop()
        self._leased_slots.add(slot)
        return slot

    def release(self, slot: int, *, state_already_reset: bool = False) -> None:
        if self._closed:
            return
        if slot not in self._leased_slots:
            return
        if not state_already_reset:
            self.reset_slots([slot])
        self._leased_slots.remove(slot)
        self._free_stream_slots.append(slot)

    def _reset_slot_ids(self, slot_ids: torch.Tensor) -> None:
        reset_streaming_slots = getattr(self._codec, "reset_decoder_state_slots", None)
        if not callable(reset_streaming_slots):
            raise RuntimeError("The streaming codec does not implement state-slot reset.")
        with torch.no_grad():
            reset_streaming_slots(slot_ids)

    def reset_slots(self, slots: list[int]) -> None:
        if not slots:
            return
        if min(slots) < 0 or max(slots) >= self._total_state_capacity:
            raise ValueError(f"Decoder state slots must be in [0, {self._total_state_capacity}), got {slots}")
        start = slots[0]
        if slots != list(range(start, start + len(slots))):
            raise ValueError(f"Decoder state slots must be contiguous, got {slots}")
        self._reset_slot_ids(self._state_slot_ids[start : start + len(slots)])

    def close(self) -> None:
        if self._closed:
            return
        close_state_pool = getattr(self._codec, "close_decoder_state_pool", None)
        with torch.no_grad():
            if callable(close_state_pool):
                close_state_pool()
        self._closed = True

    @torch.no_grad()
    def step(
        self,
        slot_codes: dict[int, torch.Tensor],
        *,
        terminal_slots: set[int] | None = None,
    ) -> dict[int, torch.Tensor]:
        if not slot_codes:
            return {}
        terminal_slots = terminal_slots or set()
        if not terminal_slots.issubset(slot_codes):
            raise ValueError("terminal_slots must be a subset of the decode batch slots.")
        step_lengths = {int(codes.shape[1]) for codes in slot_codes.values()}
        if len(step_lengths) != 1:
            raise ValueError(f"MOSS codec streaming step needs uniform T, got {sorted(step_lengths)}")
        (step_t,) = step_lengths
        slots = list(slot_codes)
        if any(slot not in self._leased_slots for slot in slots):
            raise RuntimeError(f"Streaming decode references an unleased state slot: {slots}")
        codes_step = torch.stack(
            [slot_codes[slot].to(device=self._device, dtype=torch.long) for slot in slots],
            dim=1,
        )
        state_slot_ids = torch.tensor(slots, device=self._device, dtype=torch.long)

        graph_output: tuple[torch.Tensor, torch.Tensor, int] | None = None
        if self._cudagraph_wrapper is not None:
            # A terminal tail can safely pad T to a larger graph: decoder
            # attention is causal, audio_lengths trims the padding, and every
            # affected state slot is reset immediately after this step.
            graph_output = self._cudagraph_wrapper.decode(
                codes_step,
                state_slot_ids,
                allow_frame_padding=len(terminal_slots) == len(slots),
            )

        used_cudagraph = graph_output is not None
        if used_cudagraph:
            audio_tensor, _, actual_batch_size = graph_output
            audio_tensor = audio_tensor[:actual_batch_size]
        else:
            codes_lengths = torch.full(
                (len(slots),),
                step_t,
                dtype=torch.long,
                device=self._device,
            )
            valid_rows = torch.ones(len(slots), dtype=torch.bool, device=self._device)
            result = self._codec.decode_streaming_batch(
                codes_step,
                codes_lengths,
                state_slot_ids,
                valid_rows,
            )
            if result.audio is None:
                return {}
            audio_tensor = result.audio

        if terminal_slots:
            terminal_rows = [row for row, slot in enumerate(slots) if slot in terminal_slots]
            terminal_slot_ids = state_slot_ids if len(terminal_rows) == len(slots) else state_slot_ids[terminal_rows]
            self._reset_slot_ids(terminal_slot_ids)

        audio = audio_tensor.detach().to("cpu", torch.float32)
        audio_length = step_t * self._samples_per_frame
        out: dict[int, torch.Tensor] = {}
        for row, slot in enumerate(slots):
            out[slot] = audio[row, ..., :audio_length].contiguous()
        return out


class MossTTSCodecDecoder(nn.Module):
    """Stage-1 decoder for all MOSS-TTS variants.

    Consumes ``(NQ, T)`` audio VQ codes emitted by Stage 0 and decodes them
    to a 24 kHz mono waveform using the vendored
    ``MossAudioTokenizerModel``.

    All five variants share the same codec checkpoint
    ``OpenMOSS-Team/MOSS-Audio-Tokenizer``.  The number of quantizers
    (``n_vq``) is read from ``hf_config`` at construction time and fixed for
    the lifetime of the instance; the same checkpoint can be configured as
    ``n_vq=32`` (MOSS-TTS) or ``n_vq=16`` (all other variants) without
    swapping weights.

    The codec checkpoint path comes from
    ``vllm_config.model_config.hf_config.codec_model_name_or_path``.
    """

    input_modalities = "audio"

    have_multimodal_outputs: bool = True
    has_preprocess: bool = False
    has_postprocess: bool = False
    enable_update_additional_information: bool = True
    requires_raw_input_tokens: bool = True

    _OUTPUT_SAMPLE_RATE: int = 24_000

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.vllm_config = vllm_config

        cfg = vllm_config.model_config.hf_config
        self._n_vq: int = int(getattr(cfg, "n_vq", getattr(cfg, "rvq", 16)))
        self._codec_path: str = str(
            getattr(
                cfg,
                "codec_model_name_or_path",
                getattr(cfg, "audio_tokenizer_name_or_path", "OpenMOSS-Team/MOSS-Audio-Tokenizer"),
            )
        )

        # Resolved at load_weights() time, once the codec checkpoint's own
        # config (sampling rate, channel count) is known.
        self._codec: MossAudioTokenizerModel | None = None
        self._cuda_graph_wrapper: MossTTSCUDAGraphCodecWrapper | None = None
        self._n_channels: int = 1
        self._sr_tensor = torch.tensor(self._OUTPUT_SAMPLE_RATE, dtype=torch.int32)
        self._stream_session: _MossCodecStreamSession | None = None
        scheduler_cfg = getattr(self.vllm_config, "scheduler_config", None)
        self._stream_state_capacity = max(1, int(getattr(scheduler_cfg, "max_num_seqs", 1) or 1))
        self._initial_stream_chunk_frames: int = self._connector_int("initial_codec_chunk_frames", default=0)
        self._stream_chunk_frames: int = self._connector_int("codec_chunk_frames", default=0)
        self._stream_max_step_frames: int = self._stream_chunk_frames or 100
        self._stream_req_slots: dict[str, int] = {}
        self._async_chunk = bool(getattr(self.vllm_config.model_config, "async_chunk", False))
        self._streaming_graph_batch_sizes = self._streaming_graph_batch_sizes_from_compilation_config()
        self._streaming_graph_frame_sizes = sorted(
            {frames for frames in (self._initial_stream_chunk_frames, self._stream_chunk_frames) if frames > 0}
        )

    # ------------------------------------------------------------------
    # vLLM-Omni stubs (codec has no AR loop)
    # ------------------------------------------------------------------

    def embed_input_ids(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        if input_ids.numel() == 0:
            return torch.empty((0, 1), device=input_ids.device, dtype=torch.float32)
        return torch.zeros((input_ids.shape[0], 1), device=input_ids.device, dtype=torch.float32)

    def compute_logits(
        self,
        hidden_states: torch.Tensor | OmniOutput,
        sampling_metadata: Any = None,
    ) -> None:
        return None

    # ------------------------------------------------------------------
    # Decode
    # ------------------------------------------------------------------

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        intermediate_tensors: Any = None,
        inputs_embeds: torch.Tensor | None = None,
        runtime_additional_information: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> OmniOutput:
        """Decode audio VQ codes to waveform.

        Stage 0 emits flat codebook-major ``[NQ * T_chunk]`` audio codes. The
        chunk transfer adapter assigns those to ``request.prompt_token_ids``,
        which arrives here as ``input_ids`` concatenated across all requests.
        Per-request slice boundaries are computed from
        ``kwargs["seq_token_counts"]`` (token counts, one per request).
        ``runtime_additional_information`` carries per-request metadata such as
        ``left_context_size``.

        Returns
        -------
        OmniOutput with:
          multimodal_outputs["model_outputs"] — list of (T_wav,) float32 tensors
          multimodal_outputs["sr"]            — list of scalar int32 tensors
        """
        sr_tensor = self._sr_tensor
        empty = self._empty_audio()
        info_list: list[dict[str, Any]] = list(runtime_additional_information or [{}])
        num_req = max(len(info_list), 1)

        if self._codec is None:
            logger.warning("MossTTSCodecDecoder called before load_weights(); returning silence.")
            return OmniOutput(
                text_hidden_states=None,
                multimodal_outputs={
                    "model_outputs": [empty] * num_req,
                    "sr": [sr_tensor] * num_req,
                },
            )

        if self._async_chunk and runtime_additional_information is None:
            return OmniOutput(
                text_hidden_states=None,
                multimodal_outputs={
                    "model_outputs": [empty] * num_req,
                    "sr": [sr_tensor] * num_req,
                },
            )

        audios: list[torch.Tensor] = [empty] * num_req
        srs: list[torch.Tensor] = [sr_tensor] * num_req
        device = next(self._codec.parameters()).device
        streaming_work: list[tuple[int, str, torch.Tensor, bool]] = []

        if input_ids is None or input_ids.numel() == 0:
            return OmniOutput(
                text_hidden_states=None,
                multimodal_outputs={"model_outputs": audios, "sr": srs},
            )

        # ``input_ids`` is concatenated across all requests. vLLM-Omni runners
        # pass the per-request lengths via the shared code2wav contract
        # ``seq_token_counts``.
        ids_flat = input_ids.reshape(-1).to(dtype=torch.long)
        token_counts = self._normalize_seq_token_counts(kwargs.get("seq_token_counts"))
        if token_counts is None:
            raise RuntimeError(
                "MossTTS codec requires seq_token_counts; otherwise concatenated "
                "codec tokens cannot be split per request."
            )
        real_token_count = sum(token_counts)
        input_token_count = int(ids_flat.shape[0])
        if real_token_count > input_token_count:
            raise RuntimeError(
                "MossTTS codec seq_token_counts mismatch: "
                f"counts={token_counts}, sum={real_token_count}, input_tokens={input_token_count}."
            )
        if real_token_count < input_token_count:
            ids_flat = ids_flat[:real_token_count]

        num_req = len(token_counts)
        if len(info_list) < num_req:
            info_list.extend({} for _ in range(num_req - len(info_list)))
        elif len(info_list) > num_req:
            info_list = info_list[:num_req]
        if len(audios) < num_req:
            audios.extend(empty for _ in range(num_req - len(audios)))
            srs.extend(sr_tensor for _ in range(num_req - len(srs)))
        elif len(audios) > num_req:
            audios = audios[:num_req]
            srs = srs[:num_req]

        offsets = [0]
        for n in token_counts:
            offsets.append(offsets[-1] + int(n))

        for i, info in enumerate(info_list):
            if i + 1 >= len(offsets):
                break
            seg = ids_flat[offsets[i] : offsets[i + 1]]
            if seg.numel() == 0:
                continue
            meta = (info.get("meta", {}) if isinstance(info, dict) else {}) or {}
            finished = bool(meta.get("stream_finished", meta.get("finished", False)))
            streaming_enabled = self._async_chunk
            if seg.numel() % self._n_vq != 0:
                logger.warning(
                    "MossTTS codec input length %d not divisible by n_vq %d; skipping.",
                    int(seg.numel()),
                    self._n_vq,
                )
                continue
            t_chunk = int(seg.numel() // self._n_vq)
            codes_nq_t = seg.reshape(self._n_vq, t_chunk).to(device=device)
            # Clamp out-of-range codes: the talker uses ``audio_pad_code``
            # (= ``codebook_size``) for delay-pattern padding.  The stage input
            # processor de-delays and drops pad rows before forwarding here, but
            # clamp as a defensive guard against any edge-case leakage.
            codebook_size = self._codec.config.codebook_size
            codes_nq_t = codes_nq_t.clamp_(0, int(codebook_size) - 1)

            left_ctx = meta.get("left_context_size", 0)
            if isinstance(left_ctx, (list, tuple)):
                left_ctx = int(left_ctx[0]) if left_ctx else 0
            elif isinstance(left_ctx, torch.Tensor):
                left_ctx = int(left_ctx.reshape(-1)[0].item()) if left_ctx.numel() else 0
            left_ctx = int(left_ctx)

            req_key = self._runtime_request_key(info, meta, i)

            if streaming_enabled:
                streaming_work.append((i, req_key, codes_nq_t, finished))
                continue

            if self._cuda_graph_wrapper is not None:
                out = self._cuda_graph_wrapper.decode(codes_nq_t)
            else:
                out = self._codec.batch_decode(codes_list=[codes_nq_t], num_quantizers=self._n_vq)

            if out.audio is None:
                continue

            # ``out.audio`` is ``(1, C, T)``; keep the channel axis for
            # stereo codecs (Local-v1.5) and flatten to ``(T,)`` for mono
            # ones (Delay/Realtime) to preserve their existing output shape.
            wav = out.audio[0].to(dtype=torch.float32).cpu()
            if out.audio_lengths is not None:
                wav = wav[..., : int(out.audio_lengths[0].item())]

            # Trim left-context samples (per-channel sample axis, so the
            # trim amount is identical for mono and interleaved-stereo).
            if left_ctx > 0:
                trim = min(left_ctx * self._codec.downsample_rate, wav.shape[-1])
                if trim < left_ctx * self._codec.downsample_rate:
                    logger.warning(
                        "left_ctx trim (%d samples) exceeds wav length (%d); returning empty audio.",
                        left_ctx * self._codec.downsample_rate,
                        wav.shape[-1],
                    )
                wav = wav[..., trim:]

            audios[i] = wav.reshape(-1) if wav.ndim == 1 or int(wav.shape[0]) == 1 else wav

        if streaming_work:
            for i, wav in self._decode_streaming_batch(streaming_work).items():
                audios[i] = wav.reshape(-1) if wav.ndim == 1 or int(wav.shape[0]) == 1 else wav

        return OmniOutput(
            text_hidden_states=None,
            multimodal_outputs={"model_outputs": audios, "sr": srs},
        )

    @staticmethod
    def _normalize_seq_token_counts(value: Any) -> list[int] | None:
        if value is None:
            return None
        if not isinstance(value, (list, tuple)):
            raise TypeError(
                "MossTTS codec expects seq_token_counts to be a list/tuple of per-request token counts, "
                f"got {type(value).__name__}."
            )
        counts = [int(item) for item in value]
        if not counts:
            return None
        for count in counts:
            if count < 0:
                raise ValueError(f"MossTTS codec seq_token_counts must be non-negative, got {counts}.")
        return counts

    def _runtime_request_key(self, info: Any, meta: dict[str, Any], index: int) -> str:
        for value in (
            meta.get("req_id"),
            info.get("request_id") if isinstance(info, dict) else None,
        ):
            if isinstance(value, (list, tuple)):
                value = value[0] if value else None
            if value is not None:
                return str(value)
        return f"moss-codec-stream-{index}"

    def _empty_audio(self) -> torch.Tensor:
        if self._n_channels > 1:
            return torch.zeros((self._n_channels, 0), dtype=torch.float32)
        return torch.zeros((0,), dtype=torch.float32)

    def _ensure_stream_session(self) -> _MossCodecStreamSession | None:
        if self._codec is None:
            return None
        if self._stream_session is not None:
            return self._stream_session
        self._stream_session = _MossCodecStreamSession(
            self._codec,
            state_capacity=self._stream_state_capacity,
            n_vq=self._n_vq,
            vllm_config=self.vllm_config,
            graph_batch_sizes=self._streaming_graph_batch_sizes,
            graph_frame_sizes=self._streaming_graph_frame_sizes,
        )
        return self._stream_session

    def _decode_streaming_batch(
        self,
        items: list[tuple[int, str, torch.Tensor, bool]],
    ) -> dict[int, torch.Tensor]:
        session = self._ensure_stream_session()
        if session is None:
            return {}

        outputs: dict[int, torch.Tensor] = {}
        grouped: dict[int, list[tuple[int, str, int, torch.Tensor, bool]]] = {}
        max_step_frames = max(1, int(self._stream_max_step_frames))

        for output_index, request_id, codes_nq_t, finished in items:
            slot = self._stream_req_slots.get(request_id)
            if slot is None:
                slot = session.acquire()
                if slot is None:
                    raise RuntimeError(
                        "MOSS codec state capacity exhausted. Stage-1 admission must count idle live streams "
                        f"against max_num_seqs={self._stream_state_capacity}; refusing to drop audio codes."
                    )
                self._stream_req_slots[request_id] = slot

            if int(codes_nq_t.shape[1]) > max_step_frames:
                wav = self._decode_stream_slot_sequence(
                    session,
                    slot,
                    codes_nq_t,
                    terminal=finished,
                )
                if wav is not None:
                    outputs[output_index] = wav
                if finished:
                    self._finish_stream_request(
                        request_id,
                        session,
                        slot,
                        state_already_reset=True,
                    )
                continue

            grouped.setdefault(int(codes_nq_t.shape[1]), []).append(
                (output_index, request_id, slot, codes_nq_t, finished)
            )

        for group in grouped.values():
            plan = {slot: codes_nq_t for _, _, slot, codes_nq_t, _ in group}
            terminal_slots = {slot for _, _, slot, _, finished in group if finished}
            decoded = session.step(plan, terminal_slots=terminal_slots)
            for output_index, request_id, slot, _, finished in group:
                wav = decoded.get(slot)
                if wav is not None:
                    outputs[output_index] = wav
                if finished:
                    self._finish_stream_request(
                        request_id,
                        session,
                        slot,
                        state_already_reset=True,
                    )

        return outputs

    def _decode_stream_slot_sequence(
        self,
        session: _MossCodecStreamSession,
        slot: int,
        codes_nq_t: torch.Tensor,
        *,
        terminal: bool = False,
    ) -> torch.Tensor | None:
        if codes_nq_t.numel() == 0:
            return None
        max_step_frames = max(1, int(self._stream_max_step_frames))
        parts: list[torch.Tensor] = []
        for start in range(0, int(codes_nq_t.shape[1]), max_step_frames):
            chunk = codes_nq_t[:, start : start + max_step_frames]
            is_terminal_chunk = terminal and start + max_step_frames >= int(codes_nq_t.shape[1])
            decoded = session.step(
                {slot: chunk},
                terminal_slots={slot} if is_terminal_chunk else None,
            )
            wav = decoded.get(slot)
            if wav is not None:
                parts.append(wav)
        if not parts:
            return None
        return torch.cat(parts, dim=-1) if len(parts) > 1 else parts[0]

    def _finish_stream_request(
        self,
        request_id: str,
        session: _MossCodecStreamSession,
        slot: int | None,
        *,
        state_already_reset: bool = False,
    ) -> None:
        if slot is not None:
            session.release(slot, state_already_reset=state_already_reset)
        self._stream_req_slots.pop(request_id, None)

    def on_requests_finished(self, finished_req_ids: set[str] | list[str]) -> None:
        """Release codec streaming slots when requests finish outside payload flow.

        Normal streaming completion releases slots from ``_decode_streaming_batch``
        when the Stage-0 payload carries ``finished=True``. Client disconnects
        and engine-side aborts can finish a request without delivering that
        terminal payload, so the runner calls this hook from its finished-request
        path to avoid leaking stream slots and buffered codes.
        """
        session = self._stream_session
        for req_id in finished_req_ids:
            request_id = str(req_id)
            slot = self._stream_req_slots.get(request_id)
            if slot is None:
                continue
            if session is not None:
                self._finish_stream_request(request_id, session, slot)
            else:
                self._stream_req_slots.pop(request_id, None)

    def _connector_int(self, name: str, default: int = 0) -> int:
        model_cfg = getattr(self.vllm_config, "model_config", None)
        connector_cfg = getattr(model_cfg, "stage_connector_config", None)
        if isinstance(connector_cfg, dict):
            extra_cfg: dict | None = connector_cfg.get("extra", connector_cfg)
        else:
            extra_cfg = getattr(connector_cfg, "extra", None)
        if isinstance(extra_cfg, dict) and name in extra_cfg:
            return int(extra_cfg[name])
        return default

    def _streaming_graph_batch_sizes_from_compilation_config(self) -> list[int]:
        if getattr(self.vllm_config.model_config, "enforce_eager", True):
            return []
        compilation_config = getattr(self.vllm_config, "compilation_config", None)
        capture_sizes = getattr(compilation_config, "cudagraph_capture_sizes", None)
        if capture_sizes:
            return sorted({int(size) for size in capture_sizes if 0 < int(size) <= self._stream_state_capacity})
        return []

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Drain the Stage-0 weights iterator, then load the codec from its own checkpoint.

        The codec lives in a separate HuggingFace repo
        (``OpenMOSS-Team/MOSS-Audio-Tokenizer``) and is loaded independently
        of the talker weights.
        """
        # Drain the incoming weights iterator — all Stage-0 weights are
        # irrelevant to this stage.
        for _ in weights:
            pass

        codec_path = self._codec_path
        device = self.vllm_config.device_config.device
        logger.info("Loading MOSS Audio Tokenizer from %s directly on %s", codec_path, device)

        # This codec comes from a secondary checkpoint, so it cannot be built
        # by the outer stage's normal initialize_model() call. Re-enter the
        # same target-device/default-dtype contexts used by vLLM's
        # BaseModelLoader.load_model() instead of constructing an 8 GiB FP32
        # model on CPU and copying the whole module to the GPU afterwards.
        with set_default_torch_dtype(torch.float32):
            with device:
                codec_cfg, codec = self._build_codec(codec_path)

        model_loader = DefaultModelLoader(self.vllm_config.load_config)
        source = DefaultModelLoader.Source(
            model_or_path=codec_path,
            revision=None,
            subfolder=None,
        )
        codec_weights = model_loader._get_weights_iterator(source)
        params_dict = dict(codec.named_parameters())

        # Upstream MossAudioTokenizer uses different submodule names than the
        # vendored re-implementation in ``audio_tokenizer.py``. Without this
        # remap only ~half the codec parameters load (codebooks + WN convs)
        # and the rest stay at their random init, which produces noise that
        # sounds correct in duration but is structurally garbage.
        _SUFFIX_REMAP: list[tuple[str, str]] = [
            # v1 (MOSS-Audio-Tokenizer) naming.
            (".self_attn.in_projs.0.", ".attn.in_proj."),
            (".self_attn.out_projs.0.", ".attn.out_proj."),
            (".linear1.", ".ff1."),
            (".linear2.", ".ff2."),
            # v2 checkpoint names use singular in_proj/out_proj and ffn.{0,2};
            # the vendored module keeps the original MOSS layer names.
            (".self_attn.in_proj.", ".self_attn.in_projs.0."),
            (".self_attn.out_proj.", ".self_attn.out_projs.0."),
            (".ffn.0.", ".linear1."),
            (".ffn.2.", ".linear2."),
            (".layer_scale_1.", ".ls1."),
            (".layer_scale_2.", ".ls2."),
            (".input_proj.", ".in_proj."),
            (".output_proj.", ".out_proj."),
        ]

        def _remap(name: str) -> str:
            for src, dst in _SUFFIX_REMAP:
                if src in name:
                    return name.replace(src, dst)
            return name

        loaded_names: set[str] = set()
        skipped: list[str] = []
        shape_mismatches: list[tuple[str, str, tuple[int, ...], tuple[int, ...]]] = []
        for name, tensor in codec_weights:
            # Try direct name first (e.g. ``quantizer.input_proj.*`` exists
            # under the same name in both layouts), then the remap (transformer
            # submodules need ``.linear1.``→``.ff1.`` etc.).
            tgt = name if name in params_dict else _remap(name)
            if tgt in params_dict:
                expected_shape = tuple(params_dict[tgt].shape)
                actual_shape = tuple(tensor.shape)
                if expected_shape != actual_shape:
                    shape_mismatches.append((name, tgt, actual_shape, expected_shape))
                    continue
                default_weight_loader(params_dict[tgt], tensor)
                loaded_names.add(tgt)
            else:
                skipped.append(name)

        missing = sorted(set(params_dict) - loaded_names)
        if missing or skipped or shape_mismatches:
            raise RuntimeError(
                "MOSS Audio Tokenizer weights were not fully loaded: "
                f"loaded={len(loaded_names)}/{len(params_dict)} "
                f"missing={len(missing)} skipped={len(skipped)} "
                f"shape_mismatches={len(shape_mismatches)}; "
                f"first_missing={missing[:5]} "
                f"first_skipped={skipped[:5]} "
                f"first_shape_mismatches={shape_mismatches[:3]}"
            )
        logger.info(
            "MOSS Audio Tokenizer weights: loaded=%d/%d skipped=%d (first skipped: %s)",
            len(loaded_names),
            len(params_dict),
            len(skipped),
            skipped[:3] if skipped else "none",
        )

        codec.eval()
        if device.type != "cpu":
            codec.decoder.to(dtype=torch.bfloat16)
        build_decode_lut = getattr(codec.quantizer, "build_decode_lut", None)
        if callable(build_decode_lut):
            lut_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
            build_decode_lut(self._n_vq, dtype=lut_dtype)
            lut = codec.quantizer._decode_lut
            logger.info(
                "MOSS Audio Tokenizer LFQ decoded LUT: shape=%s dtype=%s size=%.1f MiB",
                tuple(lut.shape),
                lut.dtype,
                lut.numel() * lut.element_size() / (1024**2),
            )
        self._codec = codec
        inferred_channels = 2 if "v2" in codec_path.lower() else 1
        self._n_channels = int(
            getattr(
                codec_cfg,
                "number_channels",
                getattr(codec_cfg, "num_channels", inferred_channels),
            )
            or inferred_channels
        )
        self._sr_tensor = torch.tensor(int(codec_cfg.sampling_rate), dtype=torch.int32)

        logger.info(
            "MOSS Audio Tokenizer loaded: sampling_rate=%d, n_vq=%d, n_channels=%d",
            codec_cfg.sampling_rate,
            codec_cfg.num_quantizers,
            self._n_channels,
        )

        self._configure_decoder_cudagraph(device)
        if self._async_chunk and self._streaming_graph_batch_sizes and self._streaming_graph_frame_sizes:
            self._ensure_stream_session()

        # vLLM's track_weights_loading() compares the returned set against
        # ``self.named_parameters()``. After ``self._codec = codec`` above,
        # those parameters are registered with the ``_codec.`` prefix, so
        # mirror that here.
        return {f"_codec.{name}" for name, _ in codec.named_parameters()}

    def _build_codec(self, codec_path: str) -> tuple[Any, nn.Module]:
        config_dict, _ = MossAudioTokenizerV2Config.get_config_dict(codec_path)
        is_v2 = config_dict.get("number_channels", 1) >= 2

        if is_v2:
            try:
                codec_cfg = MossAudioTokenizerV2Config.from_pretrained(codec_path)
                codec = MossAudioTokenizerV2Model(codec_cfg)
                logger.info("Using vendored MOSS Audio Tokenizer v2 classes from %s", codec_path)
                return codec_cfg, codec
            except Exception:
                logger.exception(
                    "Failed to instantiate vendored MOSS Audio Tokenizer v2; falling back to legacy vendored codec."
                )

        codec_cfg = MossAudioTokenizerConfig.from_pretrained(codec_path)
        codec = MossAudioTokenizerModel(codec_cfg)
        logger.info("Using vendored MOSS Audio Tokenizer v1 classes from %s", codec_path)
        return codec_cfg, codec

    def _configure_decoder_cudagraph(self, device: torch.device) -> None:
        """Select the codec CUDA Graph path.

        ``enforce_eager`` is the single graph on/off switch. If graphing is
        enabled, ``async_chunk`` decides whether decode uses the persistent
        streaming-state wrapper or the offline full-chunk wrapper.
        """
        if getattr(self.vllm_config.model_config, "enforce_eager", True):
            self._streaming_graph_batch_sizes = []
            return
        if self._codec is None:
            return
        if self._async_chunk:
            logger.info(
                "MOSS-TTS codec CUDA Graph selected streaming wrapper: B=%s exact_T=%s",
                self._streaming_graph_batch_sizes,
                self._streaming_graph_frame_sizes,
            )
            return

        self._enable_non_streaming_decoder_cudagraph(device)

    def _enable_non_streaming_decoder_cudagraph(self, device: torch.device) -> None:
        if self._codec is None:
            return

        # Read capture sizes from the connector's extra config (same convention
        # as Qwen3-TTS), falling back to a sensible default covering common
        # codec_chunk_frames values used in moss_tts.yaml.
        capture_sizes: list[int] = [4, 8, 16, 25, 32, 50, 64, 100, 128, 200, 256]
        model_cfg = getattr(self.vllm_config, "model_config", None)
        connector_cfg = getattr(model_cfg, "stage_connector_config", None)
        if isinstance(connector_cfg, dict):
            extra_cfg: dict | None = connector_cfg.get("extra", connector_cfg)
        else:
            extra_cfg = getattr(connector_cfg, "extra", None)
        if isinstance(extra_cfg, dict):
            raw = extra_cfg.get("decode_cudagraph_capture_sizes")
            if raw is not None:
                if isinstance(raw, (list, tuple)):
                    parsed = sorted({int(v) for v in raw if int(v) > 0})
                elif isinstance(raw, str):
                    parsed = sorted({int(v.strip()) for v in raw.split(",") if v.strip()})
                else:
                    parsed = [int(raw)]
                if parsed:
                    capture_sizes = parsed

        self._cuda_graph_wrapper = MossTTSCUDAGraphCodecWrapper(
            model=self._codec,
            capture_sizes=capture_sizes,
            num_quantizers=self._n_vq,
            enabled=True,
        )
        self._cuda_graph_wrapper.warmup(device)


__all__ = ["MossTTSCodecDecoder"]
