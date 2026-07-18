# Copyright 2026 OpenMOSS and the vLLM-Omni team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
"""MOSS-TTS Stage-1 codec decoder: RVQ codes → 24 kHz waveform."""

from __future__ import annotations

import contextlib
from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader import DefaultModelLoader
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

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
    """Persistent streaming decode session for vendored MOSS-Audio-Tokenizer-v2."""

    def __init__(
        self,
        codec: nn.Module,
        *,
        stream_slots: int,
        n_vq: int,
        cudagraph_capture_sizes: list[int] | None = None,
    ) -> None:
        self._codec = codec
        self._stream_slots = int(stream_slots)
        self._batch_size = self._stream_slots
        self._n_vq = int(n_vq)
        self._device = next(codec.parameters()).device
        self._free_stream_slots = list(range(self._stream_slots))
        self._exit_stack = contextlib.ExitStack()
        self._closed = False
        with torch.no_grad():
            self._exit_stack.enter_context(codec.streaming(self._batch_size))
        self._cudagraph_wrapper: CUDAGraphStreamingDecoderWrapper | None = None
        capture_sizes = sorted({int(size) for size in (cudagraph_capture_sizes or []) if int(size) > 0})
        if capture_sizes and self._device.type == "cuda":
            self._cudagraph_wrapper = CUDAGraphStreamingDecoderWrapper(
                codec,
                batch_size=self._batch_size,
                num_quantizers=self._n_vq,
                reset_streaming_state=lambda: self.reset_slots(list(range(self._batch_size))),
            )
            self._cudagraph_wrapper.warmup(self._device, capture_sizes)
            self.reset_slots(list(range(self._batch_size)))
            if not self._cudagraph_wrapper.is_ready:
                self._cudagraph_wrapper = None

    def acquire(self) -> int | None:
        if not self._free_stream_slots:
            return None
        return self._free_stream_slots.pop()

    def release(self, slot: int) -> None:
        if self._closed:
            return
        self.reset_slots([slot])
        self._free_stream_slots.append(slot)

    def reset_slots(self, slots: list[int]) -> None:
        if not slots:
            return
        reset_mask = torch.zeros(self._batch_size, dtype=torch.bool, device=self._device)
        reset_mask[slots] = True

        def _reset(module: nn.Module) -> None:
            state = getattr(module, "_streaming_state", None)
            if state is not None:
                state.reset(reset_mask.to(state.device))

        with torch.no_grad():
            self._codec.apply(_reset)

    def close(self) -> None:
        if self._closed:
            return
        with torch.no_grad():
            self._exit_stack.close()
        self._closed = True

    @torch.no_grad()
    def step(self, slot_codes: dict[int, torch.Tensor]) -> dict[int, torch.Tensor]:
        if not slot_codes:
            return {}
        step_lengths = {int(codes.shape[1]) for codes in slot_codes.values()}
        if len(step_lengths) != 1:
            raise ValueError(f"MOSS codec streaming step needs uniform T, got {sorted(step_lengths)}")
        (step_t,) = step_lengths
        codes_step = torch.zeros(
            self._n_vq,
            self._batch_size,
            step_t,
            dtype=torch.long,
            device=self._device,
        )
        codes_lengths = torch.zeros(self._batch_size, dtype=torch.long, device=self._device)
        exec_mask = torch.zeros(self._batch_size, dtype=torch.bool, device=self._device)
        for slot, codes in slot_codes.items():
            codes_step[:, slot, :] = codes.to(device=self._device, dtype=torch.long)
            codes_lengths[slot] = int(codes.shape[1])
            exec_mask[slot] = True

        graph_output: tuple[torch.Tensor, torch.Tensor] | None = None
        if self._cudagraph_wrapper is not None:
            graph_output = self._cudagraph_wrapper.decode(codes_step, exec_mask)

        used_cudagraph = graph_output is not None
        if used_cudagraph:
            audio_tensor, lengths_tensor = graph_output
        else:
            self._codec._set_streaming_exec_mask(exec_mask)
            result = self._codec._decode_frame(codes_step, codes_lengths)
            if result.audio is None:
                return {}
            audio_tensor = result.audio
            lengths_tensor = result.audio_lengths

        audio = audio_tensor.detach().to("cpu", torch.float32)
        lengths = lengths_tensor.detach().to("cpu") if lengths_tensor is not None else None
        out: dict[int, torch.Tensor] = {}
        for slot in slot_codes:
            wav = audio[slot]
            if lengths is not None:
                wav = wav[..., : int(lengths[slot].item())]
            out[slot] = wav.contiguous()
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
        self._stream_slots: int = self._connector_int("codec_stream_slots", default=0)
        self._stream_chunk_frames: int = self._connector_int("codec_chunk_frames", default=0)
        self._stream_max_step_frames: int = self._stream_chunk_frames or 100
        self._stream_req_slots: dict[str, int] = {}
        self._stream_pending_codes: dict[str, list[torch.Tensor]] = {}
        self._stream_starved_reqs: set[str] = set()
        self._codec_streaming: bool = self._connector_bool("codec_streaming", default=False)
        default_graph_capture_max = self._stream_chunk_frames or min(self._stream_max_step_frames, 32)
        default_graph_capture_sizes = list(range(1, max(1, default_graph_capture_max) + 1))
        self._streaming_cudagraph_capture_sizes = self._streaming_cudagraph_capture_sizes_from_compilation_config(
            default_graph_capture_sizes
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

        if self._codec_streaming and runtime_additional_information is None:
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
            for i, wav in self._finish_empty_streaming_requests(info_list).items():
                audios[i] = wav.reshape(-1) if wav.ndim == 1 or int(wav.shape[0]) == 1 else wav
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
        if sum(token_counts) != int(ids_flat.shape[0]):
            raise RuntimeError(
                "MossTTS codec seq_token_counts mismatch: "
                f"counts={token_counts}, sum={sum(token_counts)}, input_tokens={int(ids_flat.shape[0])}."
            )

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
            streaming_enabled = bool(meta.get("codec_streaming", self._codec_streaming))
            code_flat_numel = meta.get("code_flat_numel")
            if streaming_enabled and finished and code_flat_numel is not None and int(code_flat_numel) == 0:
                for _, wav in self._finish_empty_streaming_requests([info]).items():
                    audios[i] = wav.reshape(-1) if wav.ndim == 1 or int(wav.shape[0]) == 1 else wav
                continue
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

    def _finish_empty_streaming_requests(self, info_list: list[dict[str, Any]]) -> dict[int, torch.Tensor]:
        """Release codec stream state for empty finish sentinels.

        Stage-0 can finish on a step that emits no new audio frame. The stage
        input processor forwards that as an empty payload with finished=true.
        If a request never acquired a stream slot, do not offline-decode its
        buffered codes here: streaming requests must only emit client deltas.
        """
        session = self._stream_session
        outputs: dict[int, torch.Tensor] = {}
        if session is None:
            return outputs
        for i, info in enumerate(info_list):
            if not isinstance(info, dict):
                continue
            meta = (info.get("meta", {}) or {}) if isinstance(info.get("meta", {}), dict) else {}
            if not bool(meta.get("codec_streaming", self._codec_streaming)):
                continue
            finished = bool(meta.get("stream_finished", meta.get("finished", False)))
            if not finished:
                continue
            req_key = self._runtime_request_key(info, meta, i)
            slot = self._stream_req_slots.get(req_key)
            pending = req_key in self._stream_pending_codes
            if slot is not None or pending:
                if pending and slot is None:
                    logger.warning(
                        "MOSS codec stream request %s finished before a codec stream slot became available; "
                        "dropping buffered codes instead of offline-decoding a non-delta waveform.",
                        req_key,
                    )
                self._finish_stream_request(req_key, session, slot)
        return outputs

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
        slots = self._stream_slots
        if slots <= 0:
            scheduler_cfg = getattr(self.vllm_config, "scheduler_config", None)
            slots = int(getattr(scheduler_cfg, "max_num_seqs", 1) or 1)
        self._stream_session = _MossCodecStreamSession(
            self._codec,
            stream_slots=max(1, slots),
            n_vq=self._n_vq,
            cudagraph_capture_sizes=self._streaming_cudagraph_capture_sizes,
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
            pending = self._stream_pending_codes.get(request_id)
            slot = self._stream_req_slots.get(request_id)
            if slot is None:
                slot = session.acquire()
                if slot is None:
                    self._append_stream_pending(request_id, codes_nq_t)
                    if request_id not in self._stream_starved_reqs:
                        logger.warning(
                            "MOSS codec streaming slots exhausted; buffering %s until a stream slot is available.",
                            request_id,
                        )
                        self._stream_starved_reqs.add(request_id)
                    if finished:
                        logger.warning(
                            "MOSS codec stream request %s finished before a codec stream slot became available; "
                            "dropping buffered codes instead of offline-decoding a non-delta waveform.",
                            request_id,
                        )
                        self._finish_stream_request(request_id, session, None)
                    continue
                self._stream_req_slots[request_id] = slot

            if pending:
                self._append_stream_pending(request_id, codes_nq_t)
                replay_codes = self._pop_stream_pending(request_id)
                wav = self._decode_stream_slot_sequence(session, slot, replay_codes)
                if wav is not None:
                    outputs[output_index] = wav
                if finished:
                    self._finish_stream_request(request_id, session, slot)
                continue

            if int(codes_nq_t.shape[1]) > max_step_frames:
                wav = self._decode_stream_slot_sequence(session, slot, codes_nq_t)
                if wav is not None:
                    outputs[output_index] = wav
                if finished:
                    self._finish_stream_request(request_id, session, slot)
                continue

            grouped.setdefault(int(codes_nq_t.shape[1]), []).append(
                (output_index, request_id, slot, codes_nq_t, finished)
            )

        for group in grouped.values():
            plan = {slot: codes_nq_t for _, _, slot, codes_nq_t, _ in group}
            decoded = session.step(plan)
            for output_index, request_id, slot, _, finished in group:
                wav = decoded.get(slot)
                if wav is not None:
                    outputs[output_index] = wav
                if finished:
                    self._finish_stream_request(request_id, session, slot)

        return outputs

    def _append_stream_pending(self, request_id: str, codes_nq_t: torch.Tensor) -> None:
        self._stream_pending_codes.setdefault(request_id, []).append(
            codes_nq_t.detach().to("cpu", torch.long).contiguous()
        )

    def _pop_stream_pending(self, request_id: str) -> torch.Tensor:
        pending = self._stream_pending_codes.pop(request_id, [])
        if not pending:
            return torch.empty((self._n_vq, 0), dtype=torch.long)
        return torch.cat(pending, dim=1).contiguous()

    def _decode_stream_slot_sequence(
        self,
        session: _MossCodecStreamSession,
        slot: int,
        codes_nq_t: torch.Tensor,
    ) -> torch.Tensor | None:
        if codes_nq_t.numel() == 0:
            return None
        max_step_frames = max(1, int(self._stream_max_step_frames))
        parts: list[torch.Tensor] = []
        for start in range(0, int(codes_nq_t.shape[1]), max_step_frames):
            chunk = codes_nq_t[:, start : start + max_step_frames]
            decoded = session.step({slot: chunk})
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
    ) -> None:
        if slot is not None:
            session.release(slot)
        self._stream_req_slots.pop(request_id, None)
        self._stream_pending_codes.pop(request_id, None)
        self._stream_starved_reqs.discard(request_id)

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
            has_state = (
                slot is not None or request_id in self._stream_pending_codes or request_id in self._stream_starved_reqs
            )
            if not has_state:
                continue
            if session is not None:
                self._finish_stream_request(request_id, session, slot)
            else:
                self._stream_req_slots.pop(request_id, None)
                self._stream_pending_codes.pop(request_id, None)
                self._stream_starved_reqs.discard(request_id)

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

    def _connector_bool(self, name: str, default: bool = False) -> bool:
        model_cfg = getattr(self.vllm_config, "model_config", None)
        connector_cfg = getattr(model_cfg, "stage_connector_config", None)
        if isinstance(connector_cfg, dict):
            extra_cfg: dict | None = connector_cfg.get("extra", connector_cfg)
        else:
            extra_cfg = getattr(connector_cfg, "extra", None)
        if isinstance(extra_cfg, dict) and name in extra_cfg:
            return bool(extra_cfg[name])
        return default

    def _streaming_cudagraph_capture_sizes_from_compilation_config(
        self,
        default: list[int],
    ) -> list[int]:
        if getattr(self.vllm_config.model_config, "enforce_eager", True):
            return []
        compilation_config = getattr(self.vllm_config, "compilation_config", None)
        capture_sizes = getattr(compilation_config, "cudagraph_capture_sizes", None)
        if capture_sizes:
            max_step_frames = max(1, int(self._stream_max_step_frames))
            return sorted({int(size) for size in capture_sizes if 0 < int(size) <= max_step_frames})
        return list(default)

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
        logger.info("Loading MOSS Audio Tokenizer from %s", codec_path)

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

        device = self.vllm_config.device_config.device
        codec.to(device=device, dtype=torch.float32)
        codec.eval()
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
        if self._codec_streaming and self._streaming_cudagraph_capture_sizes:
            self._ensure_stream_session()

        # vLLM's track_weights_loading() compares the returned set against
        # ``self.named_parameters()``. After ``self._codec = codec`` above,
        # those parameters are registered with the ``_codec.`` prefix, so
        # mirror that here.
        return {f"_codec.{name}" for name, _ in codec.named_parameters()}

    def _build_codec(self, codec_path: str) -> tuple[Any, nn.Module]:
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
        return codec_cfg, codec

    def _configure_decoder_cudagraph(self, device: torch.device) -> None:
        """Select the codec CUDA Graph path.

        ``enforce_eager`` is the single graph on/off switch. If graphing is
        enabled, ``codec_streaming`` decides whether decode uses the persistent
        streaming-state wrapper or the offline full-chunk wrapper.
        """
        if getattr(self.vllm_config.model_config, "enforce_eager", True):
            self._streaming_cudagraph_capture_sizes = []
            return
        if self._codec is None:
            return
        if self._codec_streaming:
            logger.info(
                "MOSS-TTS codec CUDA Graph selected streaming wrapper: capture_sizes=%s",
                self._streaming_cudagraph_capture_sizes,
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
