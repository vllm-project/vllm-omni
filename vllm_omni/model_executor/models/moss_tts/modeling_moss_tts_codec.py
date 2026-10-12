# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# Copyright 2026 OpenMOSS and the vLLM-Omni team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
"""MOSS-TTS Stage-1 codec decoder: RVQ codes → 24 kHz waveform."""

from __future__ import annotations

from collections.abc import Iterable
from contextlib import nullcontext
from functools import partial
from typing import Any

import torch
import torch.nn as nn
from transformers import PretrainedConfig
from vllm.config import LoadConfig, VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader import DefaultModelLoader
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.utils.torch_utils import set_default_torch_dtype

from vllm_omni.data_entry_keys import FIRST_AUDIO_REQUIRED_KEY
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
from vllm_omni.model_executor.output_snapshot import PackedOutputSnapshot, pack_output_snapshot
from vllm_omni.model_executor.stage_input_processors.chunk_size_utils import parse_chunk_ramp

logger = init_logger(__name__)


def load_codec(
    codec_path: str,
    *,
    device: torch.device,
    load_config: LoadConfig,
    num_quantizers: int,
    attention_backend: str = "sdpa",
    skip_empty_tiles: bool = False,
    fused_slots: bool = False,
) -> tuple[PretrainedConfig, nn.Module]:
    """Load and prepare a codec without creating stream state or CUDA graphs."""
    logger.info("Loading MOSS Audio Tokenizer from %s directly on %s", codec_path, device)

    # This codec comes from a secondary checkpoint, so it cannot be built
    # by the outer stage's normal initialize_model() call. Re-enter the
    # same target-device/default-dtype contexts used by vLLM's
    # BaseModelLoader.load_model() instead of constructing an 8 GiB FP32
    # model on CPU and copying the whole module to the GPU afterwards.
    with set_default_torch_dtype(torch.float32):
        with device:
            codec_cfg, codec = _build_codec(codec_path)

    model_loader = DefaultModelLoader(load_config)
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
    # The v1 quantizer emits FP32 tensors, so its decoder must remain FP32.
    if device.type != "cpu" and isinstance(codec, MossAudioTokenizerV2Model):
        codec.decoder.to(dtype=torch.bfloat16)
    if attention_backend != "sdpa":
        if attention_backend not in {"triton", "triton_slot"} or device.type != "cuda":
            raise ValueError(f"Unsupported codec attention backend/device: {attention_backend}/{device.type}")
        from vllm_omni.model_executor.models.moss_tts.audio_tokenizer_v2 import MossAudioTokenizerMultiheadAttention
        from vllm_omni.model_executor.models.moss_tts.streaming_attention import masked_attention

        if attention_backend == "triton_slot":
            from vllm_omni.model_executor.models.moss_tts.slot_attention import (
                slot_ring_attention,
                slot_ring_attention_rows,
            )

            if skip_empty_tiles and fused_slots:
                raise ValueError("codec_skip_empty_attention_tiles requires unfused slot attention")
            if skip_empty_tiles:
                slot_ring_attention = partial(slot_ring_attention, skip_empty_tiles=True)
                logger.info("MOSS codec slot attention: skipping empty tiles for T <= 32")

        for module in codec.decoder.modules():
            if isinstance(module, MossAudioTokenizerMultiheadAttention):
                module._streaming_attention = masked_attention
                if attention_backend == "triton_slot":
                    module._slot_attention = slot_ring_attention
                    if fused_slots:
                        module._slot_attention_rows = slot_ring_attention_rows
        logger.info("Enabled codec attention backend=%s", attention_backend)
    build_decode_lut = getattr(codec.quantizer, "build_decode_lut", None)
    if callable(build_decode_lut):
        lut_dtype = torch.bfloat16 if device.type in ("cuda", "npu") else torch.float32
        build_decode_lut(num_quantizers, dtype=lut_dtype)
        lut = codec.quantizer._decode_lut
        logger.info(
            "MOSS Audio Tokenizer LFQ decoded LUT: shape=%s dtype=%s size=%.1f MiB",
            tuple(lut.shape),
            lut.dtype,
            lut.numel() * lut.element_size() / (1024**2),
        )
    return codec_cfg, codec


def _build_codec(codec_path: str) -> tuple[PretrainedConfig, nn.Module]:
    config_dict, _ = MossAudioTokenizerV2Config.get_config_dict(codec_path)
    is_v2 = config_dict.get("number_channels", 1) >= 2

    if is_v2:
        codec_cfg = MossAudioTokenizerV2Config.from_pretrained(codec_path)
        codec = MossAudioTokenizerV2Model(codec_cfg)
        logger.info("Using vendored MOSS Audio Tokenizer v2 classes from %s", codec_path)
        return codec_cfg, codec

    codec_cfg = MossAudioTokenizerConfig.from_pretrained(codec_path)
    codec = MossAudioTokenizerModel(codec_cfg)
    logger.info("Using vendored MOSS Audio Tokenizer v1 classes from %s", codec_path)
    return codec_cfg, codec


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
        gpu_output: bool = False,
        chunk_frames: int = 0,
        ring_headroom: bool = False,
    ) -> None:
        self._codec = codec
        self._state_capacity = int(state_capacity)
        self._chunk_frames = int(chunk_frames)
        self._ring_headroom = bool(ring_headroom)
        self._n_vq = int(n_vq)
        self._device = next(codec.parameters()).device
        self._gpu_output = gpu_output and self._device.type == "cuda"
        self._free_stream_slots = list(reversed(range(self._state_capacity)))
        self._leased_slots: set[int] = set()
        self._closed = False
        self._metadata_index = 0
        self._metadata_ring = []
        if self._gpu_output:
            # The metadata DMA is tiny, but a short ring forces the producer
            # CPU thread to synchronize with the codec stream while it waits
            # for a host slab to become reusable.  Keep enough entries for a
            # scheduler burst so staging remains enqueue-only in practice.
            for _ in range(64):
                host = torch.empty(2 * self._state_capacity, dtype=torch.long, pin_memory=True)
                gpu = torch.empty_like(host, device=self._device)
                self._metadata_ring.append((host, host.numpy(), gpu, torch.cuda.Event(), False))
        self._cudagraph_wrapper: CUDAGraphStreamingDecoderWrapper | None = None
        batch_sizes = sorted({int(size) for size in (graph_batch_sizes or []) if 0 < int(size) <= self._state_capacity})
        frame_sizes = sorted({int(size) for size in (graph_frame_sizes or []) if int(size) > 0})
        scratch_capacity = max(batch_sizes, default=0) if self._device.type in ("cuda", "npu") else 0
        if isinstance(codec, MossAudioTokenizerV2Model):
            scratch_capacity = 1
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
            if self._ring_headroom:
                if self._chunk_frames <= 0:
                    raise ValueError("codec_ring_headroom requires a positive codec chunk frame count.")
                # Size every attention ring for context + one full chunk so a
                # chunk write never evicts keys inside its own causal window.
                initialize_state_pool(self._state_capacity, scratch_capacity, chunk_frames=self._chunk_frames)
            else:
                initialize_state_pool(self._state_capacity, scratch_capacity)
        logger.info(
            "MOSS codec decoder state pool: live=%d padding=%d shared_metadata=%s",
            self._state_capacity,
            scratch_capacity,
            getattr(codec, "_decoder_null_slot", None) is not None,
        )
        if batch_sizes and frame_sizes and self._device.type in ("cuda", "npu"):
            if self._device.type == "npu":
                from vllm_omni.platforms.npu.models.moss_tts_streaming_decode_wrapper import (
                    NPUGraphStreamingDecoderWrapper,
                )

                self._cudagraph_wrapper = NPUGraphStreamingDecoderWrapper(
                    codec,
                    state_capacity=self._state_capacity,
                    batch_sizes=batch_sizes,
                    frame_sizes=frame_sizes,
                    num_quantizers=self._n_vq,
                    vllm_config=vllm_config,
                )
            else:
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

    def acquire(self) -> int | None:
        if not self._free_stream_slots:
            return None
        slot = self._free_stream_slots.pop()
        self._leased_slots.add(slot)
        return slot

    def release(self, slot: int, *, state_already_reset: bool = False) -> None:
        if self._closed or slot not in self._leased_slots:
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

    def _stage_slot_ids(self, slots: list[int], terminal_slots: set[int]) -> tuple[torch.Tensor, torch.Tensor]:
        # Host slabs cannot be overwritten until their DMA completes. Device
        # slabs are reused in producer-stream order, after all previous reads.
        index = self._metadata_index
        host, view, gpu, event, recorded = self._metadata_ring[index]
        if recorded and not event.query():
            event.synchronize()
        terminal_ids = [slot for slot in slots if slot in terminal_slots]
        n, end = len(slots), len(slots) + len(terminal_ids)
        view[:n] = slots
        view[n:end] = terminal_ids
        gpu[:end].copy_(host[:end], non_blocking=True)
        event.record(torch.cuda.current_stream(self._device))
        self._metadata_ring[index] = (host, view, gpu, event, True)
        self._metadata_index = (index + 1) % len(self._metadata_ring)
        return gpu[:n], gpu[n:end]

    def close(self) -> None:
        if self._closed:
            return
        for _, _, _, event, recorded in self._metadata_ring:
            if recorded:
                event.synchronize()
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
        output_buffers: dict[int, torch.Tensor] | None = None,
    ) -> dict[int, torch.Tensor]:
        if not slot_codes:
            return {}
        terminal_slots = terminal_slots or set()
        if not terminal_slots.issubset(slot_codes):
            raise ValueError("terminal_slots must be a subset of the decode batch slots.")
        step_t = max(int(codes.shape[1]) for codes in slot_codes.values())
        if any(int(codes.shape[1]) != step_t and slot not in terminal_slots for slot, codes in slot_codes.items()):
            raise ValueError("Only terminal codec rows may be padded to a larger step length")
        slots = list(slot_codes)
        if any(slot not in self._leased_slots for slot in slots):
            raise RuntimeError(f"Streaming decode references an unleased state slot: {slots}")
        codes_step = torch.stack(
            [
                torch.nn.functional.pad(
                    slot_codes[slot].to(device=self._device, dtype=torch.long),
                    (0, step_t - int(slot_codes[slot].shape[1])),
                )
                if int(slot_codes[slot].shape[1]) != step_t
                else slot_codes[slot].to(device=self._device, dtype=torch.long)
                for slot in slots
            ],
            dim=1,
        )
        staged_terminal_ids = None
        if self._metadata_ring:
            state_slot_ids, staged_terminal_ids = self._stage_slot_ids(slots, terminal_slots)
        else:
            state_slot_ids = torch.tensor(slots, device=self._device, dtype=torch.long)

        graph_output: tuple[torch.Tensor, torch.Tensor, int] | None = None
        if self._cudagraph_wrapper is not None:
            # An all-terminal group may borrow the next larger frame graph.
            # The decoder is causal, so cropping the padded tail on the time
            # axis recovers the exact-T waveform up to attention-ring
            # eviction (see codec_ring_headroom); the state is reset below.
            graph_output = self._cudagraph_wrapper.decode(
                codes_step,
                state_slot_ids,
                allow_frame_padding=bool(terminal_slots) and len(terminal_slots) == len(slots),
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
            if staged_terminal_ids is not None:
                terminal_slot_ids = staged_terminal_ids
            else:
                terminal_rows = [row for row, slot in enumerate(slots) if slot in terminal_slots]
                terminal_slot_ids = (
                    state_slot_ids if len(terminal_rows) == len(slots) else state_slot_ids[terminal_rows]
                )
            self._reset_slot_ids(terminal_slot_ids)

        if self._gpu_output and output_buffers is not None:
            # Write directly into the caller-owned per-forward slab.  This
            # preserves graph output before the next replay without creating
            # a temporary waveform tensor that would be packed again later.
            out: dict[int, torch.Tensor] = {}
            row = 0
            while row < len(slots):
                destination = output_buffers.get(slots[row])
                if destination is None:
                    row += 1
                    continue
                end = row + 1
                if end < len(slots) and destination.is_contiguous():
                    storage = destination.untyped_storage().data_ptr()
                    offset = destination.storage_offset()
                    row_elements = destination.numel()
                    while end < len(slots):
                        following = output_buffers.get(slots[end])
                        if not (
                            following is not None
                            and following.shape == destination.shape
                            and following.dtype == destination.dtype
                            and following.device == destination.device
                            and following.is_contiguous()
                            and following.untyped_storage().data_ptr() == storage
                            and following.storage_offset() == offset + (end - row) * row_elements
                        ):
                            break
                        end += 1
                if end > row + 1:
                    # A mixed-frame forward can leave gaps between equal-sized
                    # rows in its slab. Copy each adjacent run as one batch,
                    # without overwriting the intervening requests' output.
                    destination = destination.as_strided(
                        (end - row, *destination.shape), (destination.numel(), *destination.stride())
                    )
                    source = audio_tensor[row:end]
                else:
                    source = audio_tensor[row]
                # Crop time before reshaping: flattening first mixes stereo
                # channels whenever a row is shorter than the decode batch.
                destination.copy_(source[..., : destination.shape[-1]].view_as(destination))
                out.update((slot, output_buffers[slot]) for slot in slots[row:end])
                row = end
            return out

        if self._gpu_output:
            # A graph's static output is overwritten on every replay, including
            # multiple replays within one forward. Snapshot on the producer
            # stream before returning; record_stream alone cannot prevent this
            # overwrite. MRV2 owns the subsequent pinned, asynchronous D2H.
            audio = audio_tensor.detach().to(dtype=torch.float32, copy=True)
        else:
            audio = audio_tensor.detach().to("cpu", torch.float32)
        out: dict[int, torch.Tensor] = {}
        for row, slot in enumerate(slots):
            audio_length = int(slot_codes[slot].shape[1]) * self._samples_per_frame
            out[slot] = audio[row, ..., :audio_length].contiguous()
        return out


def _resolve_streaming_graph_frame_sizes(
    initial_frames: int,
    steady_frames: int,
    connector_extra_cfg: dict | None,
) -> list[int]:
    """Streaming-graph frame sizes to pre-capture: ``{initial, steady}`` + ramp ladder.

    Ladder parsing is delegated to ``parse_chunk_ramp`` so the codec and the
    stage input processor agree on one validated ``codec_chunk_ramp``:
    null / malformed / single-entry values disable the ramp on both sides
    (warn only) and never raise during codec init.
    """
    ramp = parse_chunk_ramp(connector_extra_cfg or {}, steady=steady_frames) or []
    return sorted({s for s in (initial_frames, steady_frames, *ramp) if s > 0})


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
    requires_exact_input_shape: bool = True

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
        self._codec_stream: torch.cuda.Stream | None = None
        scheduler_cfg = getattr(self.vllm_config, "scheduler_config", None)
        self._stream_state_capacity = max(1, int(getattr(scheduler_cfg, "max_num_seqs", 1) or 1))
        self._initial_stream_chunk_frames: int = self._connector_int("initial_codec_chunk_frames", default=0)
        self._stream_chunk_frames: int = self._connector_int("codec_chunk_frames", default=0)
        self._stream_max_step_frames: int = self._stream_chunk_frames or 100
        # Opt-in: size codec attention rings as context + one chunk so chunked
        # streaming reproduces whole-sequence decoding (costs ring memory).
        self._stream_ring_headroom: bool = bool(self._connector_int("codec_ring_headroom", default=0))
        self._stream_req_slots: dict[str, int] = {}
        self._stream_first_audio_requests: set[str] = set()
        self._async_chunk = bool(getattr(self.vllm_config.model_config, "async_chunk", False))
        self._gpu_stream_output = (
            self._async_chunk
            and bool(getattr(vllm_config.model_config, "use_v2_model_runner", False))
            and getattr(cfg, "model_type", None) == "moss_tts_local"
            # Diagnostic switch: 0 copies each decode group's batch to the host synchronously.
            and bool(self._connector_int("codec_gpu_stream_output", default=1))
        )
        self._streaming_graph_batch_sizes = self._streaming_graph_batch_sizes_from_compilation_config()
        ramp = parse_chunk_ramp(self._connector_extra(), steady=self._stream_chunk_frames)
        if ramp:
            # The sender gives the ramp precedence over initial chunk size.
            # Keep the resolved initial frame count consistent with the sender.
            self._initial_stream_chunk_frames = ramp[0]
        self._stream_max_step_frames = max(
            self._stream_max_step_frames, self._initial_stream_chunk_frames, *(ramp or [])
        )
        self._streaming_graph_frame_sizes = _resolve_streaming_graph_frame_sizes(
            self._initial_stream_chunk_frames, self._stream_chunk_frames, self._connector_extra()
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
        if self._gpu_stream_output and input_ids is not None and input_ids.is_cuda:
            if self._codec_stream is None:
                self._codec_stream = torch.cuda.Stream(device=input_ids.device)
            main_stream = torch.cuda.current_stream(input_ids.device)
            # Runner input storage is reused by the next batch on main_stream.
            # Own it before allowing that preparation to overlap codec decode.
            owned_input_ids = input_ids.clone()
            with torch.cuda.stream(self._codec_stream):
                self._codec_stream.wait_stream(main_stream)
                owned_input_ids.record_stream(self._codec_stream)
                output = self._forward(
                    owned_input_ids,
                    positions,
                    intermediate_tensors,
                    inputs_embeds,
                    runtime_additional_information,
                    **kwargs,
                )
                if isinstance(output.multimodal_outputs, PackedOutputSnapshot):
                    output.multimodal_outputs.record_producer_event(self._codec_stream)
            return output
        return self._forward(
            input_ids, positions, intermediate_tensors, inputs_embeds, runtime_additional_information, **kwargs
        )

    @torch.no_grad()
    def _forward(
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
        first_audio_flags = self._first_audio_flags(info_list)
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
                multimodal_outputs={
                    "model_outputs": audios,
                    "sr": srs,
                    **self._first_audio_metadata(first_audio_flags),
                },
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
        first_audio_flags = self._first_audio_flags(info_list)
        trim_first: set[int] = set()
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
                if first_audio_flags[i] and req_key not in self._stream_req_slots:
                    trim_first.add(i)
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
            stream_output_slab = None
            stream_output_buffers: dict[int, torch.Tensor] = {}
            if self._gpu_stream_output and device.type == "cuda":
                total_elements = 0
                for output_index, _, codes_nq_t, _ in streaming_work:
                    samples = int(codes_nq_t.shape[1]) * self._codec.downsample_rate
                    total_elements += samples * (self._n_channels if self._n_channels > 1 else 1)
                if total_elements:
                    stream_output_slab = torch.empty(total_elements, dtype=torch.float32, device=device)
                    offset = 0
                    for output_index, _, codes_nq_t, _ in streaming_work:
                        samples = int(codes_nq_t.shape[1]) * self._codec.downsample_rate
                        elements = samples * (self._n_channels if self._n_channels > 1 else 1)
                        shape = (self._n_channels, samples) if self._n_channels > 1 else (samples,)
                        stream_output_buffers[output_index] = stream_output_slab.narrow(0, offset, elements).view(shape)
                        offset += elements
            for i, wav in self._decode_streaming_batch(
                streaming_work,
                output_buffers=stream_output_buffers or None,
            ).items():
                if i in trim_first:
                    wav = wav[..., int(self._codec.downsample_rate) :]
                audios[i] = wav.reshape(-1) if wav.ndim == 1 or int(wav.shape[0]) == 1 else wav

        payload = {"model_outputs": audios, "sr": srs, **self._first_audio_metadata(first_audio_flags)}
        if self._gpu_stream_output and device.type == "cuda":
            # One transfer per dtype/device preserves ragged request lengths
            # without issuing one D2H per waveform. A fresh slot owns each
            # output until the runner completes its copy; no reusable ring.
            packed = pack_output_snapshot(payload, {}, max_buckets=0, reuse_existing_storage=True)
            if packed is not None:
                payload = packed
        return OmniOutput(text_hidden_states=None, multimodal_outputs=payload)

    def _first_audio_flags(self, infos):
        flags = []
        for i, info in enumerate(infos):
            meta = (info.get("meta", {}) if isinstance(info, dict) else {}) or {}
            request_id = self._runtime_request_key(info, meta, i)
            first = meta.get("first_audio", False)
            if isinstance(first, (tuple, list)):
                first = first[0] if first else False
            if isinstance(first, torch.Tensor):
                first = bool(first.numel() and first.reshape(-1)[0].item())
            if first:
                self._stream_first_audio_requests.add(request_id)
            flags.append(request_id in self._stream_first_audio_requests)
        return flags

    @staticmethod
    def _first_audio_metadata(flags):
        if not any(flags):
            return {}
        return {FIRST_AUDIO_REQUIRED_KEY: [torch.tensor(flag) for flag in flags]}

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
            gpu_output=self._gpu_stream_output,
            chunk_frames=self._stream_max_step_frames,
            ring_headroom=self._stream_ring_headroom,
        )
        if self._stream_ring_headroom:
            logger.info(
                "MOSS-TTS codec attention rings sized with headroom for %d-frame chunks",
                self._stream_max_step_frames,
            )
        return self._stream_session

    def _decode_streaming_batch(
        self,
        items: list[tuple[int, str, torch.Tensor, bool]],
        *,
        output_buffers: dict[int, torch.Tensor] | None = None,
    ) -> dict[int, torch.Tensor]:
        session = self._ensure_stream_session()
        if session is None:
            return {}

        outputs: dict[int, torch.Tensor] = {}
        grouped: dict[int, list[tuple[int, str, int, torch.Tensor, bool]]] = {}
        max_step_frames = max(1, int(self._stream_max_step_frames))
        graph = session._cudagraph_wrapper
        # Terminal tails join the regular padded frame bucket when every
        # execution batch width has a graph, so a step stays one replay
        # instead of one eager or plain-graph decode per distinct tail length.
        coalesce_tails = graph is not None and max(graph.batch_sizes, default=0) >= self._stream_state_capacity

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
                    output_buffer=output_buffers.get(output_index) if output_buffers is not None else None,
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

            frame_count = int(codes_nq_t.shape[1])
            group_frames = frame_count
            if coalesce_tails and finished:
                # Only terminal rows may be padded: their state is reset after
                # the step, and the output is cropped to the exact frame count.
                group_frames = graph._select_frame_size(frame_count, allow_padding=True) or frame_count
            grouped.setdefault(group_frames, []).append((output_index, request_id, slot, codes_nq_t, finished))

        for group in grouped.values():
            plan = {slot: codes_nq_t for _, _, slot, codes_nq_t, _ in group}
            terminal_slots = {slot for _, _, slot, _, finished in group if finished}
            step_buffers = (
                {
                    slot: output_buffers[output_index]
                    for output_index, _, slot, _, _ in group
                    if output_buffers is not None and output_index in output_buffers
                }
                if output_buffers is not None
                else None
            )
            decoded = session.step(plan, terminal_slots=terminal_slots, output_buffers=step_buffers)
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
        output_buffer: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        if codes_nq_t.numel() == 0:
            return None
        max_step_frames = max(1, int(self._stream_max_step_frames))
        parts: list[torch.Tensor] = []
        for start in range(0, int(codes_nq_t.shape[1]), max_step_frames):
            chunk = codes_nq_t[:, start : start + max_step_frames]
            is_terminal_chunk = terminal and start + max_step_frames >= int(codes_nq_t.shape[1])
            chunk_samples = int(chunk.shape[1]) * self._codec.downsample_rate
            chunk_buffer = (
                output_buffer[
                    ..., start * self._codec.downsample_rate : start * self._codec.downsample_rate + chunk_samples
                ]
                if output_buffer is not None
                else None
            )
            decoded = session.step(
                {slot: chunk},
                terminal_slots={slot} if is_terminal_chunk else None,
                output_buffers={slot: chunk_buffer} if chunk_buffer is not None else None,
            )
            wav = decoded.get(slot)
            if wav is not None:
                parts.append(wav)
        if output_buffer is not None:
            return output_buffer
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
        getattr(self, "_stream_first_audio_requests", set()).discard(request_id)

    def on_requests_finished(self, finished_req_ids: set[str] | list[str]) -> None:
        """Release codec streaming slots when requests finish outside payload flow.

        Normal streaming completion releases slots from ``_decode_streaming_batch``
        when the Stage-0 payload carries ``finished=True``. Client disconnects
        and engine-side aborts can finish a request without delivering that
        terminal payload, so the runner calls this hook from its finished-request
        path to avoid leaking stream slots and buffered codes.
        """
        # Cancellation and terminal cleanup share the decoder's stream order;
        # a reused slot must not be reset ahead of an outstanding decode.
        context = torch.cuda.stream(self._codec_stream) if self._codec_stream is not None else nullcontext()
        with context:
            session = self._stream_session
            for req_id in finished_req_ids:
                request_id = str(req_id)
                slot = self._stream_req_slots.get(request_id)
                if slot is None:
                    getattr(self, "_stream_first_audio_requests", set()).discard(request_id)
                    continue
                if session is not None:
                    self._finish_stream_request(request_id, session, slot)
                else:
                    self._stream_req_slots.pop(request_id, None)

    def _connector_extra(self) -> dict[str, Any]:
        model_cfg = getattr(self.vllm_config, "model_config", None)
        connector_cfg = getattr(model_cfg, "stage_connector_config", None)
        if isinstance(connector_cfg, dict):
            extra_cfg: dict | None = connector_cfg.get("extra", connector_cfg)
        else:
            extra_cfg = getattr(connector_cfg, "extra", None)
        return extra_cfg if isinstance(extra_cfg, dict) else {}

    def _connector_int(self, name: str, default: int = 0) -> int:
        return int(self._connector_extra().get(name, default))

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
        # The codec checkpoint is separate from the Stage0 weights iterator.
        for _ in weights:
            pass
        device = self.vllm_config.device_config.device
        codec_cfg, codec = load_codec(
            self._codec_path,
            device=device,
            load_config=self.vllm_config.load_config,
            num_quantizers=self._n_vq,
            attention_backend=getattr(self.vllm_config.model_config.hf_config, "codec_attention_backend", "sdpa"),
            skip_empty_tiles=bool(self._connector_int("codec_skip_empty_attention_tiles", default=0)),
            fused_slots=bool(self._connector_int("codec_fused_slot_attention", default=0)),
        )
        self._codec = codec
        inferred_channels = 2 if "v2" in self._codec_path.lower() else 1
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
