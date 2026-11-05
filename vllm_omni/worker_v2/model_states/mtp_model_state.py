# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""MTP execution resources, current-step frame output and following-input state."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn as nn
from vllm.config.compilation import CUDAGraphMode
from vllm.logger import init_logger
from vllm.model_executor.offloader.base import get_offloader
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.states import RequestState

from vllm_omni.platforms import current_omni_platform
from vllm_omni.utils.device_copy import DeviceStager, index_to_device
from vllm_omni.worker_v2.streaming_audio import StreamingAudioBuffer, StreamingAudioOutput

if TYPE_CHECKING:
    from vllm_omni.worker_v2.model_states.omni_model_state import OmniModelState

logger = init_logger(__name__)
_NO_SEED = object()


def _has_ref_codes(buffer: dict[str, Any]) -> bool:
    ref = buffer.get("codes", {}).get("ref") if isinstance(buffer.get("codes"), dict) else None
    return isinstance(ref, torch.Tensor) and ref.numel() > 0


def _snapshot_runtime(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to("cpu", copy=True)
    if isinstance(value, dict):
        return {key: _snapshot_runtime(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_snapshot_runtime(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_snapshot_runtime(item) for item in value)
    return value


class FrameBatch:
    """One row mapping shared by eager MTP and the streaming codec for a step."""

    def __init__(self, entries: list[tuple[int, int, str, bool]]) -> None:
        self.entries = entries
        rows, slots, ids, prefill = zip(*entries) if entries else ((), (), (), ())
        self.slots = list(slots)
        self.req_ids = list(ids)
        self.identities = np.empty(len(entries), dtype=object)
        self.identities[:] = self.req_ids
        self.meta = np.empty((5, len(entries)), dtype=np.int64)
        self.meta[0] = rows
        self.meta[1] = slots
        self.meta[2] = prefill
        self.has_prefill = bool(self.meta[2].any())

    def __len__(self) -> int:
        return len(self.entries)


@dataclass
class TalkerInputs:
    embeds: torch.Tensor
    length: int = 0
    address: int = field(init=False)

    def __post_init__(self) -> None:
        self.address = self.embeds.data_ptr()


@dataclass
class SuspendedAudio:
    position: int
    decoder_state: list[torch.Tensor]
    generator: torch.Generator | None
    runtime: dict[str, Any]
    frame_embed: torch.Tensor


class MTPState:
    """Current-step MTP and following-input state for ``has_talker_mtp`` models."""

    def __init__(self, owner: OmniModelState) -> None:
        self.owner = owner
        self._mtp_generators: dict[str, torch.Generator | None] = {}
        self._mtp_seeded: set[str] = set()
        self.embeds: torch.Tensor | None = None
        self.ready: dict[int, str] = {}
        self.settled: np.ndarray | None = None
        self.recorded: tuple[InputBatch, FrameBatch, torch.Tensor] | None = None
        self.first_audio_requests: set[str] = set()
        self.stream_pos: dict[str, int] = {}
        self.decode_event: torch.cuda.Event | None = None
        self._codec_inputs_done: torch.cuda.Event | None = None
        self.first_audio_sender: Any = None
        self.first_audio_stream: torch.cuda.Stream | None = None
        self._first_audio_valid: torch.Tensor | None = None
        self._side_stream: torch.cuda.Stream | None = None
        self._fast_ok: bool | None = None
        self._audio_buffer: StreamingAudioBuffer | None = None
        self._frame_requests: set[str] = set()
        self._suspended_audio: dict[str, SuspendedAudio] = {}
        self._restore_audio: dict[str, list[torch.Tensor]] = {}
        self._talker_inputs: dict[str, TalkerInputs] = {}
        # Per-step row indices, read by the GPU in place (one stager per call site).
        self._stage_eager = DeviceStager()
        self._stage_settled = DeviceStager()
        self._stage_meta = DeviceStager()
        self._stage_scheduled = DeviceStager()
        self._stage_history = DeviceStager()
        self._slot_stream_owner: np.ndarray | None = None
        self._slot_stream_pos: np.ndarray | None = None

        model = self.owner.model
        device = self.owner.device
        # Static buffers shared by post-sample and model-owned MTP execution.
        self._mtp_input_ids: torch.Tensor | None = None
        self._mtp_input_embeds: torch.Tensor | None = None
        self._mtp_hidden: torch.Tensor | None = None
        self._mtp_text_step: torch.Tensor | None = None
        self._mtp_offsets: torch.Tensor | None = None
        self._stage_offsets = DeviceStager()  # per-step MTP row offsets, read in place by the GPU
        self._mtp_sample_uniforms: torch.Tensor | None = None
        self._mtp_runner: Any | None = None
        # Captured whole-MTP graph entry per batch size, once a replay through
        # the wrapper has resolved it: (padded rows, entry).
        self._mtp_replays: dict[int, tuple[int, Any]] = {}
        self._raw_mtp: Callable | None = getattr(model, "mtp", None)
        if not callable(self._raw_mtp) and getattr(model, "has_talker_mtp", False):
            self._raw_mtp = getattr(model, "talker_mtp", None)
        if getattr(model, "has_talker_mtp", False) and (not callable(self._raw_mtp) or self.owner._embed_dim <= 0):
            raise TypeError("has_talker_mtp requires an MTP callable and a valid input embedding dimension")
        if self.owner._embed_dim > 0 and callable(self._raw_mtp):
            max_bs = self.owner.scheduler_config.max_num_seqs
            self._mtp_input_ids = torch.zeros(max_bs, dtype=torch.long, device=device)
            self._mtp_input_embeds = torch.zeros((max_bs, self.owner._embed_dim), dtype=self.owner.dtype, device=device)
            self._mtp_hidden = torch.zeros((max_bs, self.owner._embed_dim), dtype=self.owner.dtype, device=device)
            self._mtp_text_step = torch.zeros((max_bs, self.owner._embed_dim), dtype=self.owner.dtype, device=device)
            self._mtp_offsets = torch.zeros(max_bs, dtype=torch.long, device=device)
            sample_steps = int(getattr(model, "mtp_sample_steps", 0))
            sample_vocab_size = int(getattr(model, "mtp_sample_vocab_size", 0))
            if bool(getattr(model, "mtp_sample_uniforms", False)) and sample_steps > 0 and sample_vocab_size > 0:
                self._mtp_sample_uniforms = torch.empty(
                    (max_bs, sample_steps, sample_vocab_size),
                    dtype=torch.float32,
                    device=device,
                )
            self._mtp_runner = self._init_mtp_runner(model)
            if getattr(model, "has_talker_mtp", False):
                self.embeds = torch.zeros((max_bs, self.owner._embed_dim), dtype=self.owner.dtype, device=device)

    def mark_settled(self, slot: int, req_id: str) -> None:
        if self.settled is None:
            assert self.embeds is not None
            self.settled = np.full(self.embeds.shape[0], None, dtype=object)
        self.settled[slot] = req_id

    def release_request(self, req_id: str, slot: int) -> None:
        self._mtp_generators.pop(req_id, None)
        self._mtp_seeded.discard(req_id)
        self.stream_pos.pop(req_id, None)
        self.release_stream_slot(req_id, slot)
        self.first_audio_requests.discard(req_id)
        self.ready.pop(slot, None)
        if self.settled is not None:
            self.settled[slot] = None

    def _stream_slots(self) -> tuple[np.ndarray, np.ndarray]:
        """Per slot: the request whose stream position is tracked there, and its next frame index."""
        if self._slot_stream_owner is None:
            assert self.embeds is not None
            n = int(self.embeds.shape[0])
            self._slot_stream_owner = np.full(n, None, dtype=object)
            self._slot_stream_pos = np.zeros(n, dtype=np.int64)
        assert self._slot_stream_pos is not None
        return self._slot_stream_owner, self._slot_stream_pos

    def stream_position(self, req_id: str, req_idx: int) -> int | None:
        """Next frame index of a request's stream, wherever it is tracked."""
        owners = self._slot_stream_owner
        if owners is not None and owners[req_idx] == req_id:
            assert self._slot_stream_pos is not None
            return int(self._slot_stream_pos[req_idx])
        return self.stream_pos.get(req_id)

    def release_stream_slot(self, req_id: str, req_idx: int) -> None:
        """Stop tracking the request's stream position in its slot."""
        owners = self._slot_stream_owner
        if owners is not None and owners[req_idx] == req_id:
            owners[req_idx] = None

    def suspend_audio(self, req_id: str, req_idx: int) -> None:
        decoder = getattr(self.owner.model, "stream_decoder", None)
        position = self.stream_position(req_id, req_idx)
        self.release_stream_slot(req_id, req_idx)
        if decoder is None or position is None:
            return
        # Slot reuse is rare and must wait for the last side-stream write.
        saved = self._restore_audio.pop(req_id, None)
        if saved is None:
            self._decode_stream.synchronize()
            saved = decoder.save_slot(req_idx)
        history = self._talker_inputs[req_id]
        history.embeds = history.embeds[: history.length].to("cpu", copy=True)
        history.address = history.embeds.data_ptr()
        runtime = self.owner.intermediate_buffer.buffers[req_idx]
        assert self.embeds is not None
        self._suspended_audio[req_id] = SuspendedAudio(
            position,
            saved,
            self._mtp_generators.get(req_id),
            _snapshot_runtime(runtime),
            self.embeds[req_idx].to("cpu", copy=True),
        )

    def resume_audio(self, req_id: str, req_idx: int) -> None:
        saved = self._suspended_audio.pop(req_id, None)
        if saved is not None:
            self.stream_pos[req_id] = saved.position
            self._mtp_generators[req_id] = saved.generator
            if saved.generator is not None:
                self._mtp_seeded.add(req_id)
            self._restore_audio[req_id] = saved.decoder_state
            self.owner.intermediate_buffer.buffers[req_idx] = saved.runtime
            assert self.embeds is not None
            self.embeds[req_idx].copy_(saved.frame_embed)
            self.ready[req_idx] = req_id

    def finish_audio(self, req_ids: set[str]) -> None:
        for req_id in req_ids:
            self._suspended_audio.pop(req_id, None)
            self._restore_audio.pop(req_id, None)
            self.stream_pos.pop(req_id, None)
            self._talker_inputs.pop(req_id, None)
        if self._audio_buffer is not None:
            self._audio_buffer.finish(req_ids)

    def has_pending_replay(self) -> bool:
        """Whether any preempted request still has saved inputs to replay."""
        return bool(self._restore_audio)

    def replay_inputs(self, req_id: str, req_idx: int, offset: int, ids: torch.Tensor, embeds: torch.Tensor) -> bool:
        """Rebuild freed Talker KV from the exact conditioned inputs, without rerunning MTP."""
        if req_id not in self._restore_audio:
            return False
        history = self._talker_inputs[req_id]
        count = min(len(ids), max(0, history.length - offset))
        if count:
            embeds[:count].copy_(history.embeds[offset : offset + count])
        if count < len(ids):
            # Recompute ends with the last accepted CB0, whose full eager frame
            # was saved but has not yet been consumed as a Talker input.
            if len(ids) - count != 1 or offset + count != history.length:
                raise RuntimeError(f"Talker replay exceeds saved inputs for {req_id}")
            info = self.owner.intermediate_buffer.buffers[req_idx]
            _, _, updates = self.owner.model.preprocess(ids[count:], embeds[count:], **info, _omni_is_prefill=False)
            mtp_inputs = updates.pop("mtp_inputs", None)
            if mtp_inputs is None:
                raise RuntimeError(f"Talker replay has no text conditioning for {req_id}")
            assert self.embeds is not None
            embeds[count:].copy_(self.embeds[req_idx] + mtp_inputs[1].reshape(1, -1))
            self.owner.intermediate_buffer.update(req_idx, updates, self.owner.model.gpu_resident_buffer_keys)
        return True

    def record_inputs(self, input_batch: InputBatch, embeds: torch.Tensor) -> None:
        if getattr(self.owner.model, "stream_decoder", None) is None:
            return
        history_slices: list[torch.Tensor] = []
        input_slices: list[torch.Tensor] = []
        histories = self._talker_inputs
        # Packing a pointer table pays off for larger batches; small batches
        # keep foreach to avoid staging more metadata than they copy.
        scatter = input_batch.num_reqs >= 32 and embeds.is_cuda and embeds.is_contiguous()
        meta = np.empty((4, input_batch.num_reqs), dtype=np.int64) if scatter else None
        for i, req_id in enumerate(input_batch.req_ids):
            start, end = input_batch.query_start_loc_np[i : i + 2]
            offset = int(input_batch.num_computed_tokens_np[i])
            length = offset + int(end - start)
            history = histories.get(req_id)
            # A suspended request's inputs wait on the host; bring them back.
            if history is None or history.embeds.device != embeds.device:
                storage = torch.empty(
                    self.owner.vllm_config.model_config.max_model_len,
                    embeds.shape[-1],
                    device=embeds.device,
                    dtype=embeds.dtype,
                )
                if history is not None:
                    storage[: history.length].copy_(history.embeds)
                history = TalkerInputs(storage, history.length if history else 0)
                self._talker_inputs[req_id] = history
            if meta is not None:
                count = int(end - start)
                meta[:, i] = (history.address, start, offset, count)
            else:
                history_slices.append(history.embeds[offset:length])
                input_slices.append(embeds[start:end])
            history.length = max(history.length, length)
        if meta is not None and meta[3].sum():
            from vllm_omni.worker_v2.model_states.eager_mtp_kernels import record_history

            counts = meta[3].copy()
            meta[3] = counts.cumsum() - counts
            token_rows = np.repeat(np.arange(input_batch.num_reqs, dtype=np.int64), counts)
            packed = np.concatenate((meta.reshape(-1), token_rows))
            record_history(self._stage_history(packed, embeds.device), embeds, input_batch.num_reqs, len(token_rows))
        elif history_slices:
            # Keep exact inputs for replay without launching one copy per row.
            torch._foreach_copy_(history_slices, input_slices)

    def prepare_audio_output(
        self, input_batch: InputBatch, req_states: RequestState, outputs: dict[str, Any]
    ) -> StreamingAudioOutput | None:
        if self._audio_buffer is None:
            return None
        wav = outputs.pop("model_outputs", None)
        if not isinstance(wav, torch.Tensor):
            return None
        n = input_batch.num_reqs
        total_after = input_batch.num_computed_tokens_np[:n] + np.asarray(input_batch.num_scheduled_tokens[:n]) + 1
        limits = np.minimum(
            req_states.max_seq_len[input_batch.idx_mapping_np[:n]], self.owner.vllm_config.model_config.max_model_len
        )
        requests = [
            self._audio_buffer.requests.get(rid) if rid in self._frame_requests else None for rid in input_batch.req_ids
        ]
        self._frame_requests.clear()
        event = self.decode_event
        self.decode_event = None
        return StreamingAudioOutput(
            wav,
            outputs["meta"]["codec_frame_valid"],
            outputs["sr"][0],
            event,
            input_batch.query_start_loc_np[: n + 1].copy(),
            requests,
            total_after >= limits,
        )

    @property
    def _decode_stream(self) -> torch.cuda.Stream:
        """Side stream for Talker stream decode, so the codec overlaps the next step."""
        if self._side_stream is None:
            self._side_stream = torch.cuda.Stream()
        return self._side_stream

    def set_first_audio_sink(self, sink: Any) -> None:
        from vllm_omni.worker_v2.first_audio_sender import FirstAudioSender

        self.first_audio_sender = FirstAudioSender(sink)

    def apply_frame_embeddings(
        self,
        decode_mtp_rows: list[tuple[int, int]],
        embeds: torch.Tensor,
        input_batch: InputBatch,
        text_steps: torch.Tensor,
    ) -> None:
        """Compose backbone decode inputs from cached frame and current text embeddings."""
        req_indices = [int(input_batch.idx_mapping_np[i]) for i, _start in decode_mtp_rows]
        for req_idx in req_indices:
            req_id = self.owner.intermediate_buffer.buffers[req_idx].get("req_id")
            if self.ready.get(req_idx) != req_id:
                # Running the deferred MTP here would re-emit or drop a frame.
                raise RuntimeError(f"Eager Talker-MTP frame missing for request {req_id!r}")
        assert self.embeds is not None
        device = embeds.device
        text_step = text_steps.reshape(len(decode_mtp_rows), -1)
        rows = self._stage_eager(req_indices, device)
        offsets = self._decode_mtp_offsets(decode_mtp_rows, input_batch, device)
        frame_embeds = self.embeds.index_select(0, rows)
        embeds.index_copy_(0, offsets, (frame_embeds + text_step.to(frame_embeds.dtype)).to(embeds.dtype))

    def _apply_settled_frames(self, settled_rows: list[tuple[int, int, int, str]], embeds: torch.Tensor) -> None:
        """Settled decode rows: previous eager frame embedding plus the constant text step."""
        ready = self.ready
        for _i, req_idx, _start, req_id in settled_rows:
            if ready.get(req_idx) != req_id:
                raise RuntimeError(f"Eager Talker-MTP frame missing for request {req_id!r}")
        assert self.embeds is not None
        device = embeds.device
        _rows, req_indices, starts, _req_ids = zip(*settled_rows)
        meta = self._stage_settled(np.array((req_indices, starts), dtype=np.int64), device)
        eager = self.embeds
        text_step = self.owner.model.eager_settled_text_step()
        if embeds.is_cuda and embeds.is_contiguous() and text_step.device == device and text_step.is_contiguous():
            from vllm_omni.worker_v2.model_states.eager_mtp_kernels import settled_frames

            settled_frames(meta, len(req_indices), eager, text_step, embeds)
            return
        rows, offsets = meta
        text_step = text_step.to(device=device, dtype=eager.dtype)
        frame_embeds = eager.index_select(0, rows)
        embeds.index_copy_(0, offsets, (frame_embeds + text_step).to(embeds.dtype))

    def post_sample(
        self,
        input_batch: InputBatch,
        text_hidden: torch.Tensor,
        sampled_token_ids: torch.Tensor,
        multimodal_outputs: dict[str, Any],
        mtp_batch_descriptor_dispatcher: Callable[[int], Any] | None = None,
    ) -> None:
        """Complete each sampled row's frame in this step and publish it with this step's output.

        Consumes the rows recorded by ``run_preprocess``. Writes the frame codes
        and validity into the last token row of each request span of the
        retained multimodal output, and keeps the frame's codec embedding sum
        for the next step's input. Issued on the main stream before the async
        output copy, so no extra synchronization is needed.
        """
        recorded, self.recorded = self.recorded, None
        if recorded is None or recorded[0] is not input_batch or not recorded[1]:
            return
        batch, input_ids = recorded[1], recorded[2]
        entries = batch.entries
        codes_out = multimodal_outputs.get("codes", {}).get("audio")
        valid_out = multimodal_outputs.get("meta", {}).get("codec_frame_valid")
        if not isinstance(codes_out, torch.Tensor) or not isinstance(valid_out, torch.Tensor):
            raise RuntimeError("Eager Talker-MTP requires retained codes.audio and codec_frame_valid outputs")
        assert self._mtp_input_ids is not None and self._mtp_input_embeds is not None
        assert self._mtp_hidden is not None and self._mtp_text_step is not None
        assert self.embeds is not None

        if sampled_token_ids.reshape(input_batch.num_reqs, -1).shape[1] != 1:
            raise ValueError("Current-step MTP requires one sampled token per request")
        bsz = len(entries)
        device = text_hidden.device
        if self._fast_path_ok(codes_out, text_hidden):
            self._run_mtp_fused(
                input_batch, text_hidden, sampled_token_ids, multimodal_outputs, batch, input_ids,
                codes_out, valid_out, mtp_batch_descriptor_dispatcher,
            )  # fmt: skip
            return
        if getattr(self.owner.model, "stream_decoder", None) is not None:
            raise RuntimeError("Talker stream decode requires the fused eager-MTP path")
        batch_rows = index_to_device([i for i, _req_idx, _req_id, _prefill in entries], device)
        req_indices = [req_idx for _i, req_idx, _req_id, _prefill in entries]
        last_tokens = input_batch.query_start_loc.index_select(0, batch_rows + 1).long() - 1
        layer0 = sampled_token_ids.reshape(input_batch.num_reqs, -1)[:, 0].index_select(0, batch_rows)

        batch_ids = self._mtp_input_ids[:bsz]
        batch_emb = self._mtp_input_embeds[:bsz]
        batch_hidden = self._mtp_hidden[:bsz]
        batch_step = self._mtp_text_step[:bsz]
        batch_ids.copy_(layer0.to(batch_ids.dtype))
        batch_emb.copy_(self.owner.model.embed_input_ids(batch_ids.reshape(-1, 1)).reshape(bsz, -1))
        torch.index_select(text_hidden, 0, last_tokens, out=batch_hidden)
        # The text step is added by the next step's preprocess.
        batch_step.zero_()
        frame_embeds, codes = self._mtp_forward(
            req_indices,
            batch_ids,
            batch_emb,
            batch_hidden,
            batch_step,
            mtp_batch_descriptor_dispatcher,
        )
        assert codes is not None
        rows = index_to_device(req_indices, device)
        self.embeds.index_copy_(0, rows, frame_embeds[:bsz].reshape(bsz, -1).to(self.embeds.dtype))
        if self.owner.vllm_config.cache_config.enable_prefix_caching:
            # Prefix-hit reconstruction reads the latest frame from the buffer.
            self.owner.intermediate_buffer.update_gpu_tensor_rows(
                req_indices, self.owner.model.mtp_output_key, codes[:bsz]
            )
        if codes_out.ndim != 2:
            raise RuntimeError(f"Eager Talker-MTP expects token-major codes.audio, got {tuple(codes_out.shape)}")
        codes_out.index_copy_(0, last_tokens, codes[:bsz].to(codes_out.dtype))
        # A decode row whose input CB0 was EOS belongs to a request that already
        # ended (async scheduling runs it once more); its new sample must not
        # become a frame. Prefill rows have no codec input.
        prefill_rows = index_to_device([int(prefill) for _i, _req_idx, _req_id, prefill in entries], device).bool()
        input_valid = self.owner.model.mtp_frame_valid(input_ids.index_select(0, last_tokens)) | prefill_rows
        valid = self.owner.model.mtp_frame_valid(layer0) & input_valid
        valid_out.index_copy_(0, last_tokens, valid.to(valid_out.dtype))
        for _i, req_idx, req_id, _prefill in entries:
            self.ready[req_idx] = req_id
        self._publish_first_audio(input_batch, entries, codes[:bsz], valid)
        first_audio = multimodal_outputs.get("meta", {}).get("first_audio")
        if isinstance(first_audio, torch.Tensor):
            scheduled = index_to_device(
                [int(req_id in self.first_audio_requests) for _, _, req_id, _ in entries], device
            )
            if self._first_audio_valid is None:
                scheduled.zero_()
            else:
                scheduled *= self._first_audio_valid.index_select(0, rows)
            first_audio.index_copy_(0, last_tokens, scheduled.to(first_audio.dtype))

    def _embedding_weight(self) -> torch.Tensor | None:
        inner = getattr(self.owner.model, "model", None)
        embed = getattr(inner, "embed_tokens", None)
        weight = getattr(embed, "weight", None)
        method = getattr(embed, "quant_method", None)
        unquantized = method is None or type(method).__name__ == "UnquantizedEmbeddingMethod"
        tp = getattr(embed, "tp_size", 1)
        if isinstance(weight, torch.Tensor) and weight.ndim == 2 and unquantized and tp == 1:
            return weight
        return None

    def _fast_path_ok(self, codes_out: torch.Tensor, text_hidden: torch.Tensor) -> bool:
        if self._fast_ok is None:
            self._fast_ok = (
                text_hidden.is_cuda
                and not self.owner.vllm_config.cache_config.enable_prefix_caching
                and self._embedding_weight() is not None
                and hasattr(self.owner.model, "_codebook_vocab_size")
            )
        return self._fast_ok and codes_out.ndim == 2 and text_hidden.is_contiguous()

    def _run_mtp_fused(
        self,
        input_batch: InputBatch,
        text_hidden: torch.Tensor,
        sampled_token_ids: torch.Tensor,
        multimodal_outputs: dict[str, Any],
        batch: FrameBatch,
        input_ids: torch.Tensor,
        codes_out: torch.Tensor,
        valid_out: torch.Tensor,
        mtp_batch_descriptor_dispatcher: Callable[[int], Any] | None,
    ) -> None:
        from vllm_omni.worker_v2.model_states.eager_mtp_kernels import eager_post, eager_pre

        owner = self.owner
        entries = batch.entries
        bsz = len(batch)
        device = text_hidden.device
        first_audio = multimodal_outputs.get("meta", {}).get("first_audio")
        fa_requests = self.first_audio_requests
        model = owner.model
        stream = getattr(model, "stream_decoder", None)
        stream_out = multimodal_outputs.get("model_outputs") if stream is not None else None
        first_rows: list[int] = []
        primes: list[tuple[int, torch.Tensor]] = []
        # Reuse the mapping assembled by preprocess for MTP and codec staging.
        rows_req_idx, rows_req_id = batch.slots, batch.req_ids
        has_stream = isinstance(stream_out, torch.Tensor) and bsz > 0
        meta_np = batch.meta.reshape(-1)
        meta_np[3 * bsz : 4 * bsz] = [req_id in fa_requests for req_id in rows_req_id]
        if has_stream:
            assert stream is not None
            # Frame index of each row's frame in its request's stream; 0 starts
            # a fresh decoder state in the request's slot. A voice-clone
            # request's stream starts after its reference codes, which prime
            # the decoder state first (as Code2Wav's first-chunk context does).
            if self._audio_buffer is None:
                self._audio_buffer = StreamingAudioBuffer(int(stream.spf), int(model.stream_chunk_frames))
            self._frame_requests = set(rows_req_id)
            slots = meta_np[bsz : 2 * bsz]
            owners, next_pos = self._stream_slots()
            ids = batch.identities
            pos = meta_np[4 * bsz :]
            pos[:] = next_pos[slots]
            kept = owners[slots] == ids
            if not kept.all():
                # New or resumed streams; a request that keeps its slot is tracked in place.
                positions = self.stream_pos
                for row in np.flatnonzero(~kept).tolist():
                    req_idx, req_id = rows_req_idx[row], rows_req_id[row]
                    self._audio_buffer.add(req_id)
                    p = positions.pop(req_id, None)
                    if p is None:
                        first_rows.append(row)
                        ref = model.get_stream_ref_context(owner.intermediate_buffer.buffers[req_idx])
                        p = 0
                        if ref is not None:
                            primes.append((req_idx, ref))
                            p = int(ref.shape[0])
                    pos[row] = p
                    owners[req_idx] = req_id
            next_pos[slots] = pos + 1
        meta = self._stage_meta(meta_np, device)
        rows = meta[bsz : 2 * bsz]
        sampled = sampled_token_ids.reshape(input_batch.num_reqs, -1)
        assert self._mtp_input_ids is not None and self._mtp_input_embeds is not None
        assert self._mtp_hidden is not None and self._mtp_text_step is not None
        batch_ids = self._mtp_input_ids[:bsz]
        batch_emb = self._mtp_input_embeds[:bsz]
        batch_hidden = self._mtp_hidden[:bsz]
        batch_step = self._mtp_text_step[:bsz]
        last_tokens, layer0 = eager_pre(
            meta, bsz, input_batch.query_start_loc, sampled, text_hidden, self._embedding_weight(),
            batch_ids, batch_emb, batch_hidden, batch_step,
        )  # fmt: skip
        req_indices = rows_req_idx
        frame_embeds, codes = self._mtp_forward(
            req_indices, batch_ids, batch_emb, batch_hidden, batch_step, mtp_batch_descriptor_dispatcher,
            req_ids=rows_req_id,
        )  # fmt: skip
        assert codes is not None
        has_prefill = batch.has_prefill
        fa_in_kernel = isinstance(first_audio, torch.Tensor) and not has_prefill
        direct_inputs = None
        graphs = model.stream_graphs if has_stream else None
        # Priming may itself replay the one-frame graph; fill directly only
        # when no reference work can overwrite these inputs before decode.
        if graphs is not None and not primes:
            direct_inputs = graphs.input_buffers(bsz)
        if direct_inputs is not None:
            if self._codec_inputs_done is not None:
                torch.cuda.current_stream().wait_event(self._codec_inputs_done)
            frame_codes, codec_slots, codec_pos = direct_inputs
        else:
            frame_codes = torch.empty(bsz, 1, codes.shape[-1], dtype=torch.int32, device=device) if has_stream else None
            codec_slots = codec_pos = None
        valid = eager_post(
            meta, bsz, last_tokens, layer0, frame_embeds.reshape(frame_embeds.shape[0], -1).contiguous(),
            self.embeds, codes.contiguous(), codes_out, input_ids, valid_out,
            first_audio if fa_in_kernel else None, self._first_audio_valid if fa_in_kernel else None,
            owner.model._codebook_vocab_size, codes32=frame_codes,
            codec_slots=codec_slots, codec_pos=codec_pos, scratch=stream.scratch_slot if stream is not None else 0,
        )  # fmt: skip
        self.ready.update(zip(rows_req_idx, rows_req_id))
        if has_stream:
            assert stream is not None
            assert isinstance(stream_out, torch.Tensor)
            assert frame_codes is not None
            decode = model.stream_graphs if model.stream_graphs is not None else stream
            # The next Talker step does not read the PCM, so the codec runs
            # beside it; only the output copies wait for it.
            side = self._decode_stream
            side.wait_stream(torch.cuda.current_stream())
            for t in (frame_codes, meta, last_tokens, valid, stream_out):
                t.record_stream(side)
            with torch.cuda.stream(side):
                if self._restore_audio:
                    for _i, req_idx, req_id, _p in entries:
                        saved = self._restore_audio.pop(req_id, None)
                        if saved is not None:
                            stream.restore_slot(req_idx, saved)
                model.prime_stream_decoder(primes)
                if direct_inputs is not None:
                    pcm = graphs.replay_prepared(bsz, frame_codes.shape[0])
                else:
                    pcm = decode(frame_codes, rows, meta[4 * bsz : 5 * bsz])
                stream_out.index_copy_(0, last_tokens, pcm.reshape(bsz, -1).to(stream_out.dtype))
                self._send_stream_first_frames(entries, pcm, valid, first_rows)
                done = torch.cuda.Event()
                done.record(side)
            self.decode_event = done
            self._codec_inputs_done = done
            self._stage_meta.retire(done)
        if has_prefill:
            self._publish_first_audio(input_batch, entries, codes[:bsz], valid)
            if isinstance(first_audio, torch.Tensor):
                if self._first_audio_valid is None:
                    first_audio.index_fill_(0, last_tokens, 0)
                else:
                    # Requests accepted by this step's publish count as scheduled.
                    fa = self.first_audio_requests
                    scheduled = self._stage_scheduled(
                        np.fromiter((q in fa for q in rows_req_id), dtype=np.int64, count=bsz), device
                    )
                    scheduled = scheduled * self._first_audio_valid.index_select(0, rows)
                    first_audio.index_copy_(0, last_tokens, scheduled.to(first_audio.dtype))

    def _send_stream_first_frames(
        self,
        entries: list[tuple[int, int, str, bool]],
        pcm: torch.Tensor,
        valid: torch.Tensor,
        first_rows: list[int],
    ) -> None:
        sender = self.first_audio_sender
        if sender is None or not first_rows:
            return
        rows = index_to_device(first_rows, pcm.device)
        request_ids = [entries[row][2] for row in first_rows]
        sample_rate = torch.tensor(int(self.owner.model.stream_sample_rate), dtype=torch.int32)
        accepted = sender.submit(
            request_ids,
            pcm.reshape(pcm.shape[0], -1).index_select(0, rows).float(),
            sample_rate,
            valid=valid.index_select(0, rows),
        )
        assert self._audio_buffer is not None
        for request_id in accepted:
            self._audio_buffer.requests[request_id].accept_first_audio()

    def _publish_first_audio(
        self,
        input_batch: InputBatch,
        entries: list[tuple[int, int, str, bool]],
        codes: torch.Tensor,
        valid: torch.Tensor | None = None,
    ) -> None:
        """Decode first frames here and hand the PCM to the client output.

        Only streams without reference codes qualify: their first chunk is
        decoded with no context, so the waveform equals what Code2Wav will
        produce for it. ``valid`` (per entry, on device) marks rows whose frame
        is real; direct delivery drops the others once their copy is done, so a
        stream whose first sample is codec EOS gets no first audio from here.
        """
        decoder = getattr(self.owner.model, "first_frame_decoder", None)
        if decoder is None:
            logger.info_once("Talker first-frame audio: no decoder on %s", type(self.owner.model).__name__)
            return
        first = [
            (row, i)
            for row, (i, req_idx, _req_id, prefill) in enumerate(entries)
            if prefill
            and _req_id not in self.first_audio_requests
            and not _has_ref_codes(self.owner.intermediate_buffer.buffers[req_idx])
        ]
        if not first:
            return
        # Decode on a side stream so the Talker's next step does not queue
        # behind it; the async output copy waits on its event instead.
        stream = self.first_audio_stream
        if stream is None:
            _least, greatest = torch.cuda.Stream.priority_range()
            stream = self.first_audio_stream = torch.cuda.Stream(device=codes.device, priority=greatest)
        frame_rows = index_to_device([row for row, _i in first], codes.device)
        frame_codes = codes.index_select(0, frame_rows)
        frame_valid = valid.index_select(0, frame_rows) if valid is not None else None
        if frame_valid is not None:
            frame_codes = torch.where(frame_valid[:, None], frame_codes, 0)
        if self._first_audio_valid is None:
            assert self.embeds is not None
            self._first_audio_valid = torch.zeros(self.embeds.shape[0], dtype=torch.bool, device=codes.device)
        request_rows = index_to_device([entries[row][1] for row, _ in first], codes.device)
        if frame_valid is not None:
            self._first_audio_valid.index_copy_(0, request_rows, frame_valid)
        else:
            self._first_audio_valid.index_fill_(0, request_rows, True)
        stream.wait_stream(torch.cuda.current_stream(codes.device))
        sr = torch.tensor(decoder.sample_rate, dtype=torch.int32)
        with torch.cuda.stream(stream):
            frame_codes.record_stream(stream)
            pcm = decoder.decode(frame_codes)
            sender = self.first_audio_sender
            if sender is None:
                raise RuntimeError("First-frame decoder requires an engine output sink")
            if frame_valid is not None:
                frame_valid.record_stream(stream)
            request_ids = [entries[row][2] for row, _i in first]
            accepted = sender.submit(request_ids, pcm, sr, valid=frame_valid)
            self.first_audio_requests.update(accepted)

    def _init_mtp_runner(self, model: nn.Module) -> Any:
        mtp = self._raw_mtp or getattr(model, "mtp", None)
        if mtp is None:
            return None
        output_key = getattr(model, "mtp_output_key", None)
        if not (isinstance(output_key, str) or (isinstance(output_key, tuple) and len(output_key) == 2)):
            raise TypeError("Models with an mtp hook must declare mtp_output_key as a string or 2-tuple")

        compilation_config = self.owner.vllm_config.compilation_config
        cudagraph_mode = getattr(compilation_config, "cudagraph_mode", CUDAGraphMode.NONE)
        if bool(getattr(model, "mtp_disable_graph", False)):
            logger.info("Skipping mtp graph wrapper because the model marks it graph-unsafe.")
            return mtp
        graph_safe = bool(getattr(model, "mtp_graph_safe", False))
        if cudagraph_mode is not None and cudagraph_mode.has_full_cudagraphs() and graph_safe:
            graph_wrapper_cls = current_omni_platform.get_graph_wrapper_cls()
            return graph_wrapper_cls(mtp, self.owner.vllm_config, runtime_mode=CUDAGraphMode.FULL)
        return mtp

    def _is_mtp_graph_runner(self) -> bool:
        runner = getattr(self, "_mtp_runner", None)
        if runner is None:
            return False
        graph_wrapper_cls = current_omni_platform.get_graph_wrapper_cls()
        return isinstance(runner, graph_wrapper_cls)

    def capture_mtp_graphs(
        self, dispatch_batch_descriptor: Callable[[int], Any], *, restore_capturing: bool = False
    ) -> None:
        if not self._is_mtp_graph_runner():
            return
        if (
            self._mtp_input_ids is None
            or self._mtp_input_embeds is None
            or self._mtp_hidden is None
            or self._mtp_text_step is None
        ):
            return
        if getattr(self.owner.model, "mtp_accepts_req_infos", False):
            logger.warning("Skipping mtp graph capture because this model requires per-request req_infos.")
            return

        from vllm.compilation import monitor
        from vllm.distributed.parallel_state import graph_capture
        from vllm.forward_context import set_forward_context

        compilation_config = self.owner.vllm_config.compilation_config
        capture_sizes = self._get_mtp_capture_sizes()
        num_warmups = compilation_config.cudagraph_num_of_warmups
        capture_kwargs = self._get_mtp_base_sampling_kwargs()
        logger.info("Capturing MRv2 mtp graphs for sizes %s", capture_sizes)

        # The profiling pass runs this before the decoder's real capture, which
        # needs capturing to stay enabled, so it restores what was set.
        capturing_enabled = monitor.cudagraph_capturing_enabled and restore_capturing
        monitor.set_cudagraph_capturing_enabled(True)
        try:
            with torch.inference_mode(), graph_capture(device=self.owner.device):
                for bsz in capture_sizes:
                    batch_descriptor = dispatch_batch_descriptor(int(bsz))
                    num_tokens = int(getattr(batch_descriptor, "num_tokens", bsz))
                    ids = self._mtp_input_ids[:num_tokens]
                    emb = self._mtp_input_embeds[:num_tokens]
                    hidden = self._mtp_hidden[:num_tokens]
                    text_step = self._mtp_text_step[:num_tokens]

                    for _ in range(num_warmups):
                        with set_forward_context(
                            None,
                            self.owner.vllm_config,
                            cudagraph_runtime_mode=CUDAGraphMode.NONE,
                            batch_descriptor=batch_descriptor,
                        ):
                            self._call_mtp_runner(ids, emb, hidden, text_step, **capture_kwargs)

                    with set_forward_context(
                        None,
                        self.owner.vllm_config,
                        cudagraph_runtime_mode=CUDAGraphMode.FULL,
                        batch_descriptor=batch_descriptor,
                    ):
                        sample_uniforms = None
                        if self._mtp_sample_uniforms is not None:
                            sample_uniforms = self._mtp_sample_uniforms[:num_tokens]
                            sample_uniforms.fill_(0.5)
                        self._call_mtp_runner(
                            ids,
                            emb,
                            hidden,
                            text_step,
                            sample_uniforms=sample_uniforms,
                            **capture_kwargs,
                        )
                    torch.accelerator.synchronize()

            logger.info("Captured MRv2 mtp graphs for %d sizes", len(capture_sizes))
        except RuntimeError as e:
            raise RuntimeError(f"MRv2 mtp graph capture failed: {e}") from e
        finally:
            monitor.set_cudagraph_capturing_enabled(capturing_enabled)

    def _get_mtp_capture_sizes(self) -> list[int]:
        """Return graph buckets that a request-batched Talker can reach."""
        max_num_reqs = int(self.owner.scheduler_config.max_num_seqs)
        return sorted(
            {
                int(size)
                for size in self.owner.vllm_config.compilation_config.cudagraph_capture_sizes
                if 0 < int(size) <= max_num_reqs
            },
            reverse=True,
        )

    def _decode_mtp_offsets(
        self,
        decode_mtp_rows: list[tuple[int, int]],
        input_batch: InputBatch,
        device: torch.device,
    ) -> torch.Tensor:
        """Reuse MRv2's device query offsets for a contiguous decode prefix.

        The steady-state Talker batch contains one decode token from every
        scheduled request, in scheduler order. ``InputBatch.query_start_loc``
        already contains the exact device-side gather offsets for that shape;
        rebuilding it from a Python list creates a small synchronous H2D
        transfer on every decode step.
        """
        bsz = len(decode_mtp_rows)
        query_start_loc = getattr(input_batch, "query_start_loc", None)
        static_offsets = self._mtp_offsets
        if (
            isinstance(query_start_loc, torch.Tensor)
            and query_start_loc.device == device
            and isinstance(static_offsets, torch.Tensor)
            and static_offsets.device == device
            and bsz <= static_offsets.shape[0]
            and all(batch_index == expected for expected, (batch_index, _start) in enumerate(decode_mtp_rows))
        ):
            offsets = static_offsets[:bsz]
            offsets.copy_(query_start_loc[:bsz], non_blocking=True)
            return offsets

        return self._stage_offsets([start for _i, start in decode_mtp_rows], device)

    def _mtp_forward(
        self,
        req_indices: list[int],
        batch_ids: torch.Tensor,
        batch_emb: torch.Tensor,
        batch_hidden: torch.Tensor,
        batch_step: torch.Tensor,
        mtp_batch_descriptor_dispatcher: Callable[[int], Any] | None = None,
        req_ids: list[str] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run the residual-codebook predictor on packed rows; returns graph-owned outputs.

        A caller that already holds each row's request id (the buffers'
        ``req_id``) passes ``req_ids``; that also lets a steady batch replay
        its resolved graph directly.
        """
        from vllm.forward_context import set_forward_context

        bsz = len(req_indices)
        cache = self._mtp_generators
        seeded = self._mtp_seeded
        all_buffers = self.owner.intermediate_buffer.buffers
        device = batch_ids.device
        replay = self._mtp_replays.get(bsz) if mtp_batch_descriptor_dispatcher is not None else None
        if replay is not None and req_ids is not None:
            for req_id, req_idx in zip(req_ids, req_indices):
                if req_id not in cache:
                    # Rows new this step resolve their generator (usually "no seed") once.
                    self._get_mtp_generator(req_id, all_buffers[req_idx].get("sampling_params"), device)
            if seeded.isdisjoint(req_ids):
                # Steady batch of unseeded rows at a size whose graph is resolved:
                # what the wrapper path below does, minus its per-call Python.
                num_tokens, entry = replay
                assert self._mtp_sample_uniforms is not None
                self._mtp_sample_uniforms[:num_tokens].uniform_(1e-20, 1.0 - 1e-20)
                get_offloader().sync_prev_onload()
                entry.cudagraph.replay()
                return entry.output
        buffers = [all_buffers[req_idx] for req_idx in req_indices]
        if req_ids is None:
            req_ids = [str(buffer.get("req_id")) for buffer in buffers]
        if cache.keys() >= set(req_ids) and seeded.isdisjoint(req_ids):
            # Steady batch of resolved, unseeded requests: no per-row lookups.
            generators = [None] * bsz
        else:
            generators = [
                self._get_mtp_generator(req_id, buffer.get("sampling_params"), device)
                for req_id, buffer in zip(req_ids, buffers, strict=True)
            ]

        use_graph_runner = self._is_mtp_graph_runner()
        has_explicit_generator = any(generator is not None for generator in generators)
        sample_uniform_buffer = getattr(self, "_mtp_sample_uniforms", None)
        can_use_graph = use_graph_runner and (not has_explicit_generator or sample_uniform_buffer is not None)
        sample_uniforms = None
        batch_descriptor = None
        cudagraph_mode = CUDAGraphMode.NONE
        num_tokens = bsz
        if can_use_graph and mtp_batch_descriptor_dispatcher is not None:
            batch_descriptor = mtp_batch_descriptor_dispatcher(bsz)
            cudagraph_mode = getattr(batch_descriptor, "cg_mode", CUDAGraphMode.FULL)
            num_tokens = int(getattr(batch_descriptor, "num_tokens", bsz))
            assert self._mtp_input_ids is not None
            assert self._mtp_input_embeds is not None
            assert self._mtp_hidden is not None
            assert self._mtp_text_step is not None
            batch_ids = self._mtp_input_ids[:num_tokens]
            batch_emb = self._mtp_input_embeds[:num_tokens]
            batch_hidden = self._mtp_hidden[:num_tokens]
            batch_step = self._mtp_text_step[:num_tokens]
            if sample_uniform_buffer is not None:
                sample_uniforms = self._prepare_mtp_sample_uniforms(generators, num_tokens)
        elif can_use_graph:
            cudagraph_mode = CUDAGraphMode.FULL
            if sample_uniform_buffer is not None:
                sample_uniforms = self._prepare_mtp_sample_uniforms(generators, bsz)
        with set_forward_context(
            None,
            self.owner.vllm_config,
            num_tokens=num_tokens,
            cudagraph_runtime_mode=cudagraph_mode,
            batch_descriptor=batch_descriptor,
        ):
            new_emb, codes = self._call_mtp_with_sampling(
                batch_ids,
                batch_emb,
                batch_hidden,
                batch_step,
                buffers=buffers,
                req_ids=req_ids,
                generators=generators,
                sample_uniforms=sample_uniforms,
            )
        if batch_descriptor is not None and cudagraph_mode == CUDAGraphMode.FULL and sample_uniforms is not None:
            self._remember_mtp_replay(bsz, num_tokens, batch_descriptor, has_explicit_generator)
        return new_emb, codes

    def _remember_mtp_replay(self, bsz: int, num_tokens: int, batch_descriptor: Any, seeded: bool) -> None:
        """Record the captured graph entry the wrapper replayed for ``bsz`` unseeded rows."""
        runner = self._mtp_runner
        entries = getattr(runner, "concrete_cudagraph_entries", None)
        if seeded or not isinstance(entries, dict) or getattr(runner, "is_debugging_mode", True):
            return
        entry = entries.get(batch_descriptor)
        if getattr(entry, "cudagraph", None) is not None and getattr(entry, "output", None) is not None:
            self._mtp_replays[bsz] = (num_tokens, entry)

    def _get_mtp_base_sampling_kwargs(self) -> dict[str, Any]:
        sampling_params = getattr(self.owner.model, "mtp_sampling_params", None)
        if not isinstance(sampling_params, dict):
            sampling_params = {}
        return {
            "do_sample": sampling_params.get("do_sample"),
            "temperature": sampling_params.get("temperature"),
            "top_k": sampling_params.get("top_k"),
            "top_p": sampling_params.get("top_p"),
        }

    def _get_mtp_sampling_kwargs(
        self,
        buffers: list[dict[str, Any]],
        req_ids: list[str],
        generators: list[torch.Generator | None],
    ) -> dict[str, Any]:
        kwargs = self._get_mtp_base_sampling_kwargs()

        if len(generators) == 1:
            if generators[0] is not None:
                kwargs["generator"] = generators[0]
        elif any(generator is not None for generator in generators):
            kwargs["generators"] = generators

        if getattr(self.owner.model, "mtp_accepts_req_infos", False):
            kwargs["req_ids"] = req_ids
            kwargs["req_infos"] = buffers

        return kwargs

    def _get_mtp_generator(
        self,
        req_id: str,
        sampling_params: Any,
        device: torch.device,
    ) -> torch.Generator | None:
        cache = self._mtp_generators
        generator = cache.get(req_id, _NO_SEED)
        if generator is None:
            # Resolved earlier: this request has no seed.
            return None
        if generator is not _NO_SEED and generator.device == device:
            return generator
        resolve_seed = getattr(self.owner.model, "get_mtp_seed", None)
        seed = resolve_seed(sampling_params) if resolve_seed is not None else None
        if seed is None:
            cache[req_id] = None
            return None
        generator = cache.get(req_id)
        if generator is None or generator.device != device:
            generator = torch.Generator(device=device)
            generator.manual_seed(seed)
            cache[req_id] = generator
        self._mtp_seeded.add(req_id)
        return generator

    def _prepare_mtp_sample_uniforms(
        self,
        generators: list[torch.Generator | None],
        num_tokens: int,
    ) -> torch.Tensor | None:
        sample_uniforms = self._mtp_sample_uniforms
        if sample_uniforms is None:
            return None
        bsz = len(generators)
        if not 0 < bsz <= num_tokens <= sample_uniforms.shape[0]:
            raise ValueError(
                f"invalid Talker-MTP sampling buffer shape: bsz={bsz} num_tokens={num_tokens} "
                f"capacity={sample_uniforms.shape[0]}"
            )

        active = sample_uniforms[:bsz]
        if any(generator is not None for generator in generators):
            # Consume one request-local stream for the whole residual span so the
            # outer MTP graph can treat sampling noise as a stable tensor input.
            # The draw granularity differs from the legacy per-step path, but
            # request-local determinism and batch-order independence are kept.
            for row, generator in enumerate(generators):
                active[row : row + 1].uniform_(1e-20, 1.0 - 1e-20, generator=generator)
            if num_tokens > bsz:
                sample_uniforms[bsz:num_tokens].uniform_(1e-20, 1.0 - 1e-20)
        else:
            # Active and graph-padding rows in one draw.
            sample_uniforms[:num_tokens].uniform_(1e-20, 1.0 - 1e-20)
        return sample_uniforms[:num_tokens]

    def _call_mtp_with_sampling(
        self,
        batch_ids: torch.Tensor,
        batch_emb: torch.Tensor,
        batch_hidden: torch.Tensor,
        batch_step: torch.Tensor,
        *,
        buffers: list[dict[str, Any]],
        req_ids: list[str],
        generators: list[torch.Generator | None],
        sample_uniforms: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if sample_uniforms is None:
            kwargs = self._get_mtp_sampling_kwargs(buffers, req_ids, generators)
        else:
            kwargs = self._get_mtp_base_sampling_kwargs()
            kwargs["sample_uniforms"] = sample_uniforms
            if getattr(self.owner.model, "mtp_accepts_req_infos", False):
                kwargs["req_ids"] = req_ids
                kwargs["req_infos"] = buffers

        if (
            len(req_ids) > 1
            and "generators" in kwargs
            and "sample_uniforms" not in kwargs
            and not getattr(self.owner.model, "mtp_accepts_per_row_generators", False)
        ):
            emb_chunks = []
            code_chunks = []
            for row, (req_id, buffer, generator) in enumerate(zip(req_ids, buffers, generators, strict=True)):
                row_kwargs = self._get_mtp_sampling_kwargs(
                    [buffer],
                    [req_id],
                    [generator],
                )
                row_emb, row_codes = self._call_mtp_runner(
                    batch_ids[row : row + 1],
                    batch_emb[row : row + 1],
                    batch_hidden[row : row + 1],
                    batch_step[row : row + 1],
                    **row_kwargs,
                )
                emb_chunks.append(row_emb)
                if row_codes is not None:
                    code_chunks.append(row_codes)
            new_emb = torch.cat(emb_chunks)
            codes = torch.cat(code_chunks) if code_chunks else None
            return new_emb, codes

        return self._call_mtp_runner(
            batch_ids,
            batch_emb,
            batch_hidden,
            batch_step,
            **kwargs,
        )

    def _call_mtp_runner(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor,
        last_hidden: torch.Tensor,
        text_step: torch.Tensor,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # Explicit generators are request state, while the outer whole-MTP
        # graph owns one captured RNG stream. Keep the transformer/code-
        # predictor graphs, but call the raw MTP function so per-row streams
        # remain deterministic and independent of scheduler batch makeup.
        has_explicit_generator = kwargs.get("generator") is not None or kwargs.get("generators") is not None
        mtp = self._raw_mtp or self.owner.model.mtp
        runner = mtp if has_explicit_generator else self._mtp_runner
        runner = runner or mtp
        return runner(input_ids, input_embeds, last_hidden, text_step, **kwargs)
