# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Opt-in completion of a Talker frame immediately after sampling codebook zero."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from vllm.logger import init_logger
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.states import RequestState

from vllm_omni.utils.device_copy import index_to_device
from vllm_omni.worker_v2.streaming_audio import StreamingAudioBuffer, StreamingAudioOutput

if TYPE_CHECKING:
    from vllm_omni.worker_v2.model_states.omni_model_state import OmniModelState

logger = init_logger(__name__)


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


@dataclass
class TalkerInputs:
    embeds: torch.Tensor
    length: int = 0


@dataclass
class SuspendedAudio:
    position: int
    decoder_state: list[torch.Tensor]
    generator: torch.Generator | None
    runtime: dict[str, Any]
    frame_embed: torch.Tensor


class EagerMTPState:
    """Used only by models explicitly declaring ``mtp_eager_frames``."""

    def __init__(self, owner: OmniModelState) -> None:
        self.owner = owner
        self._first_audio_valid: torch.Tensor | None = None
        self._side_stream: torch.cuda.Stream | None = None
        self._fast_ok: bool | None = None
        self._audio_buffer: StreamingAudioBuffer | None = None
        self._frame_requests: set[str] = set()
        self._suspended_audio: dict[str, SuspendedAudio] = {}
        self._restore_audio: dict[str, list[torch.Tensor]] = {}
        self._talker_inputs: dict[str, TalkerInputs] = {}

    def suspend_audio(self, req_id: str, req_idx: int) -> None:
        decoder = getattr(self.owner.model, "stream_decoder", None)
        position = self.owner._stream_pos.get(req_id)
        if decoder is None or position is None:
            return
        # Slot reuse is rare and must wait for the last side-stream write.
        saved = self._restore_audio.pop(req_id, None)
        if saved is None:
            self._decode_stream.synchronize()
            saved = decoder.save_slot(req_idx)
        history = self._talker_inputs[req_id]
        history.embeds = history.embeds[: history.length].to("cpu", copy=True)
        runtime = self.owner.intermediate_buffer.buffers[req_idx]
        assert self.owner._eager_embeds is not None
        self._suspended_audio[req_id] = SuspendedAudio(
            position,
            saved,
            self.owner._mtp_generators.get(req_id),
            _snapshot_runtime(runtime),
            self.owner._eager_embeds[req_idx].to("cpu", copy=True),
        )

    def resume_audio(self, req_id: str, req_idx: int) -> None:
        saved = self._suspended_audio.pop(req_id, None)
        if saved is not None:
            self.owner._stream_pos[req_id] = saved.position
            self.owner._mtp_generators[req_id] = saved.generator
            self._restore_audio[req_id] = saved.decoder_state
            self.owner.intermediate_buffer.buffers[req_idx] = saved.runtime
            assert self.owner._eager_embeds is not None
            self.owner._eager_embeds[req_idx].copy_(saved.frame_embed)
            self.owner._eager_ready[req_idx] = req_id

    def finish_audio(self, req_ids: set[str]) -> None:
        for req_id in req_ids:
            self._suspended_audio.pop(req_id, None)
            self._restore_audio.pop(req_id, None)
            self.owner._stream_pos.pop(req_id, None)
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
            assert self.owner._eager_embeds is not None
            embeds[count:].copy_(self.owner._eager_embeds[req_idx] + mtp_inputs[1].reshape(1, -1))
            self.owner.intermediate_buffer.update(req_idx, updates, self.owner.model.gpu_resident_buffer_keys)
        return True

    def record_inputs(self, input_batch: InputBatch, embeds: torch.Tensor) -> None:
        if getattr(self.owner.model, "stream_decoder", None) is None:
            return
        history_slices: list[torch.Tensor] = []
        input_slices: list[torch.Tensor] = []
        histories = self._talker_inputs
        on_cpu = embeds.is_cpu
        for i, req_id in enumerate(input_batch.req_ids):
            start, end = input_batch.query_start_loc_np[i : i + 2]
            offset = int(input_batch.num_computed_tokens_np[i])
            length = offset + int(end - start)
            history = histories.get(req_id)
            # A suspended request's inputs wait on the host; bring them back.
            if history is None or history.embeds.is_cpu != on_cpu:
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
            history_slices.append(history.embeds[offset:length])
            input_slices.append(embeds[start:end])
            history.length = max(history.length, length)
        if history_slices:
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
        event = self.owner._stream_decode_event
        self.owner._stream_decode_event = None
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

        self.owner._first_audio_sender = FirstAudioSender(sink)

    def _apply_eager_frames(
        self,
        mtp_batches: list[tuple[int, int, tuple[torch.Tensor, torch.Tensor]]],
        embeds: torch.Tensor,
        input_batch: InputBatch,
        prepacked_mtp_inputs: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> None:
        """Decode input = codec embeddings of the previous (eager) frame + this step's text."""
        req_indices = [int(input_batch.idx_mapping_np[i]) for i, _start, _mtp in mtp_batches]
        for req_idx in req_indices:
            req_id = self.owner.intermediate_buffer.buffers[req_idx].get("req_id")
            if self.owner._eager_ready.get(req_idx) != req_id:
                # Running the deferred MTP here would re-emit or drop a frame.
                raise RuntimeError(f"Eager Talker-MTP frame missing for request {req_id!r}")
        assert self.owner._eager_embeds is not None
        device = embeds.device
        if prepacked_mtp_inputs is None:
            text_step = torch.cat([step.reshape(1, -1) for _i, _start, (_hidden, step) in mtp_batches], dim=0)
        else:
            text_step = prepacked_mtp_inputs[1].reshape(len(mtp_batches), -1)
        rows = index_to_device(req_indices, device)
        offsets = self.owner._mtp_batch_offsets(mtp_batches, input_batch, device)
        frame_embeds = self.owner._eager_embeds.index_select(0, rows)
        embeds.index_copy_(0, offsets, (frame_embeds + text_step.to(frame_embeds.dtype)).to(embeds.dtype))

    def _apply_settled_frames(self, settled_rows: list[tuple[int, int, int, str]], embeds: torch.Tensor) -> None:
        """Settled decode rows: previous eager frame embedding plus the constant text step."""
        for _i, req_idx, _start, req_id in settled_rows:
            if self.owner._eager_ready.get(req_idx) != req_id:
                raise RuntimeError(f"Eager Talker-MTP frame missing for request {req_id!r}")
        assert self.owner._eager_embeds is not None
        device = embeds.device
        rows = index_to_device([req_idx for _i, req_idx, _start, _req_id in settled_rows], device)
        offsets = index_to_device([start for _i, _req_idx, start, _req_id in settled_rows], device)
        text_step = self.owner.model.eager_settled_text_step().to(device=device, dtype=self.owner._eager_embeds.dtype)
        frame_embeds = self.owner._eager_embeds.index_select(0, rows)
        embeds.index_copy_(0, offsets, (frame_embeds + text_step).to(embeds.dtype))

    def run_eager_mtp(
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
        if not self.owner._eager_mtp:
            return
        recorded, self.owner._eager_rows = self.owner._eager_rows, None
        if recorded is None or recorded[0] is not input_batch or not recorded[1]:
            return
        entries, input_ids = recorded[1], recorded[2]
        codes_out = multimodal_outputs.get("codes", {}).get("audio")
        valid_out = multimodal_outputs.get("meta", {}).get("codec_frame_valid")
        if not isinstance(codes_out, torch.Tensor) or not isinstance(valid_out, torch.Tensor):
            raise RuntimeError("Eager Talker-MTP requires retained codes.audio and codec_frame_valid outputs")
        assert self.owner._mtp_input_ids is not None and self.owner._mtp_input_embeds is not None
        assert self.owner._mtp_hidden is not None and self.owner._mtp_text_step is not None
        assert self.owner._eager_embeds is not None

        bsz = len(entries)
        device = text_hidden.device
        if self._fast_path_ok(codes_out, text_hidden):
            self._run_eager_mtp_fused(
                input_batch, text_hidden, sampled_token_ids, multimodal_outputs, entries, input_ids,
                codes_out, valid_out, mtp_batch_descriptor_dispatcher,
            )  # fmt: skip
            return
        if getattr(self.owner.model, "stream_decoder", None) is not None:
            raise RuntimeError("Talker stream decode requires the fused eager-MTP path")
        batch_rows = index_to_device([i for i, _req_idx, _req_id, _prefill in entries], device)
        req_indices = [req_idx for _i, req_idx, _req_id, _prefill in entries]
        last_tokens = input_batch.query_start_loc.index_select(0, batch_rows + 1).long() - 1
        layer0 = sampled_token_ids.reshape(input_batch.num_reqs, -1)[:, 0].index_select(0, batch_rows)

        batch_ids = self.owner._mtp_input_ids[:bsz]
        batch_emb = self.owner._mtp_input_embeds[:bsz]
        batch_hidden = self.owner._mtp_hidden[:bsz]
        batch_step = self.owner._mtp_text_step[:bsz]
        batch_ids.copy_(layer0.to(batch_ids.dtype))
        batch_emb.copy_(self.owner.model.embed_input_ids(batch_ids.reshape(-1, 1)).reshape(bsz, -1))
        torch.index_select(text_hidden, 0, last_tokens, out=batch_hidden)
        # The text step is added by the next step's preprocess.
        batch_step.zero_()
        frame_embeds, codes = self.owner._mtp_forward(
            req_indices,
            batch_ids,
            batch_emb,
            batch_hidden,
            batch_step,
            mtp_batch_descriptor_dispatcher,
        )
        assert codes is not None
        rows = index_to_device(req_indices, device)
        self.owner._eager_embeds.index_copy_(
            0, rows, frame_embeds[:bsz].reshape(bsz, -1).to(self.owner._eager_embeds.dtype)
        )
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
            self.owner._eager_ready[req_idx] = req_id
        self._publish_first_audio(input_batch, entries, codes[:bsz], valid)
        first_audio = multimodal_outputs.get("meta", {}).get("first_audio")
        if isinstance(first_audio, torch.Tensor):
            scheduled = index_to_device(
                [int(req_id in self.owner._first_audio_requests) for _, _, req_id, _ in entries], device
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

    def _run_eager_mtp_fused(
        self,
        input_batch: InputBatch,
        text_hidden: torch.Tensor,
        sampled_token_ids: torch.Tensor,
        multimodal_outputs: dict[str, Any],
        entries: list[tuple[int, int, str, bool]],
        input_ids: torch.Tensor,
        codes_out: torch.Tensor,
        valid_out: torch.Tensor,
        mtp_batch_descriptor_dispatcher: Callable[[int], Any] | None,
    ) -> None:
        from vllm_omni.worker_v2.model_states.eager_mtp_kernels import eager_post, eager_pre

        owner = self.owner
        bsz = len(entries)
        device = text_hidden.device
        first_audio = multimodal_outputs.get("meta", {}).get("first_audio")
        fa_requests = owner._first_audio_requests
        model = owner.model
        stream = getattr(model, "stream_decoder", None)
        stream_out = multimodal_outputs.get("model_outputs") if stream is not None else None
        pos_list: list[int] = []
        first_rows: list[int] = []
        primes: list[tuple[int, torch.Tensor]] = []
        if isinstance(stream_out, torch.Tensor):
            assert stream is not None
            # Frame index of each row's frame in its request's stream; 0 starts
            # a fresh decoder state in the request's slot. A voice-clone
            # request's stream starts after its reference codes, which prime
            # the decoder state first (as Code2Wav's first-chunk context does).
            if self._audio_buffer is None:
                self._audio_buffer = StreamingAudioBuffer(int(stream.spf), int(model.stream_chunk_frames))
            self._frame_requests = {req_id for _i, _idx, req_id, _p in entries}
            positions = owner._stream_pos
            for row, (_i, req_idx, req_id, _p) in enumerate(entries):
                self._audio_buffer.add(req_id)
                p = positions.get(req_id)
                if p is None:
                    first_rows.append(row)
                    ref = model.get_stream_ref_context(owner.intermediate_buffer.buffers[req_idx])
                    p = 0
                    if ref is not None:
                        primes.append((req_idx, ref))
                        p = int(ref.shape[0])
                pos_list.append(p)
                positions[req_id] = p + 1
        # One transpose of the entry tuples instead of a Python pass per field.
        rows_i, rows_req_idx, rows_req_id, rows_prefill = zip(*entries) if entries else ((), (), (), ())
        meta_list = [
            *rows_i,
            *rows_req_idx,
            *map(int, rows_prefill),
            *[int(req_id in fa_requests) for req_id in rows_req_id],
            *pos_list,
        ]
        meta = index_to_device(meta_list, device)
        rows = meta[bsz : 2 * bsz]
        sampled = sampled_token_ids.reshape(input_batch.num_reqs, -1)
        assert owner._mtp_input_ids is not None and owner._mtp_input_embeds is not None
        assert owner._mtp_hidden is not None and owner._mtp_text_step is not None
        batch_ids = owner._mtp_input_ids[:bsz]
        batch_emb = owner._mtp_input_embeds[:bsz]
        batch_hidden = owner._mtp_hidden[:bsz]
        batch_step = owner._mtp_text_step[:bsz]
        last_tokens, layer0 = eager_pre(
            meta, bsz, input_batch.query_start_loc, sampled, text_hidden, self._embedding_weight(),
            batch_ids, batch_emb, batch_hidden, batch_step,
        )  # fmt: skip
        req_indices = meta_list[bsz : 2 * bsz]
        frame_embeds, codes = owner._mtp_forward(
            req_indices, batch_ids, batch_emb, batch_hidden, batch_step, mtp_batch_descriptor_dispatcher
        )
        assert codes is not None
        has_prefill = any(rows_prefill)
        fa_in_kernel = isinstance(first_audio, torch.Tensor) and not has_prefill
        valid = eager_post(
            meta, bsz, last_tokens, layer0, frame_embeds.reshape(frame_embeds.shape[0], -1).contiguous(),
            owner._eager_embeds, codes.contiguous(), codes_out, input_ids, valid_out,
            first_audio if fa_in_kernel else None, self._first_audio_valid if fa_in_kernel else None,
            owner.model._codebook_vocab_size,
        )  # fmt: skip
        owner._eager_ready.update(zip(rows_req_idx, rows_req_id))
        if pos_list:
            assert stream is not None
            assert isinstance(stream_out, torch.Tensor)
            decode = model.stream_graphs if model.stream_graphs is not None else stream
            frame_codes = codes[:bsz].reshape(bsz, 1, -1).to(torch.int32)
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
                pcm = decode(frame_codes, rows, meta[4 * bsz : 5 * bsz])
                stream_out.index_copy_(0, last_tokens, pcm.reshape(bsz, -1).to(stream_out.dtype))
                self._send_stream_first_frames(entries, pcm, valid, first_rows)
                done = torch.cuda.Event()
                done.record(side)
            owner._stream_decode_event = done
        if has_prefill:
            self._publish_first_audio(input_batch, entries, codes[:bsz], valid)
            if isinstance(first_audio, torch.Tensor):
                # Requests accepted by this step's publish count as scheduled.
                scheduled = index_to_device(
                    [int(q in owner._first_audio_requests) for _i, _r, q, _p in entries], device
                )
                if self._first_audio_valid is None:
                    scheduled.zero_()
                else:
                    scheduled *= self._first_audio_valid.index_select(0, rows)
                first_audio.index_copy_(0, last_tokens, scheduled.to(first_audio.dtype))

    def _send_stream_first_frames(
        self,
        entries: list[tuple[int, int, str, bool]],
        pcm: torch.Tensor,
        valid: torch.Tensor,
        first_rows: list[int],
    ) -> None:
        sender = self.owner._first_audio_sender
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
            and _req_id not in self.owner._first_audio_requests
            and not _has_ref_codes(self.owner.intermediate_buffer.buffers[req_idx])
        ]
        if not first:
            return
        # Decode on a side stream so the Talker's next step does not queue
        # behind it; the async output copy waits on its event instead.
        stream = self.owner._first_audio_stream
        if stream is None:
            _least, greatest = torch.cuda.Stream.priority_range()
            stream = self.owner._first_audio_stream = torch.cuda.Stream(device=codes.device, priority=greatest)
        frame_rows = index_to_device([row for row, _i in first], codes.device)
        frame_codes = codes.index_select(0, frame_rows)
        frame_valid = valid.index_select(0, frame_rows) if valid is not None else None
        if frame_valid is not None:
            frame_codes = torch.where(frame_valid[:, None], frame_codes, 0)
        if self._first_audio_valid is None:
            assert self.owner._eager_embeds is not None
            self._first_audio_valid = torch.zeros(
                self.owner._eager_embeds.shape[0], dtype=torch.bool, device=codes.device
            )
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
            sender = self.owner._first_audio_sender
            if sender is None:
                raise RuntimeError("First-frame decoder requires an engine output sink")
            if frame_valid is not None:
                frame_valid.record_stream(stream)
            request_ids = [entries[row][2] for row, _i in first]
            accepted = sender.submit(request_ids, pcm, sr, valid=frame_valid)
            self.owner._first_audio_requests.update(accepted)
