# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Opt-in completion of a Talker frame immediately after sampling codebook zero."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch
from vllm.logger import init_logger
from vllm.v1.worker.gpu.input_batch import InputBatch

from vllm_omni.utils.device_copy import index_to_device

if TYPE_CHECKING:
    from vllm_omni.worker_v2.model_states.omni_model_state import OmniModelState

logger = init_logger(__name__)


def _has_ref_codes(buffer: dict[str, Any]) -> bool:
    ref = buffer.get("codes", {}).get("ref") if isinstance(buffer.get("codes"), dict) else None
    return isinstance(ref, torch.Tensor) and ref.numel() > 0


class EagerMTPState:
    """Used only by models explicitly declaring ``mtp_eager_frames``."""

    def __init__(self, owner: OmniModelState) -> None:
        self.owner = owner
        self._first_audio_valid: torch.Tensor | None = None

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
