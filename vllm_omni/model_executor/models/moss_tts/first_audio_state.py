# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""First PCM from normal batched Local MTP, before the next backbone step."""

import torch
from vllm.sampling_params import SamplingParams


class MossEarlyFirstAudioState:
    def __init__(self, owner, decoder):
        self.owner = owner
        self.decoder = decoder
        self.waiting = set()
        self.seen = set()
        self.delivered = {}
        self.updates = set()
        self.stream = None
        owner._first_audio_sender = None

    def set_sink(self, sink):
        from vllm_omni.worker_v2.first_audio_sender import FirstAudioSender

        self.owner._first_audio_sender = FirstAudioSender(sink)

    def remove(self, request_id):
        self.waiting.discard(request_id)
        self.seen.discard(request_id)
        self.delivered.pop(request_id, None)
        self.updates.discard(request_id)

    def record_prefill(self, request_id, sampling_params: SamplingParams | None, *, prompt_len: int):
        if self.owner._first_audio_sender is None or request_id in self.seen or sampling_params is None:
            return
        config = self.owner.vllm_config.model_config
        cap = sampling_params.max_tokens
        if cap is None or cap <= 1 or prompt_len + 1 >= config.max_model_len:
            return
        # Final prefill normally produces an audio-slot token before any codes
        # enter the regular output path. Do not publish ahead of its stop check.
        first_token = self.owner.model.audio_assistant_slot_token_id
        if first_token == sampling_params.eos_token_id or first_token in (sampling_params.stop_token_ids or ()):
            return
        # Constraints can change that token or stop at the output processor.
        # Leave those requests entirely on the canonical delivery path.
        if (
            sampling_params.stop
            or sampling_params.min_tokens
            or sampling_params.allowed_token_ids is not None
            or sampling_params.bad_words
            or sampling_params.logit_bias
            or sampling_params.structured_outputs is not None
            or sampling_params.repetition_detection is not None
            or sampling_params.thinking_token_budget is not None
            or sampling_params.trace_decode_token_ids is not None
            or config.logits_processors
        ):
            return
        self.waiting.add(request_id)

    def _publish(self, ids, codes, valid):
        if self.stream is None:
            _, greatest = torch.cuda.Stream.priority_range()
            self.stream = torch.cuda.Stream(device=codes.device, priority=greatest)
        stream = self.stream
        frame_codes = torch.where(valid[:, None], codes, 0)
        stream.wait_stream(torch.cuda.current_stream(codes.device))
        with torch.cuda.stream(stream):
            frame_codes.record_stream(stream)
            valid.record_stream(stream)
            pcm = self.decoder.decode(frame_codes)
            return self.owner._first_audio_sender.submit(ids, pcm, self.decoder.sample_rate, valid=valid)

    def after_mtp(self, request_ids, codes, input_ids=None):
        selected = [(i, rid) for i, rid in enumerate(request_ids) if rid in self.waiting]
        if not selected:
            return
        ids = [rid for _, rid in selected]
        rows = torch.tensor([i for i, _ in selected], dtype=torch.long, pin_memory=codes.is_cuda).to(
            codes.device, non_blocking=True
        )
        first_codes = codes.index_select(0, rows)
        valid = first_codes.ne(self.owner.model.audio_pad_token_id).any(dim=1)
        # Preprocess MTP also checks the current token. Eager MTP runs after
        # the forward and has already masked non-emitting rows in its codes.
        if input_ids is not None:
            valid &= input_ids.reshape(-1).index_select(0, rows).eq(self.owner.model.audio_assistant_slot_token_id)
        accepted = set(self._publish(ids, first_codes, valid))
        for rid, row_valid in zip(ids, valid.unbind(), strict=True):
            self.waiting.discard(rid)
            self.seen.add(rid)
            if rid in accepted:
                self.delivered[rid] = row_valid
                self.updates.add(rid)

    def take_flags(self, request_ids, device):
        if not self.updates:
            return None
        zero = torch.zeros((), dtype=torch.bool, device=device)
        flags = torch.stack([self.delivered.get(rid, zero) for rid in request_ids])
        self.updates.clear()
        return flags
