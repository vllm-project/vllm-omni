# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Deliver first PCM from the regular batched MTP before the next backbone.

First and subsequent frames retain the normal MTP batching and sampling.
Only PCM decoding/delivery bypasses the regular codec stage's output queue.
"""

from __future__ import annotations

import torch

from vllm_omni.model_executor.models.output_templates import RequestBatchTensor


def _indices(values, device):
    if device.type == "cuda":
        return torch.tensor(values, dtype=torch.long, pin_memory=True).to(device, non_blocking=True)
    return torch.tensor(values, dtype=torch.long, device=device)


class MossEarlyFirstAudioState:
    def __init__(self, owner, decoder):
        self.owner = owner
        self.decoder = decoder
        self.waiting = set()
        self.seen = set()
        self.delivered = {}
        self.updates = {}
        self.stream = None
        owner._first_audio_sender = None

    def set_sink(self, sink):
        from vllm_omni.worker_v2.first_audio_sender import FirstAudioSender

        self.owner._first_audio_sender = FirstAudioSender(sink)

    def remove(self, req_id):
        self.waiting.discard(req_id)
        self.seen.discard(req_id)
        self.delivered.pop(req_id, None)
        self.updates.pop(req_id, None)

    def record_prefills(self, batch, entries):
        if self.owner._first_audio_sender is None:
            return
        for i, req_idx, start, n_tok, info, is_prefill in entries:
            rid = str(info["req_id"])
            prompt = info.get("_omni_prompt_len")
            computed = info.get("_omni_num_computed_tokens")
            cap = getattr(info.get("sampling_params"), "max_tokens", None)
            if (
                is_prefill
                and rid not in self.seen
                and prompt is not None
                and computed is not None
                and computed + n_tok >= prompt
                and cap is not None
                and cap > 1
            ):
                self.waiting.add(rid)

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

    def after_mtp(self, request_ids, codes, input_ids):
        if not self.waiting:
            return
        selected = [(i, rid) for i, rid in enumerate(request_ids) if rid in self.waiting]
        if not selected:
            return
        ids = [rid for _, rid in selected]
        rows = _indices([i for i, _ in selected], codes.device)
        # Own the gathered codes before another MTP replay can reuse storage.
        first_codes = codes.index_select(0, rows)
        valid = first_codes.ne(self.owner.model.audio_pad_token_id).any(dim=1)
        valid &= input_ids.reshape(-1).index_select(0, rows).eq(self.owner.model.audio_assistant_slot_token_id)
        accepted = set(self._publish(ids, first_codes, valid))
        for rid, row_valid in zip(ids, valid.unbind(), strict=True):
            self.waiting.discard(rid)
            self.seen.add(rid)
            if rid in accepted:
                self.delivered[rid] = row_valid
                self.updates[rid] = row_valid

    def after_sample(self, batch, hidden, sampled, num_sampled, outputs, dispatcher):
        if not self.updates:
            return outputs
        outputs = dict(outputs)
        outputs["meta"] = dict(outputs.get("meta", {}))
        zero = hidden.new_zeros((), dtype=torch.bool)
        flags = torch.stack([self.delivered.get(rid, zero) for rid in batch.req_ids])
        outputs["meta"]["first_audio"] = RequestBatchTensor(flags, keepdim=False)
        self.updates.clear()
        return outputs
