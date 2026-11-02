# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CosyVoice3's MRv2 sampling-to-audio lifecycle through post_sample."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from vllm_omni.model_executor.models.cosyvoice3.async_codec import AsyncCodec, SampleEntry
from vllm_omni.worker_v2.model_states.omni_model_state import OmniModelState


class CosyVoice3ModelState(OmniModelState):
    def __init__(self, vllm_config, model, encoder_cache, device):
        if vllm_config.speculative_config is not None:
            raise ValueError("CosyVoice3 single-stage does not yet support speculative decoding")
        super().__init__(vllm_config, model, encoder_cache, device)
        self.vocab_size = int(model.config.llm["speech_token_size"])
        self.codec_max_model_len = int(vllm_config.model_config.max_model_len)
        self.codec = None
        self.codec_device = device
        self.codec_pending_audio_threshold = max(2, int(vllm_config.max_concurrent_batches) * 2)

    def bind_async_audio(self, scheduler):
        from .async_scheduler import CosyVoice3AsyncScheduler

        if not isinstance(scheduler, CosyVoice3AsyncScheduler):
            raise ValueError("CosyVoice3 single-stage requires CosyVoice3AsyncScheduler")
        self.model._maybe_enable_code2wav_trt()
        self.model.code2wav.keep_stream_cache_on_device = True
        self.codec = AsyncCodec(
            self.model.code2wav,
            self.codec_device,
            self.codec_max_model_len,
            self.codec_pending_audio_threshold,
            chunk_size=int(getattr(self.model.config, "codec_chunk_frames", 25)),
            lookahead=int(self.model.config.flow["pre_lookahead_len"]),
        )
        # Share the existing request table; it remains live through codec
        # tails and is removed on final delivery/cancellation.
        self.audio_requests = scheduler.audio_requests
        scheduler.codec = self.codec
        scheduler.audio_sample_rate = int(self.model.config.sample_rate)

    def close_first_audio_sender(self):
        if self.codec is not None:
            self.codec.close()

    def prepare_inputs(self, input_batch, req_states):
        # Borrow runner-owned state; do not copy limits into request/batch buffers.
        self.req_states = req_states
        return super().prepare_inputs(input_batch, req_states)

    def post_sample(
        self,
        input_batch,
        text_hidden,
        sampled_token_ids,
        multimodal_outputs,
        mtp_batch_descriptor_dispatcher=None,
    ):
        if sampled_token_ids.shape[1] != 1:
            raise ValueError("CosyVoice3 single-stage expects one sampled token per request")
        count = input_batch.num_reqs
        scheduled = input_batch.num_scheduled_tokens[:count]
        is_last_prefill = (
            input_batch.num_computed_prefill_tokens_np[:count] + scheduled >= input_batch.prefill_len_np[:count]
        )
        kept = ~input_batch.is_prefilling_np[:count] | is_last_prefill
        payloads = self.intermediate_buffer.gather(input_batch)
        limits = np.minimum(self.req_states.max_seq_len[input_batch.idx_mapping_np[:count]], self.codec_max_model_len)
        end = input_batch.num_computed_tokens_np[:count] + scheduled
        positions = end - self.req_states.prompt_len.np[input_batch.idx_mapping_np[:count]]
        length_finished = end + 1 >= limits
        tokens = sampled_token_ids[:count, 0]
        token_mask = (tokens >= 0) & (tokens < self.vocab_size)
        step_embed = multimodal_outputs.get("embed", {})
        entries, rows = [], []
        for row, payload in enumerate(payloads):
            req_id = input_batch.req_ids[row]
            # Retain live conditioning even on an intermediate prefill step.
            embed = payload.setdefault("embed", {})
            if isinstance(step_embed, Mapping):
                for key in ("speech_token", "speech_feat", "embedding", "speech_token_len"):
                    values = step_embed.get(key)
                    if isinstance(values, (list, tuple)) and row < len(values) and values[row] is not None:
                        embed[key] = values[row]
            if not kept[row] or str(req_id).startswith("_warmup_"):
                continue
            if self.codec is None:
                raise RuntimeError("CosyVoice3 codec worker is not bound to the engine")
            if req_id not in self.audio_requests:
                # An already retired request can have an in-flight AR row.
                continue
            entries.append(
                SampleEntry(
                    req_id,
                    int(positions[row]),
                    bool(length_finished[row]),
                    dict(embed),
                )
            )
            rows.append(row)
        if entries:
            # Copy basic row slices into owned buffers on this stream before
            # graph replay. A Python index list on CUDA tensors would upload
            # indices synchronously, twice per step.
            self.codec.submit(entries, tokens, token_mask, rows)
        # PCM is published separately, never through this step's AR future.
        multimodal_outputs.clear()

    def on_requests_finished(self, req_ids: set[str]) -> None:
        # Normal EOS tails are owned by the codec until final PCM delivery.
        super().on_requests_finished(req_ids)
