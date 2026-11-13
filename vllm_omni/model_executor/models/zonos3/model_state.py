# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Native MRV2 request state for the Zonos3 AR and Codec stages."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.model_states.interface import ModelSpecificAttnMetadata
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.states import RequestState
from vllm.v1.worker.utils import AttentionGroup

from vllm_omni.utils.device_copy import index_to_device, to_device_nonblocking
from vllm_omni.worker_v2.model_states.omni_model_state import OmniModelState
from vllm_omni.worker_v2.warmup import is_warmup_request

from .ar_config import ZONOS3_PROMPT_KEY
from .modeling_zonos3_mini import Zonos3Backbone, Zonos3BackboneBatch, Zonos3PrefillSpan
from .sampler import Zonos3Sampler
from .sampling_state import Zonos3ARSamplingState


@dataclass
class _Prompt:
    embeddings: torch.Tensor
    has_audio_prefix: bool


class Zonos3ARModelState(OmniModelState):
    supports_prompt_embeds = False

    def __init__(
        self, vllm_config: VllmConfig, model: Zonos3Backbone, encoder_cache: EncoderCache | None, device: torch.device
    ) -> None:
        if vllm_config.model_config.enable_prompt_embeds or encoder_cache is not None:
            raise ValueError("Zonos3 conditioning must use its request metadata, not generic prompt/MM embeddings")
        if vllm_config.speculative_config is not None:
            raise ValueError("Zonos3 V1 does not support speculative scheduler tokens")
        super().__init__(vllm_config, model, encoder_cache, device)
        self.has_preprocess = True
        self._embed_dim = model.config.dim
        if self._static_inputs_embeds is None:
            self._static_inputs_embeds = torch.zeros(
                (self.max_num_tokens, self._embed_dim), dtype=self.dtype, device=device
            )
        self.gdn_states = model.allocate_gdn_states(2 * self.max_num_reqs)
        self.sampling_state = Zonos3ARSamplingState(
            2 * self.max_num_reqs, model.config.eoa_id, model.config.audio_pad_id, device
        )
        self.next_embeddings = torch.zeros((2 * self.max_num_reqs, self._embed_dim), dtype=self.dtype, device=device)
        self._slots = torch.empty(self.max_num_reqs, dtype=torch.int64, device=device)
        self._batches: dict[int, Zonos3BackboneBatch] = {}
        self._prompts: dict[str, _Prompt] = {}
        self._lengths: dict[str, int] = {}
        self._request_slots: dict[str, int] = {}
        self._decode_started: set[str] = set()
        self._startup = True

    def custom_sampler(self, sampler: Sampler) -> tuple[Zonos3Sampler, None]:
        return Zonos3Sampler(
            sampler,
            self.model.config,
            self.model.multi_embedder,
            self.model.multi_output,
            self.sampling_state,
            self.next_embeddings,
            self.max_model_len,
        ), None

    def add_request(self, req_index: int, new_req_data: NewRequestData) -> None:
        if new_req_data.num_computed_tokens:
            raise ValueError("Zonos3 cannot restore a KV-only prefix or recomputed/preempted GDN state")
        if new_req_data.req_id in self._request_slots:
            raise ValueError("Zonos3 request was reinitialized without completion cleanup")
        super().add_request(req_index, new_req_data)
        req_id = new_req_data.req_id
        self._request_slots[req_id] = req_index
        self._lengths[req_id] = len(new_req_data.prompt_token_ids)
        self._startup = self._startup and is_warmup_request(req_id)
        for state in self.gdn_states.values():
            state.reset([req_index])
        slots = index_to_device([req_index], self.device)
        self.sampling_state.reset(slots, torch.zeros(1, dtype=torch.bool, device=self.device))
        self.next_embeddings.index_fill_(0, slots, 0)

    def _drop(self, req_id: str) -> None:
        self._prompts.pop(req_id, None)
        self._lengths.pop(req_id, None)
        self._request_slots.pop(req_id, None)
        self._decode_started.discard(req_id)

    def remove_request(self, req_index_or_id: int | str) -> None:
        index = self._resolve_req_index(req_index_or_id)
        if index is not None:
            req_id = self.intermediate_buffer.buffers[index].get("req_id")
            if req_id is not None:
                self._drop(req_id)
        super().remove_request(req_index_or_id)

    def on_requests_finished(self, req_ids: set[str]) -> None:
        for req_id in req_ids:
            self._drop(req_id)
        super().on_requests_finished(req_ids)

    def on_request_preempted(self, req_id: str, req_index: int) -> None:
        del req_index
        raise RuntimeError(f"Zonos3 AR does not support preemption/state recomputation: {req_id}")

    def _compile_prompt(self, req_id: str, info: dict[str, Any]) -> _Prompt:
        length = self._lengths[req_id]
        if is_warmup_request(req_id):
            return _Prompt(torch.zeros((length, self._embed_dim), dtype=self.dtype, device=self.device), False)
        payload = info.get(ZONOS3_PROMPT_KEY)
        if not isinstance(payload, dict):
            raise ValueError("Missing Zonos3 prompt conditioning metadata")
        ids = payload.get("frame_ids")
        c = self.model.config
        if (
            not isinstance(ids, torch.Tensor)
            or ids.device.type != "cpu"
            or ids.dtype not in (torch.int32, torch.int64)
            or ids.shape != (length, 11)
        ):
            raise ValueError("Zonos3 placeholder length must exactly match CPU prompt frame_ids [length, 11]")
        if bool(((ids[:, :10] < 0) | (ids[:, :10] >= c.audio_vocab)).any()) or bool(
            ((ids[:, 10] < 0) | (ids[:, 10] >= c.text_rows)).any()
        ):
            raise ValueError("Zonos3 conditioning contains out-of-vocabulary frame IDs")
        embeddings = self.model.multi_embedder(to_device_nonblocking(ids, self.device))
        speaker = payload.get("speaker_embedding")
        waveform = payload.get("reference_waveform")
        if waveform is not None:
            encoder = getattr(self.model, "speaker_encoder", None)
            if (
                speaker is not None
                or encoder is None
                or not isinstance(waveform, torch.Tensor)
                or waveform.device.type != "cpu"
            ):
                raise ValueError("Zonos3 raw reference requires the model's ownmect_s2 encoder and CPU waveform")
            sample_rate = payload.get("reference_sample_rate")
            max_samples = int(self.model.config.speaker_reference_max_seconds * int(sample_rate or 0))
            if (
                not isinstance(sample_rate, int)
                or sample_rate <= 0
                or waveform.ndim not in (1, 2)
                or waveform.numel() == 0
                or waveform.shape[-1] > max_samples
            ):
                raise ValueError(
                    "Zonos3 raw speaker reference must be non-empty and no longer than "
                    f"{self.model.config.speaker_reference_max_seconds:g} seconds"
                )
            # _compile_prompt runs once per request; chunked prefill slices the
            # cached embeddings and never repeats this expensive extraction.
            speaker = encoder(waveform, sample_rate).squeeze(0)
        if speaker is not None:
            row = payload.get("speaker_row")
            if (
                not isinstance(row, int)
                or not 0 <= row < length
                or not isinstance(speaker, torch.Tensor)
                or (waveform is None and speaker.device.type != "cpu")
                or speaker.shape != (c.speaker_embedding_dim,)
            ):
                raise ValueError("Invalid Zonos3 speaker embedding row")
            embeddings[row] = self.model.speaker_projection(to_device_nonblocking(speaker, self.device).to(self.dtype))
        buckets, start = payload.get("condition_buckets"), payload.get("condition_start")
        if (
            not isinstance(buckets, (list, tuple))
            or len(buckets) != c.condition_count
            or not isinstance(start, int)
            or not 0 <= start <= length - c.condition_count
        ):
            raise ValueError("Invalid Zonos3 condition rows")
        for i, (bucket, count) in enumerate(zip(buckets, c.condition_bucket_counts, strict=True)):
            if not isinstance(bucket, int) or not 0 <= bucket <= count:
                raise ValueError("Invalid Zonos3 condition bucket ID")
            embeddings[start + i] = self.model.condition_embeddings[i].weight[bucket]
        return _Prompt(embeddings, bool(payload.get("has_audio_prefix", False)))

    def _batch(self, tokens: int, decode: bool) -> Zonos3BackboneBatch:
        batch = self._batches.get(tokens)
        if batch is None:
            batch = Zonos3BackboneBatch(
                self.gdn_states,
                self.model.allocate_workspace(tokens),
                self._slots[:tokens],
                torch.arange(tokens, device=self.device),
                decode_only=decode,
            )
        batch.decode_only = decode
        return batch

    def prepare_dummy_inputs(self, num_reqs: int, num_tokens: int) -> dict[str, Any]:
        if not 0 < num_reqs <= self.max_num_reqs or num_tokens > self.max_num_tokens:
            raise ValueError("Invalid Zonos3 native dummy batch shape")
        slots = list(range(self.max_num_reqs, self.max_num_reqs + num_reqs))
        for state in self.gdn_states.values():
            state.reset(slots)
        self._static_inputs_embeds[:num_tokens].zero_()
        batch = self._batch(num_tokens, num_reqs == num_tokens)
        if batch.decode_only:
            # Retain the actual capture workspace, even if upstream finalizes
            # capture sizes after ModelState initialization. Graphs keep raw
            # pointers to it; eager prefill shapes are deliberately not cached.
            self._batches[num_tokens] = batch
            self._slots[:num_tokens].copy_(index_to_device(slots, self.device))
            batch.decode_slots = self._slots[:num_tokens]
            batch.prefill_spans = ()
        else:
            lengths = [num_tokens // num_reqs] * num_reqs
            for i in range(num_tokens % num_reqs):
                lengths[-1 - i] += 1
            spans, start = [], 0
            for slot, length in zip(slots, lengths, strict=True):
                spans.append(Zonos3PrefillSpan(start, length, slot, 0))
                start += length
            batch.prefill_spans = tuple(spans)
            batch.decode_slots = torch.empty(0, dtype=torch.int64, device=self.device)
            batch.decode_rows = batch.decode_slots
        return {"inputs_embeds": self._static_inputs_embeds[:num_tokens], "zonos3_batch": batch}

    def prepare_inputs(self, input_batch: InputBatch, req_states: RequestState) -> dict[str, Any]:
        result = super().prepare_inputs(input_batch, req_states)
        if self._startup and all(req_id not in self._request_slots for req_id in input_batch.req_ids):
            result.update(self.prepare_dummy_inputs(input_batch.num_reqs, input_batch.num_tokens_after_padding))
        return result

    def run_preprocess(
        self,
        input_batch: InputBatch,
        model_inputs: dict[str, Any],
        req_states: RequestState | None = None,
        mtp_batch_descriptor_dispatcher: Callable[[int], Any] | None = None,
    ) -> None:
        tokens = input_batch.num_tokens_after_padding
        embeddings = self._static_inputs_embeds[:tokens]
        embeddings.zero_()
        one_token = bool(np.all(input_batch.num_scheduled_tokens == 1))
        decode_only = one_token and input_batch.decode_graph_eligible
        batch = self._batch(tokens, decode_only)
        spans, decode_slots, decode_rows = [], [], []
        for row, req_id in enumerate(input_batch.req_ids):
            slot = int(input_batch.idx_mapping_np[row])
            if self._request_slots.get(req_id) != slot:
                raise RuntimeError("Zonos3 request/slot identity mismatch")
            start = int(input_batch.query_start_loc_np[row])
            length = int(input_batch.num_scheduled_tokens[row])
            if input_batch.is_prefilling_np[row]:
                if req_id in self._decode_started:
                    raise RuntimeError("Zonos3 cannot resume prefill after AR decode")
                prompt = self._prompts.get(req_id)
                if prompt is None:
                    prompt = self._compile_prompt(req_id, self.intermediate_buffer.buffers[slot])
                    self._prompts[req_id] = prompt
                    self.sampling_state.has_audio_prefix[slot] = prompt.has_audio_prefix
                offset = int(input_batch.num_computed_prefill_tokens_np[row])
                if (
                    offset + length > prompt.embeddings.shape[0]
                    or int(input_batch.prefill_len_np[row]) != prompt.embeddings.shape[0]
                ):
                    raise ValueError("Zonos3 scheduled prefill differs from its exact logical prompt length")
                embeddings[start : start + length].copy_(prompt.embeddings[offset : offset + length])
                if decode_only:
                    # One-token prefills may use a native decode graph. Update
                    # the CPU prefill frontier without reading recurrent state.
                    for state in self.gdn_states.values():
                        state.prefill_tokens[slot] = offset + 1
                else:
                    spans.append(Zonos3PrefillSpan(start, length, slot, offset))
            else:
                if length != 1 or req_id not in self._prompts:
                    raise ValueError("Zonos3 AR decode needs one logical token after conditioning prefill")
                self._decode_started.add(req_id)
                decode_slots.append(slot)
                decode_rows.append(start)
        if decode_rows:
            rows = index_to_device(decode_rows, self.device)
            slots = index_to_device(decode_slots, self.device)
            embeddings.index_copy_(0, rows, self.next_embeddings.index_select(0, slots))
        if decode_only:
            count = input_batch.num_reqs
            if tokens > self.max_num_reqs:
                raise ValueError("Zonos3 decode graph padding exceeds its configured request capacity")
            slots = [int(slot) for slot in input_batch.idx_mapping_np] + list(
                range(self.max_num_reqs, self.max_num_reqs + tokens - count)
            )
            self._slots[:tokens].copy_(index_to_device(slots, self.device))
            batch.decode_slots, batch.prefill_spans = self._slots[:tokens], ()
        else:
            batch.prefill_spans = tuple(spans)
            batch.decode_slots = index_to_device(decode_slots, self.device)
            batch.decode_rows = index_to_device(decode_rows, self.device)
        model_inputs["inputs_embeds"] = embeddings
        model_inputs["zonos3_batch"] = batch

    def prepare_attn(
        self,
        input_batch: InputBatch,
        cudagraph_mode: CUDAGraphMode,
        block_tables: tuple[torch.Tensor, ...],
        slot_mappings: torch.Tensor,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        for_capture: bool = False,
        ubatch_idx: int = 0,
        model_specific_attn_metadata: ModelSpecificAttnMetadata | None = None,
    ) -> dict[str, Any]:
        warmup_only = bool(input_batch.req_ids) and all(is_warmup_request(req_id) for req_id in input_batch.req_ids)
        if (
            not for_capture
            and not warmup_only
            and not input_batch.has_prefill
            and cudagraph_mode is not CUDAGraphMode.FULL
        ):
            raise RuntimeError("Zonos3 steady-state AR decode requires one native FULL Backbone graph replay")
        return super().prepare_attn(
            input_batch,
            cudagraph_mode,
            block_tables,
            slot_mappings,
            attn_groups,
            kv_cache_config,
            for_capture=for_capture,
            ubatch_idx=ubatch_idx,
            model_specific_attn_metadata=model_specific_attn_metadata,
        )


class Zonos3CodecModelState(OmniModelState):
    def on_request_preempted(self, req_id: str, req_index: int) -> None:
        del req_index
        raise RuntimeError(f"Zonos3 Codec does not support preemption/state recomputation: {req_id}")
