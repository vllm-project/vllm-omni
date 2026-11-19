# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from vllm import envs
from vllm.config import VllmConfig
from vllm.platforms import current_platform
from vllm.v1.outputs import DraftTokenIds
from vllm.v1.worker.gpu.spec_decode.speculator import BaseSpeculator

from vllm_omni.utils.seeded_exponential import fill_exponential_steps

from .higgs_audio_v3_talker import (
    _mask_delayed_logits,
    _scan_draft_rows,
    _verified_audio_prefix,
)


@dataclass
class DraftBatch:
    batch_rows: torch.Tensor
    lengths: torch.Tensor
    logit_rows: torch.Tensor
    codes: torch.Tensor
    delay: torch.Tensor
    remaining: torch.Tensor


class HiggsAudioSpeculator(BaseSpeculator):
    """Self speculation over 8-codebook delayed audio rows.

    MRv2 schedules scalar control tokens as drafts; this object owns the audio
    rows behind them. The Higgs model state embeds the candidates and verifies
    them with the noise the target sampler would use at each position.
    """

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        parallel = vllm_config.parallel_config
        if (
            not current_platform.is_cuda()
            or not envs.VLLM_BATCH_INVARIANT
            or parallel.tensor_parallel_size != 1
            or parallel.data_parallel_size != 1
            or vllm_config.scheduler_config.async_scheduling
            or vllm_config.cache_config.enable_prefix_caching
        ):
            raise ValueError(
                "Higgs audio speculation requires CUDA, VLLM_BATCH_INVARIANT=1, TP=DP=1, "
                "synchronous scheduling, and disabled prefix caching"
            )
        if vllm_config.num_speculative_tokens != 1:
            raise ValueError("Higgs audio self speculation supports num_speculative_tokens=1")
        self.max_model_len = vllm_config.model_config.max_model_len
        self.pending: dict[str, tuple[int, int, torch.Tensor]] = {}
        # Target rows sampled after a rejected candidate; the next candidates.
        self.guesses: dict[str, torch.Tensor] = {}
        # Requests resumed after preemption, which decode one row per step.
        self.single_step: set[str] = set()
        self.drafts: DraftTokenIds | None = None
        self.batch: DraftBatch | None = None

    def bind_model_state(self, state) -> None:
        if not hasattr(state, "audio_speculator"):
            raise ValueError("Higgs audio speculation requires the Higgs Audio v3 Talker")
        if state.model._sample_graph_enabled or state._static_inputs_embeds is None or not state.direct_payload:
            raise ValueError(
                "Higgs audio speculation requires static input embeddings, direct audio payloads, "
                "and disabled sampler graphs"
            )
        state.audio_speculator = self

    def init_cudagraph_manager(self, cudagraph_mode) -> None:
        pass

    def capture(self) -> None:
        pass

    def propose(self, *args: Any, dummy_run: bool = False, **kwargs: Any) -> torch.Tensor | None:
        # Omni's AR runner calls propose_audio for real steps.
        if dummy_run:
            return None
        raise RuntimeError("Higgs audio drafts are proposed through propose_audio")

    def prepare(self, state, input_batch, embeds):
        if input_batch.num_draft_tokens == 0:
            self.batch = None
            return
        model = state.model
        lengths_np = input_batch.num_draft_tokens_per_req
        rows_np = np.flatnonzero(lengths_np)
        width = int(lengths_np.max())
        raw = []
        for row in rows_np:
            req_id = input_batch.req_ids[row]
            pending = self.pending.pop(req_id, None)
            length = int(lengths_np[row])
            if (
                pending is None
                or pending[0] != input_batch.idx_mapping_np[row]
                or pending[1] != input_batch.num_computed_tokens_np[row]
                or pending[2].shape[0] < length
            ):
                raise RuntimeError(f"Higgs audio drafts do not match the scheduled slot/position for {req_id}")
            raw.append(F.pad(pending[2][:length], (0, 0, 0, width - length)))
        codes = torch.stack(raw)
        batch_rows = torch.as_tensor(rows_np, device=embeds.device)
        lengths = torch.as_tensor(lengths_np[rows_np], device=embeds.device)
        codebooks = model._get_codebook_indices(embeds.device)
        effective, delays, countdowns = _scan_draft_rows(
            codes,
            model._decode_delay_count[batch_rows],
            model._decode_eoc_countdown[batch_rows],
            codebooks,
            model.num_codebooks,
        )
        starts = input_batch.query_start_loc_np[rows_np]
        offsets = np.arange(width + 1)
        logit_rows = input_batch.cu_num_logits_np[rows_np, None] + np.minimum(offsets, lengths_np[rows_np, None])
        inputs = torch.cat((model._decode_last_codes[batch_rows, None], effective), dim=1)
        positions = starts[:, None] + offsets
        valid = offsets <= lengths_np[rows_np, None]
        embeds.index_copy_(
            0,
            torch.as_tensor(positions[valid], device=embeds.device),
            model.multimodal_embedding(
                inputs.flatten(0, 1)[torch.as_tensor(np.flatnonzero(valid), device=embeds.device)]
            ).to(embeds.dtype),
        )
        self.batch = DraftBatch(
            batch_rows, lengths, torch.as_tensor(logit_rows, device=embeds.device), codes, delays, countdowns
        )

    def verify(self, state, hidden, input_batch, first_tokens, counts, payload):
        batch = self.batch
        self.batch = None
        assert batch is not None
        model = state.model
        batch_rows = batch.batch_rows
        req_ids = [input_batch.req_ids[i] for i in np.flatnonzero(input_batch.num_draft_tokens_per_req)]
        width = batch.codes.shape[1]
        books = model.num_codebooks
        vocab = model.multimodal_embedding.vocab_size
        codebooks = model._get_codebook_indices(hidden.device)
        generators = tuple(state.generators[r] for r in req_ids)
        metadata = state.sampling_metadata(req_ids)
        # The ordinary sampler committed position zero and advanced its RNG.
        # Preview the remaining positions at the resulting offset.
        noise = torch.empty((width, len(req_ids), books * vocab), device=hidden.device, dtype=torch.float32)
        increment = fill_exponential_steps(noise, generators)
        logits = model.modality_head.generate(hidden[batch.logit_rows[:, 1:].flatten()])
        _mask_delayed_logits(logits, batch.delay[:, 1:].flatten(), batch.remaining[:, 1:].flatten(), codebooks, books)
        expanded = replace(
            metadata,
            temperature=metadata.temperature.repeat_interleave(width),
            top_k=metadata.top_k.repeat_interleave(width),
            top_p=metadata.top_p.repeat_interleave(width),
        )
        rest = model._sample_audio_codes(
            logits.flatten(0, 1), expanded, books, noise=noise.transpose(0, 1).reshape(-1, vocab)
        )
        snapshot = payload["_higgs_audio_snapshot"]
        # At the stable positions eligible for drafting, the mask makes raw
        # and effective codes identical, including the first EOC in codebook 0.
        first = snapshot[batch_rows, :books].long()
        target = torch.cat((first[:, None], rest.view(len(req_ids), width, books)), dim=1)
        codes, delay, remaining, done, last = _verified_audio_prefix(
            target, batch.codes, batch.lengths, batch.delay, batch.remaining, codebooks, books
        )
        rows = torch.arange(len(req_ids), device=hidden.device)
        model._decode_last_codes[batch_rows] = codes[rows, last]
        model._decode_delay_count[batch_rows] = delay[rows, last]
        model._decode_eoc_countdown[batch_rows] = remaining[rows, last]
        model._decode_generation_done[batch_rows] = done[rows, last]
        model._decode_has_codes[batch_rows] = ~done[rows, last]
        ends = last.tolist()
        for generator, extra in zip(generators, ends, strict=True):
            generator.set_offset(generator.get_offset() + extra * increment)
        # Rows after the first mismatch already used their final noise
        # offsets; only their inputs were wrong. Reuse them as candidates.
        lengths = input_batch.num_draft_tokens_per_req[np.flatnonzero(input_batch.num_draft_tokens_per_req)]
        for j, (req_id, end) in enumerate(zip(req_ids, ends, strict=True)):
            if end < lengths[j]:
                self.guesses[req_id] = target[j, end + 1 : lengths[j] + 1]
        counts[batch_rows] = last.to(counts.dtype) + 1
        tokens = first_tokens.new_full((input_batch.num_reqs, width + 1), model._audio_continuation_id)
        tokens[:, :1] = first_tokens
        owned = snapshot.new_zeros((input_batch.num_reqs, width + 1, books))
        owned[:, 0] = snapshot[:, :books]
        owned[batch_rows] = codes.to(snapshot.dtype)
        payload["_higgs_spec_audio"] = owned
        return tokens

    def propose_audio(self, state, input_batch, hidden_states, num_sampled, width):
        model = state.model
        for req_id in input_batch.req_ids:
            self.pending.pop(req_id, None)
        self.drafts = None
        guesses, self.guesses = self.guesses, {}
        if not width or model._step_audio_mode_rows != input_batch.num_reqs:
            return None
        num_reqs = input_batch.num_reqs
        committed, stable = torch.stack(
            (
                num_sampled,
                (
                    model._decode_has_codes[:num_reqs]
                    & (model._decode_delay_count[:num_reqs] >= model.num_codebooks - 1)
                    & (model._decode_eoc_countdown[:num_reqs] < 0)
                ).to(num_sampled.dtype),
            )
        ).tolist()
        # The last committed output is the next round's input anchor.
        logits_per_request = np.diff(input_batch.cu_num_logits_np)
        anchor = input_batch.num_computed_tokens_np + input_batch.num_scheduled_tokens - logits_per_request + committed
        rows, lengths = [], []
        for i, req_id in enumerate(input_batch.req_ids):
            if not (committed[i] and stable[i] and req_id in state.generators) or req_id in self.single_step:
                continue
            params = state.requests[req_id].sampling_params
            limit = min(self.max_model_len, int(state.prompt_len_np[input_batch.idx_mapping_np[i]]) + params.max_tokens)
            length = min(width, limit - int(anchor[i]) - 2)
            if length > 0:
                rows.append(i)
                lengths.append(length)
        if not rows:
            return None
        req_ids = [input_batch.req_ids[i] for i in rows]
        idx = torch.as_tensor(rows, device=hidden_states.device)
        # Without a guess, the committed row only reserves the verify
        # position whose output becomes the next candidate.
        committed_rows = model._decode_last_codes[idx]
        proposed = []
        for j, req_id in enumerate(req_ids):
            guess = guesses.get(req_id, committed_rows[j : j + 1])
            proposed.append(guess[torch.arange(lengths[j], device=guess.device).clamp(max=len(guess) - 1)])
        return self._store(input_batch, rows, lengths, anchor, proposed, idx, model._audio_continuation_id)

    def _store(self, input_batch, rows, lengths, anchor, proposed, idx, control):
        req_ids = [input_batch.req_ids[i] for i in rows]
        for j, i in enumerate(rows):
            self.pending[req_ids[j]] = (int(input_batch.idx_mapping_np[i]), int(anchor[i]), proposed[j])
        self.drafts = DraftTokenIds(req_ids, [[control] * length for length in lengths])
        return idx, control
