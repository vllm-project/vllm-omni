# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import torch
from vllm import envs
from vllm.config import VllmConfig
from vllm.platforms import current_platform
from vllm.triton_utils import triton
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p
from vllm.v1.worker.gpu.spec_decode.speculator import BaseSpeculator

from vllm_omni.model_executor.stage_input_processors.higgs_audio_v3 import parse_async_chunk_config
from vllm_omni.platforms import current_omni_platform
from vllm_omni.utils.seeded_exponential import (
    _generator_exponential_kernel,
    fill_exponential_steps,
    torch_exponential_policy,
)

from .higgs_audio_v3_talker import _advance_delay_state, _mask_delayed_logits
from .int8_drafter import HiggsInt8Drafter


@dataclass
class DraftBatch:
    req_ids: list[str]
    batch_rows: torch.Tensor
    bonus_rows: torch.Tensor
    codes: torch.Tensor
    delay: torch.Tensor
    remaining: torch.Tensor
    valid: torch.Tensor


class HiggsAudioSpeculator(BaseSpeculator):
    """Draft and verify full eight-codebook delayed audio rows.

    MRv2 schedules scalar control tokens as drafts; this object owns the audio
    rows behind them. The Higgs model state embeds the candidates and verifies
    them with the noise the target sampler would use at each position.
    """

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        self.vllm_config = vllm_config
        self.first_window_rows = 0
        parallel = vllm_config.parallel_config
        backend = vllm_config.attention_config.backend
        capability = current_platform.get_device_capability(device.index)
        if (
            not current_platform.is_cuda()
            or capability is None
            or capability.major < 8
            or backend is None
            or backend.name != "FLASH_ATTN"
            or vllm_config.model_config.dtype != torch.bfloat16
            or vllm_config.model_config.quantization is not None
            or not envs.VLLM_BATCH_INVARIANT
            or parallel.tensor_parallel_size != 1
            or parallel.data_parallel_size != 1
            or parallel.pipeline_parallel_size != 1
            or vllm_config.cache_config.enable_prefix_caching
            or not vllm_config.scheduler_config.async_scheduling
        ):
            raise ValueError(
                "Higgs INT8 drafting requires BF16 weights, FlashAttention on CUDA SM80+, async scheduling, "
                "VLLM_BATCH_INVARIANT=1, a single device, and disabled prefix caching"
            )
        if vllm_config.num_speculative_tokens != 1:
            raise ValueError("Higgs INT8 drafting requires num_speculative_tokens=1")
        self.max_model_len = vllm_config.model_config.max_model_len
        self.pending: dict[str, tuple[int, torch.Tensor]] = {}
        self.single_step: set[str] = set()
        self.device_rng = {}
        self.batch: DraftBatch | None = None
        self._bonus_graphs = {}

    def bind_model_state(self, state) -> None:
        if not hasattr(state, "audio_speculator"):
            raise ValueError("Higgs audio speculation requires the Higgs Audio v3 Talker")
        if state._static_inputs_embeds is None or not state.direct_payload:
            raise ValueError("Higgs audio speculation requires static input embeddings and direct audio payloads")
        model_config = self.vllm_config.model_config
        if model_config.async_chunk:
            chunk, _, holdback, initial = parse_async_chunk_config(model_config.stage_connector_config)
            self.first_window_rows = min(initial or chunk, chunk) + holdback + state.model.num_codebooks - 1
        state.audio_speculator = self
        self.drafter = HiggsInt8Drafter(state.model, self.vllm_config)

    def bind_kv_cache(self, config, tables) -> None:
        if len(config.kv_cache_groups) != 1 or tables.cp_size != 1:
            raise ValueError("Higgs INT8 drafting requires one unsharded KV cache group")
        self.drafter.block_tables = tables
        self.drafter.layer_names = config.kv_cache_groups[0].layer_names
        self.drafter.graphs.clear()

    def init_cudagraph_manager(self, cudagraph_mode) -> None:
        pass

    def capture(self) -> None:
        pass

    def propose(self, *args: Any, dummy_run: bool = False, **kwargs: Any) -> torch.Tensor | None:
        if dummy_run:
            return None
        raise RuntimeError("Higgs audio drafts are generated during model input preparation")

    def prepare(self, state, input_batch, embeds):
        if input_batch.num_draft_tokens == 0:
            self.batch = None
            return
        model = state.model
        rows, req_ids, anchors = [], [], []
        for row in np.flatnonzero(
            (input_batch.num_draft_tokens_per_req == 1) & (input_batch.num_scheduled_tokens == 2)
        ):
            req_id = input_batch.req_ids[row]
            pending = self.pending.pop(req_id, None)
            if req_id in state.generators and pending is not None and pending[0] == input_batch.idx_mapping_np[row]:
                rows.append(row)
                req_ids.append(req_id)
                anchors.append(pending[1])
        if not rows:
            self.batch = None
            return
        starts = input_batch.query_start_loc_np[rows]
        indices = async_tensor_h2d(
            np.stack((rows, starts, starts + 1, input_batch.cu_num_logits_np[rows] + 1)), device=embeds.device
        )
        batch_rows, start_rows, draft_rows, bonus_rows = indices
        start_positions = input_batch.positions[start_rows]
        valid = start_positions == torch.stack(anchors)
        context = self.max_model_len
        if len(rows) == 1:
            length = int(input_batch.num_computed_tokens_np[rows[0]]) + 2
            context = next(size for size in (128, 256, 512, context) if size >= length)

        # The target forward overwrites the first scheduled KV slot of each draft request.
        feedback = embeds[start_rows]
        logits = self.drafter.forward(batch_rows, start_rows, start_positions, feedback, context)
        books = model.num_codebooks
        vocab = model.multimodal_embedding.vocab_size
        codebooks = model._get_codebook_indices(embeds.device)
        _mask_delayed_logits(
            logits, model._decode_delay_count[batch_rows], model._decode_eoc_countdown[batch_rows], codebooks, books
        )
        noise, _, _ = self.preview_noise(state, req_ids, embeds.device)
        codes = model._sample_audio_codes(
            logits.flatten(0, 1), state.sampling_metadata(req_ids), books, noise=noise[0].view(-1, vocab)
        ).reshape(len(req_ids), books)
        effective, delay, remaining, _ = _advance_delay_state(
            codes,
            model._decode_delay_count[batch_rows],
            model._decode_eoc_countdown[batch_rows],
            codebooks,
            model.num_codebooks,
        )
        embeds.index_copy_(0, draft_rows, model.multimodal_embedding(effective).to(embeds.dtype))
        self.batch = DraftBatch(req_ids, batch_rows, bonus_rows, codes, delay, remaining, valid)

    def preview_noise(self, state, req_ids, device):
        generators = [state.generators.get(r) for r in req_ids]
        if not all(generators):
            for req, gen in zip(req_ids, generators, strict=True):
                entry = self.device_rng.pop(req, None)
                if entry is not None and entry[0] is gen:
                    gen.set_offset(int(entry[2].item()))
            return None, None, 0
        seeds, offsets = [], []
        for r, gen in zip(req_ids, generators, strict=True):
            entry = self.device_rng.get(r)
            if entry is None or entry[0] is not gen:
                seed = gen.initial_seed()
                seed -= (1 << 64) if seed >= 1 << 63 else 0
                seed_tensor, offset_tensor = async_tensor_h2d(
                    (seed, gen.get_offset()), dtype=torch.int64, device=device
                )
                entry = self.device_rng[r] = gen, seed_tensor, offset_tensor
            seeds.append(entry[1])
            offsets.append(entry[2])
        seed_tensor = torch.stack(seeds)
        offset_tensor = torch.stack(offsets)
        width = state.model.num_codebooks * state.model.multimodal_embedding.vocab_size
        threads, increment = torch_exponential_policy(width, device)
        noise = torch.empty((2, len(req_ids), width), dtype=torch.float32, device=device)
        block = 1024
        _generator_exponential_kernel[(len(req_ids), triton.cdiv(width, block), 2)](
            noise,
            seed_tensor,
            seed_tensor,
            offset_tensor,
            width,
            threads,
            HAS_ROWS=False,
            BLOCK=block,
            STEP_INCREMENT=increment,
        )
        return noise, offset_tensor, increment

    def update_offsets(self, req_ids, offsets):
        for req_id, offset in zip(req_ids, offsets.unbind(), strict=True):
            gen, seed, _ = self.device_rng[req_id]
            self.device_rng[req_id] = gen, seed, offset

    def verify(self, state, hidden, first_tokens, counts, payload, preview=None):
        batch = self.batch
        self.batch = None
        assert batch is not None
        model = state.model
        batch_rows = batch.batch_rows
        req_ids = batch.req_ids
        books = model.num_codebooks
        vocab = model.multimodal_embedding.vocab_size
        codebooks = model._get_codebook_indices(hidden.device)
        generators = tuple(state.generators[r] for r in req_ids)
        metadata = state.sampling_metadata(req_ids)
        # Mixed seeded/unseeded batches use the ordinary sampler's Torch RNG.
        # Its offset already points past the first row.
        if preview is None:
            noise = torch.empty((1, len(req_ids), books * vocab), device=hidden.device, dtype=torch.float32)
            increment = fill_exponential_steps(noise, generators)
        else:
            noise = preview[batch_rows][None]
        _, codes, delay, remaining, done = self._replay_bonus(
            model, hidden[batch.bonus_rows], batch.delay, batch.remaining, metadata, noise, codebooks
        )
        first = model._decode_last_codes[batch_rows]
        # The first row is already committed. Accept the bonus only if its
        # input matched that row and the target has not started the EOC ramp.
        first_remaining = model._decode_eoc_countdown[batch_rows]
        first_active = model._decode_has_codes[batch_rows]
        commit_bonus = (first == batch.codes).all(dim=1) & (first_remaining < 0) & first_active
        commit_bonus &= batch.valid
        model._decode_last_codes[batch_rows] = torch.where(commit_bonus[:, None], codes, first)
        model._decode_delay_count[batch_rows] = torch.where(commit_bonus, delay, model._decode_delay_count[batch_rows])
        model._decode_eoc_countdown[batch_rows] = torch.where(commit_bonus, remaining, first_remaining)
        model._decode_has_codes[batch_rows] = torch.where(commit_bonus, ~done, first_active)
        model._decode_generation_done[batch_rows] |= commit_bonus & done
        if preview is None:
            for generator, accepted in zip(generators, commit_bonus.tolist(), strict=True):
                if accepted:
                    generator.set_offset(generator.get_offset() + increment)
        counts[batch_rows] = commit_bonus.to(counts.dtype) + 1
        tokens = first_tokens.new_full((len(counts), 2), model._audio_continuation_id)
        tokens[:, :1] = first_tokens
        snapshot = payload["_higgs_audio_snapshot"]
        owned = snapshot.new_zeros((len(counts), 2, books))
        owned[:, 0] = snapshot[:, :books]
        owned[batch_rows, 1] = codes.to(snapshot.dtype)
        payload["_higgs_spec_audio"] = owned
        return tokens

    def _replay_bonus(self, model, hidden, delay, remaining, metadata, noise, codebooks):
        books = model.num_codebooks
        vocab = model.multimodal_embedding.vocab_size
        params = (metadata.temperature, metadata.top_k, metadata.top_p)
        key = (len(hidden), metadata.all_greedy, tuple(p is None for p in params))
        entry = self._bonus_graphs.get(key)
        args = (hidden, delay, remaining, noise, *(p for p in params if p is not None))
        if entry is None:
            if not self._bonus_graphs:
                # Top-k's shared scratch grows with the row count. Reserve its
                # maximum size before any bonus graph captures its address.
                rows = self.vllm_config.scheduler_config.max_num_seqs * books
                logits = torch.zeros((rows, vocab), dtype=torch.float32, device=hidden.device)
                top_k = torch.ones(rows, dtype=torch.int32, device=hidden.device)
                apply_top_k_top_p(logits, top_k, None)
            static = tuple(x.clone() for x in args)
            h, d, r, n, *controls = static
            it = iter(controls)
            temp, k, p = (None if field is None else next(it) for field in params)
            meta = replace(metadata, temperature=temp, top_k=k, top_p=p, generators={})

            def sample():
                logits = model.modality_head.generate(h)
                _mask_delayed_logits(logits, d, r, codebooks, books)
                raw = model._sample_audio_codes(logits.flatten(0, 1), meta, books, noise=n.view(-1, vocab))
                raw = raw.view(len(h), books)
                return (raw, *_advance_delay_state(raw, d, r, codebooks, books))

            outputs = None

            def capture():
                nonlocal outputs
                outputs = sample()

            capture()
            graph = current_omni_platform.capture_graph(capture)
            assert graph is not None
            graph.replay()
            self._bonus_graphs[key] = graph, static, outputs
        else:
            graph, static, outputs = entry
            for dst, src in zip(static, args, strict=True):
                dst.copy_(src)
            graph.replay()
        return outputs

    def reserve_audio_drafts(self, state, input_batch, num_sampled, width):
        model = state.model
        for req_id in input_batch.req_ids:
            self.pending.pop(req_id, None)
        if not width or model._step_audio_mode_rows != input_batch.num_reqs:
            return None
        rows = [
            i
            for i, req_id in enumerate(input_batch.req_ids)
            if req_id in state.generators and req_id not in self.single_step
        ]
        if not rows:
            return None
        if self.first_window_rows:
            drafts = input_batch.num_draft_tokens_per_req
            # A batch can draft once any request can emit its first codec window.
            # A reserved bonus row may be rejected, so count only the first target row.
            for i in rows:
                scheduled = input_batch.num_scheduled_tokens[i] - (drafts[i] if drafts is not None else 0)
                prompt = state.prompt_len_np[input_batch.idx_mapping_np[i]]
                if input_batch.num_computed_tokens_np[i] + scheduled - prompt + 1 >= self.first_window_rows:
                    break
            else:
                return None
        idx = async_tensor_h2d(rows, dtype=torch.int64, device=num_sampled.device)
        req_ids = [input_batch.req_ids[i] for i in rows]
        first_logits = input_batch.logits_indices[input_batch.cu_num_logits[:-1].long()]
        anchor = input_batch.positions[first_logits[idx]] + num_sampled[idx]
        limits = [
            min(
                self.max_model_len,
                int(state.prompt_len_np[input_batch.idx_mapping_np[i]]) + state.requests[r].sampling_params.max_tokens,
            )
            for i, r in zip(rows, req_ids, strict=True)
        ]
        ceiling = async_tensor_h2d(limits, dtype=anchor.dtype, device=anchor.device)
        stable = (
            (num_sampled[idx] > 0)
            & model._decode_has_codes[idx]
            & (model._decode_delay_count[idx] >= model.num_codebooks - 1)
            & (model._decode_eoc_countdown[idx] < 0)
            & (anchor + 2 < ceiling)
        )
        for i, req_id, position in zip(rows, req_ids, torch.where(stable, anchor, -1).unbind(), strict=True):
            self.pending[req_id] = int(input_batch.idx_mapping_np[i]), position
        return idx, model._audio_continuation_id
