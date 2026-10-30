# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Ming3 Stage-0 request state on Model Runner V2.

Each AR step conditions the flow head on the backbone's last hidden row per
request. Two device tensors per request slot carry the feedback between steps:
the latent history and the next input embedding. The stop decision (stop head,
minimum and maximum decode steps) is made on the device as well, so a step
needs no host synchronization: the sampler returns the continue or stop token
computed here, and the latent rows reach the host through the runner's
asynchronous output copy.

In the fused pipeline Stage 0 also decodes its latents with the
AudioVAE decoder each step and returns PCM in growing chunks
(``prepare_streaming_audio_output``), so no AudioVAE stage follows.

With async scheduling a request can run one step after it stopped. Its slot is
inactive by then, so that step emits no latent and changes no state.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from vllm.v1.worker.gpu.input_batch import get_num_sampled_and_rejected
from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState
from vllm.v1.worker.gpu.sample.bad_words import BadWordsState
from vllm.v1.worker.gpu.sample.logit_bias import LogitBiasState
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu.sample.penalties import PenaltiesState
from vllm.v1.worker.gpu.sample.sampler import Sampler

from vllm_omni.model_executor.models.ming_tts.config_ming_tts import (
    KEY_CFG,
    KEY_MAX_DECODE_STEPS,
    KEY_MIN_DECODE_STEPS,
    KEY_REQUEST_ID,
    KEY_SIGMA,
    KEY_TEMPERATURE,
)
from vllm_omni.model_executor.models.ming_tts.patch_emission import (
    _resolve_optional_runtime_int,
    _resolve_runtime_float,
    _resolve_runtime_int,
)
from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.model_executor.output_snapshot import PackedOutputSnapshot, RequestOutputSnapshot
from vllm_omni.utils.device_copy import index_to_device
from vllm_omni.worker_v2.model_states.omni_model_state import OmniModelState

from .config_ming3_tts import resolve_latent_chunk_sizes

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from vllm.config import VllmConfig
    from vllm.v1.core.sched.output import NewRequestData
    from vllm.v1.worker.gpu.input_batch import InputBatch
    from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
    from vllm.v1.worker.gpu.states import RequestState

    from .ming3_tts import Ming3TTSForConditionalGeneration


class _LatentRowsSnapshot(PackedOutputSnapshot):
    """Owned per-request latent rows and emit flags, copied to the host once.

    Both tensors are allocated by the step that produced them, so later steps
    cannot overwrite them before the runner's copy completes.
    """

    def __init__(
        self, latent: torch.Tensor, emitted: torch.Tensor, *, latent_key: str, flag_key: str, partitioned: bool
    ) -> None:
        dict.__init__(self, {latent_key: latent, flag_key: emitted})
        self._latent = latent
        self._emitted = emitted
        self._keys = (latent_key, flag_key)
        self._partitioned = partitioned
        self.producer_event = None

    def copy_to_cpu(self, copy_tensor):
        latent = copy_tensor(self._latent)
        emitted = copy_tensor(self._emitted)
        latent_key, flag_key = self._keys
        if self._partitioned:
            return RequestOutputSnapshot(
                [
                    {latent_key: latent[row : row + 1], flag_key: emitted[row : row + 1]}
                    for row in range(latent.shape[0])
                ]
            )
        return {latent_key: list(latent.split(1)), flag_key: list(emitted.split(1))}


class _AudioChunks:
    """A request's PCM, released in growing chunks like the two-stage stream.

    The first audio leaves at once; later chunks hold ``chunk_latents`` latents
    of audio each, so a request emits a few outputs instead of one per step.
    """

    def __init__(self, samples_per_latent: int, chunk_latents: tuple[int, ...]) -> None:
        self.samples_per_latent = samples_per_latent
        self.chunk_latents = chunk_latents
        self.chunks = 0
        self.pending: list[torch.Tensor] = []
        self.pending_samples = 0
        self.lock = threading.Lock()

    def push(self, pcm: torch.Tensor, ended: bool) -> torch.Tensor | None:
        with self.lock:
            if pcm.numel():
                self.pending.append(pcm)
                self.pending_samples += pcm.numel()
            # The first release is immediate; the ramp starts after it.
            ramp = self.chunk_latents[min(max(self.chunks - 1, 0), len(self.chunk_latents) - 1)]
            target = ramp * self.samples_per_latent
            if not self.pending_samples or not (ended or self.chunks == 0 or self.pending_samples >= target):
                return None
            out = self.pending[0] if len(self.pending) == 1 else torch.cat(self.pending)
            self.pending, self.pending_samples = [], 0
            self.chunks += 1
            return out


@dataclass(eq=False)
class _StepAudioOutput:
    """One step's PCM per request, from the single-stage AudioVAE decode.

    ``audio`` holds each decoded row's overlap-add span; the row's PCM is
    ``audio[row, start:end]``, bounds that the device decided this step, and
    ``last`` marks a request's final audio.
    """

    audio: torch.Tensor | None
    start: torch.Tensor | None
    end: torch.Tensor | None
    last: torch.Tensor | None
    # (batch index, request chunker) per decoded row.
    rows: list[tuple[int, _AudioChunks]]
    num_reqs: int
    sample_rate: torch.Tensor
    event: torch.cuda.Event | None = None

    def to_cpu(
        self, copy_stream: torch.cuda.Stream, copy_tensor: Callable[[torch.Tensor], torch.Tensor]
    ) -> _StepAudioOutput:
        if self.event is not None:
            copy_stream.wait_event(self.event)
        audio, start, end, last = (
            copy_tensor(value) if value is not None else None for value in (self.audio, self.start, self.end, self.last)
        )
        return _StepAudioOutput(audio, start, end, last, self.rows, self.num_reqs, self.sample_rate)

    def get_output(self) -> list[dict[str, torch.Tensor] | None]:
        output: list[dict[str, torch.Tensor] | None] = [None] * self.num_reqs
        if self.audio is None:
            return output
        assert self.start is not None and self.end is not None and self.last is not None
        bounds = zip(self.rows, self.start.tolist(), self.end.tolist(), self.last.tolist(), strict=True)
        for row, ((request, chunks), start, end, last) in enumerate(bounds):
            pcm = self.audio[row, start:end]
            chunk = chunks.push(pcm, bool(last))
            if chunk is not None and chunk.numel():
                output[request] = {"model_outputs": chunk, "sr": self.sample_rate}
        return output


class Ming3ModelState(OmniModelState, MambaHybridModelState):
    """Slot-indexed flow-head feedback and device-side stopping for Ming3 Stage 0.

    MRO: Omni -> MambaHybrid -> Default, so attention metadata carries the
    prefill mask the KDA layers need.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        model: Ming3TTSForConditionalGeneration,
        encoder_cache: EncoderCache | None,
        device: torch.device,
    ) -> None:
        super().__init__(vllm_config, model, encoder_cache, device)
        llm = model.model
        cfg = model.ming_config
        self.llm = llm
        slots = self.scheduler_config.max_num_seqs
        history_len = cfg.history_patch_size * cfg.patch_size
        self.history = torch.zeros((slots, history_len, cfg.latent_dim), dtype=llm.fm_dtype, device=device)
        self.next_embed = torch.zeros((slots, cfg.llm_hidden_size), dtype=self.dtype, device=device)
        self.active = torch.zeros(slots, dtype=torch.bool, device=device)
        self.min_steps = torch.ones(slots, dtype=torch.long, device=device)
        self.max_steps = torch.ones(slots, dtype=torch.long, device=device)
        self._sde: list[tuple[float, float, float]] = [
            (float(cfg.cfg), float(cfg.sigma), float(cfg.temperature))
        ] * slots
        # Requests with ``tts_local_seed`` draw flow-head noise from their own
        # generator, created at the first flow step as on the V1 runner (apart
        # from the one the prompt-latent posterior used at prefill); others
        # draw from the global RNG.
        self._seeds: list[int | None] = [None] * slots
        self._flow_generators: list[torch.Generator | None] = [None] * slots
        # Slot -> (minimum, maximum) decode steps, written before the next step.
        self._pending: dict[int, tuple[int, int]] = {}
        self._partitioned = bool(getattr(vllm_config.model_config, "async_chunk", False))
        self._tokens: torch.Tensor | None = None
        self._request_errors: tuple[torch.Tensor, str] | None = None
        self._continue_token = int(cfg.audio_dummy_token_id)
        self._stop_token = int(cfg.text_eos_token_id)
        self._threshold = float(cfg.stop_head_threshold)
        self._patch_size = int(cfg.patch_size)
        # Single-stage serving: Stage 0 decodes its latents to PCM itself. A
        # stream's last latent leaves the frame-holdback upsampler only on a
        # flush, so a request that halts emits one more step (draining) in which
        # the flush runs, and stops there; async scheduling runs that step anyway.
        self._step_vae = getattr(model, "step_vae", None)
        self.draining = torch.zeros(slots, dtype=torch.bool, device=device)
        self._vae_positions = torch.zeros(slots, dtype=torch.long, device=device)
        self._sample_rate = torch.tensor(int(cfg.sample_rate), dtype=torch.int32)
        self._step_audio: _StepAudioOutput | None = None
        self._audio_chunks: dict[str, _AudioChunks] = {}
        if self._step_vae is not None:
            connector = getattr(vllm_config.model_config, "stage_connector_config", None)
            connector = (connector.get("extra", connector) if isinstance(connector, dict) else None) or {}
            options = {**(vllm_config.additional_config or {}), **connector}
            steady, chunk = resolve_latent_chunk_sizes(options)
            ramp = []
            while chunk < steady:
                ramp.append(chunk)
                chunk *= 2
            self._chunk_latents = (*ramp, steady)
            self._samples_per_latent = self._step_vae.scale * self._step_vae.hop

    # ------------------------------------------------------------------
    # Requests
    # ------------------------------------------------------------------

    def _request_info(self, req_index: int) -> dict[str, Any]:
        info = dict(self.intermediate_buffer.buffers[req_index])
        extra = info.pop("additional_information", None)
        if isinstance(extra, dict):
            info = {**extra, **info}
        # Normalize the public Ming3 controls before admission caches them,
        # using the same mapping as V1 preprocess.
        self.model._copy_request_state(info, info)
        return info

    def add_request(self, req_index: int, new_req_data: NewRequestData) -> None:
        super().add_request(req_index, new_req_data)
        req_id = str(new_req_data.req_id)
        prompt = list(new_req_data.prompt_token_ids or [])
        if len(new_req_data.prefill_token_ids or prompt) > len(prompt):
            # Generated positions are embedded from flow latents, not token ids,
            # so a recomputed prefix would not match what the request saw.
            raise RuntimeError(
                f"Ming3 cannot resume preempted request {req_id}; size the KV cache for max_num_seqs requests"
            )
        info = self._request_info(req_index)
        cfg = self.llm.ming_config
        self._sde[req_index] = (
            _resolve_runtime_float(info, KEY_CFG, cfg.cfg),
            _resolve_runtime_float(info, KEY_SIGMA, cfg.sigma),
            _resolve_runtime_float(info, KEY_TEMPERATURE, cfg.temperature),
        )
        maximum = _resolve_runtime_int(info, KEY_MAX_DECODE_STEPS, cfg.max_decode_steps)
        minimum = max(1, _resolve_optional_runtime_int(info, KEY_MIN_DECODE_STEPS, cfg.stop_head_min_steps))
        if maximum < minimum:
            raise ValueError(
                f"Ming3 request {req_id}: maximum decode steps {maximum} is smaller than minimum {minimum}"
            )
        seed = info.get("tts_local_seed")
        self._seeds[req_index] = None if seed is None else int(seed)
        self._flow_generators[req_index] = None
        self._pending[req_index] = (minimum, maximum)

    def remove_request(self, req_index_or_id: int | str) -> None:
        index = self._resolve_req_index(req_index_or_id)
        if index is not None:
            self._pending.pop(index, None)
            self._seeds[index] = None
            self._flow_generators[index] = None
        super().remove_request(req_index_or_id)

    def on_requests_finished(self, req_ids: set[str]) -> None:
        super().on_requests_finished(req_ids)
        # The runner notifies the wrapper before this model-state callback.
        for req_id in req_ids:
            self._audio_chunks.pop(req_id, None)

    def _flush_pending(self) -> None:
        """Write the decode windows of newly added requests and activate their slots."""
        if not self._pending:
            return
        values = [value for slot, (minimum, maximum) in self._pending.items() for value in (slot, minimum, maximum)]
        self._pending.clear()
        packed = index_to_device(values, self.device).view(-1, 3)
        slots = packed[:, 0]
        self.min_steps.index_copy_(0, slots, packed[:, 1])
        self.max_steps.index_copy_(0, slots, packed[:, 2])
        self.active.index_fill_(0, slots, True)
        self.draining.index_fill_(0, slots, False)

    # ------------------------------------------------------------------
    # Step inputs
    # ------------------------------------------------------------------

    def run_preprocess(
        self,
        input_batch: InputBatch,
        model_inputs: dict[str, Any],
        req_states: RequestState | None = None,
        mtp_batch_descriptor_dispatcher: Callable[[int], Any] | None = None,
    ) -> None:
        self._flush_pending()
        num_reqs = input_batch.num_reqs
        if num_reqs == 0:
            return
        input_ids = model_inputs.get("input_ids")
        if input_ids is None:
            input_ids = input_batch.input_ids
        embeds = model_inputs.get("inputs_embeds")
        if embeds is None:
            embeds = self.model.embed_input_ids(input_batch.input_ids[: input_batch.num_tokens])
            model_inputs["inputs_embeds"] = embeds
        slots = np.asarray(input_batch.idx_mapping_np[:num_reqs])
        scheduled = np.asarray(input_batch.num_scheduled_tokens[:num_reqs])
        computed = np.asarray(input_batch.num_computed_tokens_np[:num_reqs])
        starts = np.asarray(input_batch.query_start_loc_np[:num_reqs])
        prompt_len = np.asarray(input_batch.prefill_len_np[:num_reqs])
        prefill = computed < prompt_len

        # Decode rows read the embedding the previous step produced for their slot.
        if not prefill.any() and input_batch.num_tokens == num_reqs:
            embeds[:num_reqs].copy_(self.next_embed.index_select(0, input_batch.idx_mapping[:num_reqs]))
        elif not prefill.all():
            rows = np.flatnonzero(~prefill)
            positions = index_to_device(starts[rows].tolist(), embeds.device)
            embeds.index_copy_(
                0,
                positions,
                self.next_embed.index_select(0, index_to_device(slots[rows].tolist(), embeds.device)),
            )

        infos: dict[int, dict[str, Any]] = {}
        for row in np.flatnonzero(prefill).tolist():
            slot, start, count, done = int(slots[row]), int(starts[row]), int(scheduled[row]), int(computed[row])
            req_id = str(input_batch.req_ids[row])
            if req_id.startswith("_warmup_"):
                ids = input_ids[start : start + count]
                embeds[start : start + count].copy_(self.model.embed_input_ids(ids))
                continue
            info = self._request_info(slot)
            info["request_id"] = req_id
            info[KEY_REQUEST_ID] = req_id
            info["_omni_prompt_len"] = int(prompt_len[row])
            info["_omni_num_computed_tokens"] = done
            info["_omni_is_prefill"] = True
            infos[row] = info
        if not infos:
            return

        # Batch reference encoding; each request retains its full projection so
        # scheduler chunk boundaries cannot change the projection's GEMM shape.
        self.model.prefetch_prompt_encodings([info for row, info in infos.items() if computed[row] == 0])
        try:
            for row, info in infos.items():
                slot, start, count = int(slots[row]), int(starts[row]), int(scheduled[row])
                _, new_embeds, _ = self.model.preprocess(
                    input_ids[start : start + count], embeds[start : start + count], **info
                )
                embeds[start : start + count].copy_(new_embeds[:count])
                state = self.llm._decode_state.pop(info["request_id"], None)
                if state is not None:
                    self.history[slot].copy_(state[0].reshape(self.history.shape[1:]))
        finally:
            self.model.end_prompt_batch()

    def run_postprocess(self, hidden_states: torch.Tensor, input_batch: InputBatch) -> None:
        # The flow head and request state updates run in postprocess_model_output.
        return

    # ------------------------------------------------------------------
    # Flow head, stopping and outputs
    # ------------------------------------------------------------------

    def _generator(self, slot: int, device: torch.device) -> torch.Generator | None:
        seed = self._seeds[slot]
        if seed is None:
            return None
        generator = self._flow_generators[slot]
        if generator is None:
            generator = self._flow_generators[slot] = torch.Generator(device=device)
            generator.manual_seed(seed)
        return generator

    def postprocess_model_output(
        self, model_output: Any, input_batch: InputBatch, req_states: RequestState
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        hidden = model_output.text_hidden_states if isinstance(model_output, OmniOutput) else model_output
        num_reqs = input_batch.num_reqs
        self._tokens = None
        self._request_errors = None
        self.llm._mrv2_next_token_ids = None
        if num_reqs == 0 or not isinstance(hidden, torch.Tensor):
            return hidden, {}
        device = hidden.device
        llm = self.llm
        slots_np = np.asarray(input_batch.idx_mapping_np[:num_reqs])
        done = np.asarray(input_batch.num_computed_tokens_np[:num_reqs]) + np.asarray(
            input_batch.num_scheduled_tokens[:num_reqs]
        )
        # A request samples once its prompt is complete: decode rows and final prefill chunks.
        rows = np.flatnonzero(done >= np.asarray(input_batch.prefill_len_np[:num_reqs]))

        tokens = torch.full((num_reqs,), self._stop_token, dtype=torch.long, device=device)
        latent_out = torch.zeros(
            (num_reqs, self._patch_size, llm.ming_config.latent_dim), dtype=llm.fm_dtype, device=device
        )
        emitted_out = torch.zeros(num_reqs, dtype=torch.bool, device=device)
        flush_out = torch.zeros(num_reqs, dtype=torch.bool, device=device)
        failed_out = torch.zeros(num_reqs, dtype=torch.bool, device=device)
        groups: dict[tuple[float, float, float], list[int]] = {}
        for row in rows.tolist():
            groups.setdefault(self._sde[int(slots_np[row])], []).append(row)
        sample_index = input_batch.logits_indices[:num_reqs]
        seq_lens = input_batch.seq_lens[:num_reqs]
        for (cfg_scale, sigma, temperature), group in groups.items():
            whole = len(group) == num_reqs
            group_rows = None if whole else index_to_device(group, device)
            # index_copy_ takes int64 indices; the runner's slot mapping is int32.
            slots = (
                input_batch.idx_mapping[:num_reqs].long()
                if whole
                else index_to_device(slots_np[group].tolist(), device)
            )
            cond = hidden.index_select(0, sample_index if whole else sample_index.index_select(0, group_rows))
            history = self.history.index_select(0, slots)
            latent, next_embeds, stop_probs, _ = llm._sample_flow_rows(
                cond,
                history,
                [self._generator(int(slots_np[row]), device) for row in group],
                cfg_scale=cfg_scale,
                sigma=sigma,
                temperature=temperature,
                # Rows are checked one by one below.
                check_finite=False,
            )
            latent = latent.to(self.history.dtype)
            next_embeds = next_embeds.reshape(len(group), -1).to(self.next_embed.dtype)
            stop_probs = stop_probs.reshape(-1).float()
            # One-based count of the latent this step emits for each request.
            count = (
                (seq_lens if whole else seq_lens.index_select(0, group_rows))
                - req_states.prompt_len.gpu.index_select(0, slots)
                + 1
            )
            finite = (
                torch.isfinite(cond).flatten(1).all(1)
                & torch.isfinite(latent).flatten(1).all(1)
                & torch.isfinite(next_embeds).all(1)
                & torch.isfinite(stop_probs)
            )
            # Report failed live rows through the asynchronous output copy.
            # Already stopped/draining rows no longer consume the flow result.
            active = self.active.index_select(0, slots)
            failed = active & ~finite
            failed_out = failed if whole else failed_out.index_copy(0, group_rows, failed)
            live = active & finite
            halt = (
                (count >= self.max_steps.index_select(0, slots))
                | ((count >= self.min_steps.index_select(0, slots)) & (stop_probs > self._threshold))
                | ~finite
            )
            if self._step_vae is not None:
                # Halting rows continue for their flush step and stop there.
                drain = self.draining.index_select(0, slots)
                self.draining.index_copy_(0, slots, live & halt)
                flush_out = drain if whole else flush_out.index_copy(0, group_rows, drain)
                continuing = live
            else:
                continuing = live & ~halt
            new_history = torch.cat([history[:, self._patch_size :], latent], dim=1)
            self.history.index_copy_(0, slots, torch.where(live[:, None, None], new_history, history))
            self.next_embed.index_copy_(
                0, slots, torch.where(live[:, None], next_embeds, self.next_embed.index_select(0, slots))
            )
            self.active.index_copy_(0, slots, live & ~halt)
            group_tokens = torch.where(
                continuing,
                torch.full_like(tokens[: len(group)], self._continue_token),
                torch.full_like(tokens[: len(group)], self._stop_token),
            )
            if whole:
                # The flow head returns views of its graph's static outputs,
                # which the next step overwrites while the runner's copy of
                # this snapshot may still be pending: copy into owned storage.
                tokens, emitted_out = group_tokens, live
                latent_out.copy_(latent)
            else:
                tokens.index_copy_(0, group_rows, group_tokens)
                latent_out.index_copy_(0, group_rows, latent)
                emitted_out.index_copy_(0, group_rows, live)

        self._tokens = tokens
        self._request_errors = (failed_out, "Non-finite Ming3 flow-head output in decode step.")
        # The logits path (when sampler features need it) forces the same tokens.
        llm._mrv2_next_token_ids = tokens
        if self._step_vae is not None:
            self._decode_step_audio(input_batch, rows, slots_np, latent_out, emitted_out, flush_out)
            return hidden, {"ming3_step_audio": True}
        return hidden, _LatentRowsSnapshot(
            latent_out,
            emitted_out,
            latent_key=llm.latent_patch_key,
            flag_key=llm.has_patch_key,
            partitioned=self._partitioned,
        )

    def _decode_step_audio(
        self,
        input_batch: InputBatch,
        rows: NDArray[np.integer],
        slots_np: NDArray[np.integer],
        latent: torch.Tensor,
        emitted: torch.Tensor,
        flush: torch.Tensor,
    ) -> None:
        """Hold each stream's first latent and decode every other stream's held latent to PCM."""
        num_reqs = input_batch.num_reqs
        device = latent.device
        prompt_done = np.asarray(input_batch.num_computed_tokens_np[:num_reqs]) >= np.asarray(
            input_batch.prefill_len_np[:num_reqs]
        )
        first_latent = [row for row in rows.tolist() if not prompt_done[row]]
        decode = [row for row in rows.tolist() if prompt_done[row]]
        vae = self._step_vae
        assert vae is not None
        if not first_latent and not decode:
            self._step_audio = _StepAudioOutput(None, None, None, None, [], num_reqs, self._sample_rate)
            return
        slots = slots_np[decode]
        # Host indices for both groups in one upload.
        index = index_to_device(np.concatenate([slots_np[first_latent], first_latent, slots, decode]).tolist(), device)
        primed = len(first_latent)
        side = vae.stream
        if side is not None:
            side.wait_stream(torch.cuda.current_stream(device))
            # These are step-owned tensors, not graph outputs reused by the next AR step.
            for value in (index, latent, emitted, flush):
                value.record_stream(side)
        # Keep priming, decoder state and cursors ordered on the same stream,
        # including when a cancelled request's slot is reused.
        with torch.cuda.stream(side) if side is not None else nullcontext():
            if primed:
                vae.prime(index[:primed], latent.index_select(0, index[primed : 2 * primed]).flatten(1))
                self._vae_positions.index_fill_(0, index[:primed], 0)
            if decode:
                packed = index[2 * primed :].view(2, len(decode))
                positions = self._vae_positions.index_select(0, packed[0])
                flush = flush.index_select(0, packed[1])
                valid = emitted.index_select(0, packed[1]) | flush
                audio, start, end = vae.step(
                    packed[0],
                    positions,
                    positions == 0,
                    latent.index_select(0, packed[1]).flatten(1),
                    flush,
                    valid,
                )
                # Advance the cursor and decoder KV length under the same mask.
                self._vae_positions.index_copy_(0, packed[0], positions + valid * vae.scale)
            event = side.record_event() if decode and side is not None else None
        if not decode:
            self._step_audio = _StepAudioOutput(None, None, None, None, [], num_reqs, self._sample_rate)
            return
        audio_rows: list[tuple[int, _AudioChunks]] = []
        for row in decode:
            req_id = str(input_batch.req_ids[row])
            chunks = self._audio_chunks.get(req_id)
            if chunks is None:
                chunks = self._audio_chunks[req_id] = _AudioChunks(self._samples_per_latent, self._chunk_latents)
            audio_rows.append((row, chunks))
        self._step_audio = _StepAudioOutput(audio, start, end, flush, audio_rows, num_reqs, self._sample_rate, event)

    def get_request_errors(self) -> tuple[torch.Tensor, str] | None:
        return self._request_errors

    def prepare_streaming_audio_output(
        self, input_batch: InputBatch, req_states: RequestState, outputs: dict[str, Any]
    ) -> _StepAudioOutput | None:
        audio, self._step_audio = self._step_audio, None
        return audio

    def sample_determined_tokens(self, batch: InputBatch, sampler: Sampler) -> SamplerOutput | None:
        """Return the tokens decided in postprocess unless a sampler feature needs logits."""
        tokens = self._tokens
        if tokens is None or tokens.numel() != batch.num_reqs or type(sampler) is not Sampler:
            return None
        rows = batch.idx_mapping_np
        if (
            sampler.compute_nans
            or sampler.return_sampling_mask
            or sampler.trace_replay_state is not None
            or sampler.get_logprobs_dims(rows) is not None
            or (
                sampler.thinking_budget_state.enabled
                and np.any(sampler.thinking_budget_state.use_thinking_budget[rows])
            )
        ):
            return None
        for processor in sampler.logits_processors:
            if type(processor) is LogitBiasState:
                active = np.any(processor.use_logit_bias[rows])
            elif type(processor) is PenaltiesState:
                active = np.any(processor.use_penalty[rows])
            elif type(processor) is BadWordsState:
                active = np.any(processor.num_bad_words.np[rows])
            else:
                return None
            if active:
                return None
        num_sampled, num_rejected = get_num_sampled_and_rejected(
            batch.seq_lens.new_ones(batch.num_reqs),
            batch.seq_lens,
            batch.cu_num_logits,
            batch.idx_mapping,
            sampler.req_states.prefill_len.gpu,
        )
        return SamplerOutput(tokens.reshape(-1, 1), None, None, num_sampled, num_rejected)


__all__ = ["Ming3ModelState"]
