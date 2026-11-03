# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Drama 3 request state on Model Runner V2.

The model samples every frame inside its forward. This state keeps what that
forward reads at fixed device addresses:

* per request slot: sampling parameters, RAS window and the newest frame;
* per step: each token row's input frame and next-frame label (prompt rows),
  which slot a row reads (decode rows), and the one sample row per request.

Prompt rows are the only per-step host-to-device data; a pure decode step
uploads one slot index per request.

A preempted request resumes as a prefill over its prompt plus the frames it
had generated: their main tokens come from the token history, their codes
from the slot's code history, copied to the host when the request was
preempted.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from vllm.logger import init_logger
from vllm.utils.platform_utils import is_uva_available
from vllm.v1.worker.gpu import buffer_utils
from vllm.v1.worker.gpu.input_batch import get_num_sampled_and_rejected
from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState
from vllm.v1.worker.gpu.sample.output import SamplerOutput

from vllm_omni.model_executor.output_snapshot import PackedOutputSnapshot, RequestOutputSnapshot
from vllm_omni.worker_v2.model_states.omni_model_state import OmniModelState
from vllm_omni.worker_v2.warmup import is_warmup_request

logger = init_logger(__name__)

_MASK64 = (1 << 64) - 1


def _broadcast_tp_tensor(tensor: torch.Tensor) -> None:
    """Copy ``tensor`` from tensor-parallel rank 0 when that group exists."""
    from vllm.distributed.parallel_state import (
        get_tensor_model_parallel_world_size,
        get_tp_group,
        model_parallel_is_initialized,
    )

    if not model_parallel_is_initialized():
        return
    if get_tensor_model_parallel_world_size() <= 1:
        return
    get_tp_group().broadcast(tensor, src=0)


def _frame_seed(seed: int, frame: int) -> int:
    """Seed of a request's ``frame``-th sampled frame (splitmix64 of seed and index)."""
    x = (seed * 0x9E3779B97F4A7C15 + frame + 1) & _MASK64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _MASK64
    return x ^ (x >> 31)


class _FrameRowsSnapshot(PackedOutputSnapshot):
    """One owned ``[num_reqs, 11]`` frame tensor; rows that sampled nothing are empty."""

    def __init__(self, frames: torch.Tensor, valid: list[bool], *, partitioned: bool) -> None:
        dict.__init__(self, {"codes": {"audio": self._rows(frames, valid)}})
        self._frames = frames
        self._valid = valid
        self._partitioned = partitioned
        self.producer_event = None

    @staticmethod
    def _rows(frames: torch.Tensor, valid: list[bool]) -> list[torch.Tensor]:
        empty = frames.new_empty((0, frames.shape[-1]))
        return [row if ok else empty for row, ok in zip(frames.split(1), valid, strict=True)]

    def copy_to_cpu(self, copy_tensor):
        rows = self._rows(copy_tensor(self._frames), self._valid)
        if self._partitioned:
            return RequestOutputSnapshot([{"codes.audio": row} for row in rows])
        return {"codes": {"audio": rows}}


class _SuspendedRequest:
    """A preempted request's generated codes, on their way to the host."""

    def __init__(self, history: torch.Tensor, count: torch.Tensor, event: torch.cuda.Event | None) -> None:
        self.history = history
        self.count = count
        self.event = event

    def codes(self, frames: int) -> torch.Tensor:
        """``[frames, num_codebooks]`` int64 codes of the first ``frames`` generated frames."""
        if self.event is not None:
            self.event.synchronize()
        available = int(self.count.item())
        if available < frames:
            raise RuntimeError(f"resuming needs {frames} generated frames; {available} were kept")
        return self.history[:frames].to(torch.int64)


class FishOmniModelState(OmniModelState, MambaHybridModelState):
    """Omni request lifecycle over the upstream hybrid (Gated DeltaNet) state.

    MRO: Omni -> MambaHybrid -> Default, so attention metadata carries the
    padding-aware prefill mask the GDN kernels need.
    """

    def __init__(self, vllm_config, model, encoder_cache, device):
        super().__init__(vllm_config, model, encoder_cache, device)
        self.buffers = model.buffers
        self.max_num_seqs = self.scheduler_config.max_num_seqs
        self.frame_width = model.frame_width
        self.prompts: dict[int, np.ndarray] = {}
        self.prompt_len = np.zeros(self.max_num_seqs, dtype=np.int64)
        # The request's own prompt rows (prompt_len also counts the generated
        # frames a resumed request brings back as prompt rows).
        self.base_prompt_len = np.zeros(self.max_num_seqs, dtype=np.int64)
        # Slot writes waiting for the next step, by slot: a slot reused before
        # the step keeps only its newest request's values (index_copy_ with
        # repeated indices would pick one of them arbitrarily).
        self._pending_slots: dict[int, tuple[list[float], list[int], list[int]]] = {}
        self._pending_history: dict[int, torch.Tensor] = {}
        self._valid_rows: list[bool] = []
        self._partitioned = bool(getattr(vllm_config.model_config, "async_chunk", False))
        self._warned_top_k = False
        # Seeds of seeded requests by request id. A sampled frame's noise comes
        # from (seed, frame index) alone: not from its batch row, the other
        # requests, how its prompt was chunked or whether it was preempted.
        self._seeds: dict[str, int] = {}
        self._seed_generator: torch.Generator | None = None
        self._suspended: dict[str, _SuspendedRequest] = {}
        depth = buffer_utils._DEFAULT_MAX_CONCURRENCY + 2
        max_tokens = self.buffers.max_num_tokens

        def pool(shape, dtype=torch.int64):
            return buffer_utils.UvaBufferPool(shape, dtype, depth)

        self._token_pool = pool((max_tokens, 2 * self.frame_width + 4))
        self._decode_pool = pool((max_tokens, 1))
        self._sample_pool = pool((self.max_num_seqs, 3))
        self._slot_pool = pool((self.max_num_seqs, 5 + model.buffers.slot_ras.shape[1]))
        self._last_decode_key: tuple[int, ...] | None = None
        if not is_uva_available():
            logger.warning("fish_omni: UVA is unavailable; per-step uploads use staged copies")

    # ------------------------------------------------------------------
    # Requests
    # ------------------------------------------------------------------

    def _request_info(self, req_index: int) -> dict[str, Any]:
        info = dict(self.intermediate_buffer.buffers[req_index])
        extra = info.pop("additional_information", None)
        if isinstance(extra, dict):
            info = {**extra, **info}
        return info

    def add_request(self, req_index: int, new_req_data) -> None:
        super().add_request(req_index, new_req_data)
        req_id = str(new_req_data.req_id)
        prompt_ids = list(new_req_data.prompt_token_ids or [])
        token_ids = list(new_req_data.prefill_token_ids or prompt_ids)
        if is_warmup_request(req_id):
            # Warmup requests: text rows with zero codes.
            rows = torch.zeros((len(prompt_ids), self.frame_width), dtype=torch.int64)
            rows[:, 0] = torch.tensor(prompt_ids, dtype=torch.int64)
        else:
            rows = self.model.build_frames(self._request_info(req_index), self.device)
        if rows.shape[0] != len(prompt_ids):
            raise ValueError(
                f"fish_omni prompt has {rows.shape[0]} rows but request {req_id} reserved {len(prompt_ids)} "
                "placeholder tokens; size prompt_token_ids with "
                "estimate_prompt_len_from_additional_information"
            )
        generated = len(token_ids) - len(prompt_ids)
        suspended = self._suspended.pop(req_id, None)
        history = None
        if generated > 0:
            # A preempted request resumes: its generated frames become prompt rows.
            if suspended is None:
                raise RuntimeError(f"fish_omni request {req_id} resumes without the codes of its generated frames")
            history = suspended.codes(generated)
            main = torch.tensor(token_ids[len(prompt_ids) :], dtype=torch.int64).unsqueeze(-1)
            rows = torch.cat((rows, torch.cat((main, history), dim=-1)))
            logger.info("fish_omni: resuming preempted request %s after %d generated frames", req_id, generated)
        self.prompts[req_index] = np.ascontiguousarray(rows.numpy())
        self.prompt_len[req_index] = rows.shape[0]
        self.base_prompt_len[req_index] = len(prompt_ids)
        if history is not None:
            self._pending_history[req_index] = history.to(torch.int16)
        else:
            self._pending_history.pop(req_index, None)

        params = new_req_data.sampling_params
        temperature = float(params.temperature) if params is not None else 1.0
        top_k = int(params.top_k) if params is not None and params.top_k and params.top_k > 0 else 1 << 20
        if top_k > self.model.max_top_k:
            if not self._warned_top_k:
                logger.warning(
                    "fish_omni fused sampling caps top_k at %d (request asked for %s); "
                    "set hf_overrides fish_omni_max_top_k to raise it",
                    self.model.max_top_k,
                    params.top_k if params is not None else None,
                )
                self._warned_top_k = True
            top_k = self.model.max_top_k
        top_p = float(params.top_p) if params is not None else 1.0
        seed = getattr(params, "seed", None) if params is not None else None
        if seed is not None:
            self._seeds[req_id] = int(seed) & _MASK64
        window = self.buffers.slot_ras.shape[1]
        tail = rows[-window:, 0].tolist()
        ras = [-1] * (window - len(tail)) + tail
        self._pending_slots[req_index] = ([temperature, top_p], [top_k, generated], ras)

    def remove_request(self, req_index_or_id) -> None:
        index = self._resolve_req_index(req_index_or_id)
        if index is not None:
            self.prompts.pop(index, None)
        super().remove_request(req_index_or_id)

    def on_request_preempted(self, req_id: str, req_index: int) -> None:
        """Copy the request's generated codes to the host before its slot is reused."""
        super().on_request_preempted(req_id, req_index)
        buffers = self.buffers
        history = buffers.slot_history[req_index]
        count = buffers.slot_generated[req_index : req_index + 1]
        event = None
        if history.is_cuda:
            # Stream-ordered after the step that sampled the last frame.
            host_history = torch.empty(history.shape, dtype=history.dtype, pin_memory=True)
            host_count = torch.empty(count.shape, dtype=count.dtype, pin_memory=True)
            host_history.copy_(history, non_blocking=True)
            host_count.copy_(count, non_blocking=True)
            event = torch.cuda.Event()
            event.record()
        else:
            # Native NPU tensors are not CUDA tensors. Blocking host copies
            # complete before the slot can be reused; copy=True also keeps CPU
            # snapshots independent of the original slot storage.
            host_history = history.to(device="cpu", non_blocking=False, copy=True)
            host_count = count.to(device="cpu", non_blocking=False, copy=True)
        self._suspended[req_id] = _SuspendedRequest(host_history, host_count, event)

    def on_requests_finished(self, req_ids: set[str]) -> None:
        super().on_requests_finished(req_ids)
        for req_id in req_ids:
            self._seeds.pop(req_id, None)
            self._suspended.pop(req_id, None)

    def _flush_slot_writes(self) -> None:
        if not self._pending_slots:
            return
        buffers = self.buffers
        # [slot, temperature bits, top_p bits, top_k, generated, ras...] as int64.
        packed = np.zeros((len(self._pending_slots), self._slot_pool.size[1]), dtype=np.int64)
        for i, (slot, (floats, ints, ras)) in enumerate(self._pending_slots.items()):
            packed[i, 0] = slot
            packed[i, 1:3] = np.asarray(floats, dtype=np.float64).view(np.int64)
            packed[i, 3:5] = ints
            packed[i, 5:] = ras
        self._pending_slots.clear()
        device = self._slot_pool.copy_to_uva(packed).to(self.device, non_blocking=True)
        slots = device[:, 0]
        floats = device[:, 1:3].view(torch.float64).to(torch.float32)
        buffers.slot_temperature.index_copy_(0, slots, floats[:, 0])
        buffers.slot_top_p.index_copy_(0, slots, floats[:, 1])
        buffers.slot_top_k.index_copy_(0, slots, device[:, 3])
        buffers.slot_generated.index_copy_(0, slots, device[:, 4])
        buffers.slot_ras.index_copy_(0, slots, device[:, 5:])
        for slot, history in self._pending_history.items():
            buffers.slot_history[slot, : history.shape[0]].copy_(history, non_blocking=True)
        self._pending_history.clear()
        self._last_decode_key = None

    # ------------------------------------------------------------------
    # Step inputs
    # ------------------------------------------------------------------

    def run_preprocess(self, input_batch, model_inputs, req_states=None, mtp_batch_descriptor_dispatcher=None):
        self._flush_slot_writes()
        buffers = self.buffers
        num_reqs = input_batch.num_reqs
        num_tokens = input_batch.num_tokens
        padded_tokens = input_batch.num_tokens_after_padding
        slots = input_batch.idx_mapping_np[:num_reqs].astype(np.int64)
        scheduled = input_batch.num_scheduled_tokens[:num_reqs]
        computed = input_batch.num_computed_tokens_np[:num_reqs]
        prompt_len = self.prompt_len[slots]
        uniform = num_tokens == num_reqs
        # The forward always runs this many sample rows (see FishOmniBuffers).
        num_samples = min(padded_tokens, self.max_num_seqs)
        last_positions = computed + scheduled - 1
        valid = last_positions >= prompt_len - 1
        self._valid_rows = valid.tolist()
        # A sampling row draws the request's frame number last_position + 1 - prompt rows.
        self._draw_noise(input_batch.req_ids[:num_reqs], last_positions + 1 - self.base_prompt_len[slots], valid)

        if uniform and not np.any(computed < prompt_len):
            # Pure decode: every row reads its slot's newest frame and samples.
            key = (padded_tokens, *slots.tolist())
            if key != self._last_decode_key:
                padded_slots = np.full((padded_tokens, 1), buffers.dummy_slot, dtype=np.int64)
                padded_slots[:num_reqs, 0] = slots
                device = self._decode_pool.copy_to_uva(padded_slots).to(self.device, non_blocking=True)[:, 0]
                buffers.token_slots[:padded_tokens].copy_(device)
                buffers.sample_slots[:num_samples].copy_(device[:num_samples])
                buffers.token_from_slot[:padded_tokens].fill_(True)
                rows = torch.arange(padded_tokens, device=self.device)
                # Padding rows inject nothing, so any sample-row count >= num_reqs works.
                sampling = rows < num_reqs
                buffers.sample_token_rows[:padded_tokens].copy_(sampling)
                buffers.token_sample_index[:padded_tokens].copy_(torch.where(sampling, rows, 0))
                buffers.sample_rows[:num_samples].copy_(rows[:num_samples])
                buffers.sample_valid[:num_samples].copy_(rows[:num_samples] < num_reqs)
                self._last_decode_key = key
        else:
            self._stage_mixed_step(input_batch, slots, scheduled, computed, prompt_len, num_samples, valid)
            self._last_decode_key = None
        self.model.num_sample_rows = num_reqs

    def _draw_noise(self, req_ids: list[str], frames: np.ndarray, sampling: np.ndarray) -> None:
        """Fresh sampling noise for the step. A seeded request's sample row (= its
        batch row) that samples frame ``frames[row]`` draws from that frame's seed."""
        noise = self.buffers.noise
        noise.uniform_()
        if self._seeds:
            for row, req_id in enumerate(req_ids):
                seed = self._seeds.get(req_id)
                if seed is None or not sampling[row]:
                    continue
                if self._seed_generator is None:
                    self._seed_generator = torch.Generator(device=noise.device)
                self._seed_generator.manual_seed(_frame_seed(seed, int(frames[row])))
                noise[row].uniform_(generator=self._seed_generator)
        # Each tensor-parallel rank samples inside the forward. Device RNG is
        # not shared, so rank 0's noise is the one every rank uses.
        _broadcast_tp_tensor(noise)

    def run_postprocess(self, hidden_states, input_batch) -> None:
        self.model.num_sample_rows = None
        super().run_postprocess(hidden_states, input_batch)

    def _stage_mixed_step(self, input_batch, slots, scheduled, computed, prompt_len, num_samples, valid) -> None:
        buffers = self.buffers
        num_reqs = input_batch.num_reqs
        num_tokens = input_batch.num_tokens
        padded_tokens = input_batch.num_tokens_after_padding
        width = self.frame_width
        # Per token row: frame | label | slot | from_slot | samples_next |
        # sample row. Decode rows read their slot's frame and sample; padding
        # rows read the dummy slot and sample nothing.
        token = np.zeros((padded_tokens, 2 * width + 4), dtype=np.int64)
        slot_col, from_col, next_col, index_col = 2 * width, 2 * width + 1, 2 * width + 2, 2 * width + 3
        token[:, slot_col] = buffers.dummy_slot
        token[:, from_col] = 1
        token[:num_tokens, next_col] = 1
        token[:num_tokens, slot_col] = np.repeat(slots, scheduled)
        token[:num_tokens, index_col] = np.repeat(np.arange(num_reqs, dtype=np.int64), scheduled)
        starts = input_batch.query_start_loc_np[:num_reqs]
        for i in np.flatnonzero(computed < prompt_len).tolist():
            slot, start, count, done, length = (
                int(slots[i]),
                int(starts[i]),
                int(scheduled[i]),
                int(computed[i]),
                int(prompt_len[i]),
            )
            end = done + count
            if end > length:
                raise RuntimeError(f"fish_omni request in slot {slot} schedules prompt and generated rows together")
            prompt = self.prompts[slot]
            rows = token[start : start + count]
            rows[:, :width] = prompt[done:end]
            labels = prompt[done + 1 : min(end + 1, length)]
            rows[: labels.shape[0], width : 2 * width] = labels
            rows[:, from_col] = 0
            rows[:, next_col] = 0
            if end == length:
                rows[-1, next_col] = 1
        # Sample row i is request i's last row; rows past num_reqs pad.
        sample = np.zeros((num_samples, 3), dtype=np.int64)
        sample[:, 1] = buffers.dummy_slot
        sample[:num_reqs, 0] = starts + scheduled - 1
        sample[:num_reqs, 1] = slots
        sample[:num_reqs, 2] = valid
        device_token = self._token_pool.copy_to_uva(token).to(self.device, non_blocking=True)
        device_sample = self._sample_pool.copy_to_uva(sample).to(self.device, non_blocking=True)
        buffers.frames[:padded_tokens].copy_(device_token[:, :width])
        buffers.labels[:padded_tokens].copy_(device_token[:, width : 2 * width])
        buffers.token_slots[:padded_tokens].copy_(device_token[:, slot_col])
        buffers.token_from_slot[:padded_tokens].copy_(device_token[:, from_col].bool())
        buffers.sample_token_rows[:padded_tokens].copy_(device_token[:, next_col].bool())
        buffers.token_sample_index[:padded_tokens].copy_(device_token[:, index_col])
        buffers.sample_rows[:num_samples].copy_(device_sample[:, 0])
        buffers.sample_slots[:num_samples].copy_(device_sample[:, 1])
        buffers.sample_valid[:num_samples].copy_(device_sample[:, 2].bool())

    # ------------------------------------------------------------------
    # Outputs
    # ------------------------------------------------------------------

    def sample_determined_tokens(self, batch, sampler):
        num_reqs = batch.num_reqs
        # Owned: the next forward overwrites the sampled-frame buffer.
        tokens = self.buffers.sampled_frames[:num_reqs, 0:1].clone()
        num_sampled, num_rejected = get_num_sampled_and_rejected(
            batch.seq_lens.new_ones(num_reqs),
            batch.seq_lens,
            batch.cu_num_logits,
            batch.idx_mapping,
            sampler.req_states.prefill_len.gpu,
        )
        return SamplerOutput(tokens, None, None, num_sampled, num_rejected)

    def postprocess_model_output(self, model_output: Any, input_batch, req_states):
        hidden = model_output.text_hidden_states if hasattr(model_output, "text_hidden_states") else model_output
        num_reqs = input_batch.num_reqs
        frames = self.buffers.sampled_frames[:num_reqs].clone()
        valid = self._valid_rows[:num_reqs] if len(self._valid_rows) >= num_reqs else [False] * num_reqs
        return hidden, _FrameRowsSnapshot(frames, valid, partitioned=self._partitioned)


def create_fish_omni_model_state(vllm_config, model, encoder_cache, device):
    """CUDA keeps the upstream hybrid state; NPU builds Ascend attention metadata."""
    if getattr(device, "type", None) == "npu":
        from vllm_omni.platforms.npu.models.fish_omni_model_state import NPUFishOmniModelState

        return NPUFishOmniModelState(vllm_config, model, encoder_cache, device)
    return FishOmniModelState(vllm_config, model, encoder_cache, device)


__all__ = ["FishOmniModelState", "create_fish_omni_model_state"]
