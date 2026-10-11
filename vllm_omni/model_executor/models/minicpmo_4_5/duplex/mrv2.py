# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""MRv2 registration of the native MiniCPM duplex sampling policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from vllm.logger import init_logger
from vllm.sampling_params import SamplingParams
from vllm.v1.sample.logits_processor import LogitsProcessors
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import Sampler as LegacySampler
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu.sample.sampler import Sampler

from vllm_omni.model_executor.duplex_sampling import DuplexSamplingRow
from vllm_omni.platforms import current_omni_platform
from vllm_omni.utils.device_copy import index_to_device
from vllm_omni.worker_v2.omni_sampler import OmniSampler

logger = init_logger(__name__)

if TYPE_CHECKING:
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import (
        MiniCPMO45OmniForConditionalGeneration,
        _MiniCPMO45PendingSamples,
    )
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_tts import (
        MiniCPMO45OmniTTSForConditionalGeneration,
    )


def _cuda_sampler_device(sampler: Any) -> torch.device | None:
    """The sampler's CUDA device, or None where the Triton warmups do not apply."""
    device = getattr(getattr(sampler, "req_states", None), "device", None)
    if not isinstance(device, torch.device) or device.type != "cuda" or not current_omni_platform.is_cuda():
        return None
    return device


def _top_k_top_p_warmup_batches(max_rows: int, device: torch.device) -> list[int]:
    """One batch size per Triton specialization reachable within ``max_rows``.

    ``_topk_topp_kernel`` takes the batch size as a plain int (specialized on
    ``== 1`` and ``% 16``), and the small-batch top-p pipeline compiles one
    variant per split count ``S`` (a constexpr derived from batch size and SM
    count). 1, 2, 3 and the capacity are always included.
    """
    batches = {rows for rows in (1, 2, 3, max_rows) if 1 <= rows <= max_rows}
    try:
        from vllm.utils.platform_utils import num_compute_units
        from vllm.v1.sample.ops.topk_topp_triton import _SPLIT_MAX_BATCH, _topp_split_count
    except ImportError:
        return sorted(batches)
    num_sm = num_compute_units(device.index if device.index is not None else torch.accelerator.current_device_index())
    keys: dict[tuple[bool, bool, int], int] = {}
    for rows in range(1, max_rows + 1):
        splits = _topp_split_count(rows, num_sm) if rows <= _SPLIT_MAX_BATCH else 0
        keys.setdefault((rows == 1, rows % 16 == 0, splits), rows)
    return sorted(batches | set(keys.values()))


def _warmup_top_k_top_p(sampler: Any, device: torch.device) -> None:
    """Compile vLLM's Triton top-k/top-p masks on disposable tensors.

    Mirrors ``SamplingStates.get_top_k_top_p``: int32 ``top_k`` and float32
    ``top_p`` slot tables gathered by the int32 ``expanded_idx_mapping``, and
    ``Sampler.apply_sampling_params``' contiguous float32 logits at the stage's
    vocab. Both the seeded codec path and the stock draw reach this function
    with active top-k and top-p (the Talker's deployed top_k/top_p). The masks
    draw no random numbers. vLLM's lazily built per-device kernel caches are
    restored afterwards so the warmup leaves no allocation behind.
    """
    from vllm.v1.sample.ops import topk_topp_triton
    from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p

    states = sampler.sampling_states
    vocab_size = int(states.vocab_size)
    max_rows = int(states.max_num_reqs) * max(1, int(getattr(sampler, "num_speculative_tokens", 1)))
    caches = [
        cache
        for cache in (
            getattr(topk_topp_triton, name, None)
            for name in ("_TRITON_BUFFER_CACHE", "_TRITON_TABLE_CACHE", "_TRITON_SPLIT_CACHE")
        )
        if isinstance(cache, dict)
    ]
    snapshots = [dict(cache) for cache in caches]
    # Any k < vocab and p < 1 enables both masks; values never affect compilation.
    top_k = torch.full((states.max_num_reqs,), max(1, vocab_size // 2), dtype=torch.int32, device=device)
    top_p = torch.full((states.max_num_reqs,), 0.5, dtype=torch.float32, device=device)
    try:
        for rows in _top_k_top_p_warmup_batches(max_rows, device):
            expanded_idx_mapping = torch.arange(rows, dtype=torch.int32, device=device) % states.max_num_reqs
            # Deterministic logits: the default CUDA generator stays untouched.
            logits = torch.linspace(-8.0, 8.0, vocab_size, dtype=torch.float32, device=device).repeat(rows, 1)
            apply_top_k_top_p(logits, top_k[expanded_idx_mapping], top_p[expanded_idx_mapping])
    finally:
        for cache, snapshot in zip(caches, snapshots, strict=True):
            cache.clear()
            cache.update(snapshot)


@dataclass(frozen=True)
class _CodecRNGCheckpoint:
    generator: torch.Generator
    condition_seq: int
    offset: int


class MiniCPMO45SeededCodecSampler(Sampler):
    """Keep V1's request-local seeded draw for native duplex codec rows.

    MRv2's Gumbel draw changes the seeded codec sequence and onset length.
    Apply MRv2's processors once, reuse V1's Torch draw for these rows, and
    let MRv2 own accepted counts, logprobs and token-state writes. Ordinary
    and unseeded rows retain the runner's sampler. RNG survives slot changes
    and streaming appends; only accepted prefill/decode rows advance it.
    """

    omni_static_staged_writes = True
    decode_graphs: SeededCodecDecodeGraphs | None = None

    def __init__(self, base_sampler: Sampler, model: MiniCPMO45OmniTTSForConditionalGeneration) -> None:
        # Delegate buffers and staged writes instead of copying sampler state.
        self.base_sampler = base_sampler
        self.model = model
        self._params_by_slot: dict[int, tuple[int | None, bool]] = {}
        self._generators: dict[str, torch.Generator] = {}
        self._legacy = LegacySampler()
        self._rows: tuple[int, ...] = ()
        self._accepted: tuple[int, ...] = ()
        config = getattr(model, "vllm_config", None)
        self._rewind_async_lookahead = bool(config is not None and config.scheduler_config.async_scheduling)
        self._condition_seqs: dict[str, int] = {}
        self._committed_offsets: dict[str, _CodecRNGCheckpoint] = {}
        # A decode burst's draws past each row's last kept one (see defer_rng_rewind).
        self._pending_burst_rewind: (
            tuple[list[torch.Generator | None], list[list[int]], np.ndarray, torch.cuda.Event] | None
        ) = None
        # Capture supported native-duplex sampling at readiness. Explicit
        # eager mode and unsupported processor combinations keep the fallback.
        self.decode_graphs: SeededCodecDecodeGraphs | None = None
        model_config = getattr(config, "model_config", None)
        if (
            _cuda_sampler_device(self) is not None
            and getattr(model_config, "session_mode", "duplex") == "duplex"
            and not getattr(model_config, "enforce_eager", False)
            and SeededCodecDecodeGraphs.supported(self)
        ):
            self.decode_graphs = SeededCodecDecodeGraphs(self)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base_sampler, name)

    def add_request(self, req_idx: int, sampling_params: SamplingParams) -> None:
        # Slot-keyed admission parameters stay bounded by the runner capacity.
        self._params_by_slot[req_idx] = (sampling_params.seed, sampling_params.temperature == 0)
        self.base_sampler.add_request(req_idx, sampling_params)
        if self.decode_graphs is not None:
            self.decode_graphs.admit(req_idx)

    def on_requests_finished(self, request_ids: set[str] | list[str]) -> None:
        for request_id in request_ids:
            self._generators.pop(request_id, None)
            self._condition_seqs.pop(request_id, None)
            self._committed_offsets.pop(request_id, None)

    def _prepare_condition_rng(self, request_id: str, generator: torch.Generator) -> None:
        if not self._rewind_async_lookahead:
            return
        seq = self.model._request_condition_states[request_id]["condition_seq"]
        previous = self._condition_seqs.get(request_id)
        if previous is not None and previous != seq:
            checkpoint = self._committed_offsets.pop(request_id, None)
            if checkpoint is not None and checkpoint.condition_seq == previous and checkpoint.generator is generator:
                # Async scheduling can draw once more with EOS as the input.
                # Its result belongs to the retired condition; the successor
                # starts after the last real draw, including the terminal draw.
                generator.set_offset(checkpoint.offset)
        self._condition_seqs[request_id] = seq

    def codec_rng_checkpoint(self, request_id: str) -> _CodecRNGCheckpoint | None:
        """Snapshot a host Philox offset; never read or synchronize CUDA output."""
        if not self._rewind_async_lookahead:
            return None
        generator = self._generators.get(request_id)
        seq = self._condition_seqs.get(request_id)
        if generator is None or seq is None:
            return None
        return _CodecRNGCheckpoint(generator, seq, generator.get_offset())

    def commit_codec_rng(self, request_id: str, checkpoint: _CodecRNGCheckpoint) -> None:
        if (
            self._generators.get(request_id) is checkpoint.generator
            and self._condition_seqs.get(request_id) == checkpoint.condition_seq
        ):
            self._committed_offsets[request_id] = checkpoint

    @torch.inference_mode()
    def warmup(self) -> None:
        """Prime the live codec sampling kernels without consuming live RNG."""
        device = _cuda_sampler_device(self)
        if device is None:
            return
        # Every Talker step applies top-k/top-p before either draw. On CUDA
        # vLLM's MRv2 runner skips these kernels' registered JIT warmups.
        _warmup_top_k_top_p(self.base_sampler, device)
        if self.decode_graphs is not None:
            try:
                self.decode_graphs.capture(device)
            except Exception:
                logger.exception("MiniCPM-o Talker: seeded codec sampler graph capture failed; staying eager")
                self.decode_graphs = None
        if self.req_states.max_num_reqs < 2:
            return
        from vllm_omni.utils.seeded_exponential import fill_exponential_rows

        # C1 warmup never enters the batched seeded path. Its first dispatch
        # otherwise loads/compiles the Triton kernel on a live C8 onset.
        # Match the row-indexed specialization used by random_sample; these
        # disposable generators leave request and default generators intact.
        # The packed int64 state table's offset row has different pointer
        # alignment for odd batches. Prime both layouts within slot capacity.
        for rows in range(2, min(self.req_states.max_num_reqs, 3) + 1):
            noise = torch.empty((rows, self.req_states.vocab_size), dtype=torch.float32, device=device)
            generators = [torch.Generator(device=device).manual_seed(seed) for seed in range(rows)]
            fill_exponential_rows(noise, generators, list(range(rows)))

    def _seeded_rows(self, logits: torch.Tensor, input_batch: InputBatch) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """(seeded native duplex rows, those whose sample is accepted this step)."""
        infos = getattr(self.model, "_mrv2_output_infos", ())
        slots = input_batch.idx_mapping_np
        if self.num_speculative_tokens != 1 or logits.shape[0] != len(slots) or len(infos) != len(slots):
            return (), ()
        rows = []
        for row, slot in enumerate(slots):
            params = self._params_by_slot.get(int(slot))
            if (
                infos[row].get("native_duplex") is True
                and params is not None
                and params[0] is not None
                and self.req_states.index_to_req_id.get(int(slot)) is not None
            ):
                rows.append(row)
        if not rows:
            return (), ()
        # The optimistic async upper bound can exceed a partial prefill's
        # accepted span. These scheduled CPU fields describe it exactly.
        complete = (
            input_batch.num_computed_prefill_tokens_np + input_batch.num_scheduled_tokens >= input_batch.prefill_len_np
        )
        return tuple(rows), tuple(row for row in rows if complete[row])

    def defer_rng_rewind(
        self,
        generators: list[torch.Generator | None],
        offsets: list[list[int]],
        num_sampled_np: np.ndarray,
        ready: torch.cuda.Event,
    ) -> None:
        """Rewind each burst row's generator to just after its last kept draw.

        A decode burst draws for every row on every step; a row that drew
        codec EOS keeps only its draws up to that one. ``offsets[k]`` holds the
        offsets after step ``k``; ``num_sampled_np`` (valid once ``ready``)
        counts each row's kept steps. Applied before the next draw.
        """
        self._pending_burst_rewind = (generators, offsets, num_sampled_np, ready)

    def _apply_burst_rewind(self) -> None:
        pending, self._pending_burst_rewind = self._pending_burst_rewind, None
        assert pending is not None
        generators, offsets, num_sampled_np, ready = pending
        ready.synchronize()
        for row, generator in enumerate(generators):
            kept = int(num_sampled_np[row])
            if generator is not None and 0 < kept < len(offsets):
                generator.set_offset(offsets[kept - 1][row])

    def _request_generator(self, slot: int, device: torch.device) -> torch.Generator:
        """The request's codec generator (created on its first accepted draw)."""
        if self._pending_burst_rewind is not None:
            self._apply_burst_rewind()
        seed = self._params_by_slot[slot][0]
        request_id = self.req_states.index_to_req_id[slot]
        generator = self._generators.get(request_id)
        if generator is None:
            assert seed is not None
            generator = torch.Generator(device=device).manual_seed(seed)
            self._generators[request_id] = generator
        self._prepare_condition_rng(request_id, generator)
        return generator

    def __call__(self, logits: torch.Tensor, input_batch: InputBatch) -> SamplerOutput:
        rows, accepted = self._seeded_rows(logits, input_batch)
        if not rows:
            return self.base_sampler(logits, input_batch)
        self._rows = rows
        self._accepted = accepted
        try:
            return super().__call__(logits, input_batch)
        finally:
            self._rows = self._accepted = ()

    def sample(
        self,
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
        pos: torch.Tensor,
        input_ids: torch.Tensor,
        expanded_local_pos: torch.Tensor,
        seq_lens_upper_bound_np: np.ndarray,
        return_logprobs: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        processed = self.base_sampler.apply_sampling_params(
            logits,
            expanded_idx_mapping,
            idx_mapping,
            idx_mapping_np,
            pos,
            input_ids,
            expanded_local_pos,
            seq_lens_upper_bound_np,
        )
        if self._accepted and len(self._accepted) == logits.shape[0]:
            # Every row is redrawn below; no placeholder ids are needed.
            sampled = None
        elif len(self._rows) == logits.shape[0]:
            sampled = torch.full(
                (logits.shape[0],), int(self.model._codec_eos_id), dtype=torch.int64, device=logits.device
            )
        else:
            # Seeded rows already make the fused stock sampler ineligible.
            # Preserve its Gumbel draw for the other rows after one processor pass.
            sampled, _ = self.base_sampler._sample_random(
                processed, expanded_idx_mapping, idx_mapping_np, pos, None, None, False
            )
        if not self._accepted:
            return sampled, processed
        generators, greedy = {}, []
        for row, index in enumerate(self._accepted):
            slot = int(idx_mapping_np[index])
            generators[row] = self._request_generator(slot, logits.device)
            greedy.append(self._params_by_slot[slot][1])
        indices = None if len(self._accepted) == logits.shape[0] else index_to_device(self._accepted, logits.device)
        selected = processed if indices is None else processed.index_select(0, indices)
        if all(greedy):
            picked = selected.argmax(dim=-1)
        else:
            picked, _ = self._legacy.topk_topp_sampler(selected, generators, None, None)
            if any(greedy):
                picked = torch.where(
                    index_to_device(greedy, logits.device, dtype=torch.bool), selected.argmax(dim=-1), picked
                )
        if indices is None:
            sampled = picked.to(dtype=torch.int64)
        else:
            assert sampled is not None
            sampled.index_copy_(0, indices, picked.to(dtype=torch.int64))
        return sampled, processed


@dataclass
class _SeededDecodeGraph:
    logits: torch.Tensor
    expanded_idx_mapping: torch.Tensor
    mask_eos: torch.Tensor
    forced_eos: torch.Tensor
    # [2, rows] int64: signed Philox seeds, then pre-draw offsets.
    rng_state: torch.Tensor
    # [rows] int64: each row's sequence position (min_tokens stop-id masking).
    pos: torch.Tensor
    graph: torch.cuda.CUDAGraph | None = None
    sampled: torch.Tensor | None = None
    # vLLM's lazily cached Triton scratch tensors the graph reads/writes.
    keepalive: tuple[Any, ...] = ()


class SeededCodecDecodeGraphs:
    """Replay the seeded codec draw of a fixed-size, all-codec batch as one CUDA graph.

    Covers exactly the eager kernels ``MiniCPMO45TalkerSampler`` +
    ``MiniCPMO45SeededCodecSampler.sample`` launch when every row is an
    accepted, seeded, non-greedy native duplex row with active top-k and top-p:
    EOS mask, FP32 copy, then ``Sampler.logits_processors`` in list order --
    ``LogitBiasState`` (allowed ids, logit bias, ``min_tokens`` stop-id
    masking), the Talker's ``_CodecWindowPenaltiesState`` (16-frame codec
    penalty) that ``_install_mrv2_talker_sampler`` puts in the stock penalty
    slot, and ``BadWordsState`` -- then temperature, top-k/top-p, softmax,
    exponential draw, ``probs / q`` argmax and forced EOS. Per-row early
    returns (no bias / stop ids past ``min_tokens``, penalty 1, temperature 1)
    are device-side in those kernels, so their outcome equals the eager
    host-side skip.

    The graph declines (the step runs eagerly) whenever a stage it does not
    replay would act: a processor list other than the installed
    ``[LogitBiasState, _CodecWindowPenaltiesState, BadWordsState]`` objects,
    stock frequency/presence penalties, bad words, stop-id restore for
    structured outputs, min-p, thinking budget, logprobs, greedy rows or
    disabled top-k/top-p.

    RNG: torch's ``exponential_(generator=g)`` is reproduced by
    ``launch_exponential_rows`` from device-resident (seed, offset) pairs, the
    same kernel the batched eager path already uses. Before each replay the
    host claims each request generator's Philox offset exactly as the eager
    draw would (``claim_exponential_draws``) and uploads (seed, offset); the
    request generators therefore advance identically and remain the single
    source of truth for checkpoints and rewinds.
    """

    def __init__(self, sampler: MiniCPMO45SeededCodecSampler) -> None:
        self.sampler = sampler
        base = sampler.base_sampler
        self.vocab_size = int(base.sampling_states.vocab_size)
        self.slot_ok = np.zeros(int(sampler.req_states.max_num_reqs), dtype=bool)
        self.graphs: dict[int, _SeededDecodeGraph] = {}
        self._threads: int | None = None
        # The exact pipeline objects the graph replays (checked by ``supported``).
        self._processors = tuple(base.logits_processors)
        self._logit_bias, self._window, self._bad_words = self._processors
        # [10, max_num_reqs] int32 mirror of the per-slot UVA tables (see _table_sources).
        self._tables: torch.Tensor | None = None

    def _table_sources(self) -> list[torch.Tensor]:
        """The rotating ``UvaBackedTensor`` buffers the replayed kernels read.

        Row order: temperature, top_k, top_p, codec penalty, prompt_len, then
        LogitBiasState's num_allowed_token_ids, num_logit_bias, min_lens,
        num_stop_token_ids and restore_when_all_masked. The ``StagedWriteTensor``
        payloads (token history, allowed/bias/stop ids) and the window's
        ``prefix_history`` keep fixed addresses and are read directly.
        """
        sampler = self.sampler
        states = sampler.base_sampler.sampling_states
        bias = self._logit_bias
        return [
            states.temperature.gpu.view(torch.int32),
            states.top_k.gpu,
            states.top_p.gpu.view(torch.int32),
            self._window.repetition_penalty.gpu.view(torch.int32),
            sampler.req_states.prompt_len.gpu,
            bias.num_allowed_token_ids.gpu,
            bias.num_logit_bias.gpu,
            bias.min_lens.gpu,
            bias.num_stop_token_ids.gpu,
            bias.restore_when_all_masked.gpu,
        ]

    def _refresh_tables(self) -> None:
        """Snapshot the per-slot parameter tables into the graph's static mirror.

        vLLM's ``UvaBackedTensor.copy_to_uva`` rotates ``.gpu`` through a pool
        of UVA buffers on every staged write (request admission), so a graph
        must not capture those addresses: it would keep reading the buffer that
        was current at capture. One stream-ordered stack copies the buffers the
        eager kernels would read at this point into a fixed device tensor.
        """
        assert self._tables is not None
        torch.stack(self._table_sources(), out=self._tables)

    @staticmethod
    def supported(sampler: MiniCPMO45SeededCodecSampler) -> bool:
        from vllm.v1.worker.gpu.sample.bad_words import BadWordsState
        from vllm.v1.worker.gpu.sample.logit_bias import LogitBiasState

        from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_tts import _CodecWindowPenaltiesState

        base = sampler.base_sampler
        processors = list(getattr(base, "logits_processors", ()))
        legacy = getattr(sampler._legacy, "topk_topp_sampler", None)
        # Exactly the stock pipeline with the Talker's window state in the
        # penalty slot; any extra (custom) processor is not replayed.
        return (
            [type(p) for p in processors] == [LogitBiasState, _CodecWindowPenaltiesState, BadWordsState]
            and getattr(base, "penalties_state", None) is processors[1]
            and not getattr(base, "compute_nans", True)
            and getattr(base, "trace_replay_state", None) is None
            and not getattr(base, "return_sampling_mask", True)
            and int(getattr(base, "num_speculative_tokens", 1)) == 1
            and legacy is not None
            and not getattr(legacy, "use_fp64_gumbel", True)
        )

    def admit(self, slot: int) -> None:
        """Record whether ``slot``'s just-admitted parameters fit the captured signature."""
        sampler = self.sampler
        base = sampler.base_sampler
        states = base.sampling_states
        thinking = base.thinking_budget_state
        seed, greedy = sampler._params_by_slot[slot]
        self.slot_ok[slot] = bool(
            seed is not None
            and not greedy
            and base.needs_logits_processing[slot]
            and states.temperature.np[slot] != 0.0
            and states.min_p.np[slot] == 0.0
            and states.top_k.np[slot] != self.vocab_size
            and states.top_p.np[slot] != 1.0
            and states.num_logprobs[slot] < 0
            # LogitBiasState is replayed except its structured-output restore
            # (a host-selected kernel variant); stock frequency/presence
            # penalties and bad words are not replayed.
            and not self._logit_bias.restore_when_all_masked.np[slot]
            and not self._window.base.use_penalty[slot]
            and int(self._bad_words.num_bad_words.np[slot]) == 0
            and not (getattr(thinking, "enabled", False) and thinking.use_thinking_budget[slot])
        )

    def _body(self, entry: _SeededDecodeGraph) -> torch.Tensor:
        from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p
        from vllm.v1.worker.gpu.sample.gumbel import apply_temperature
        from vllm.v1.worker.gpu.sample.logit_bias import apply_logit_bias

        from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_tts import (
            _apply_codec_window_penalty_gpu,
        )
        from vllm_omni.utils.seeded_exponential import launch_exponential_rows

        sampler = self.sampler
        req_states = sampler.req_states
        bias, window = self._logit_bias, self._window
        eos_id = int(sampler.model._codec_eos_id)
        idx = entry.expanded_idx_mapping
        # Per-slot tables: device mirrors refreshed before every replay, never
        # the rotating UvaBackedTensor buffers themselves (see _refresh_tables).
        tables = self._tables
        assert tables is not None
        temperature, top_k, top_p = tables[0].view(torch.float32), tables[1], tables[2].view(torch.float32)
        repetition_penalty, prompt_len = tables[3].view(torch.float32), tables[4]
        # MiniCPMO45TalkerSampler: EOS mask on the raw logits.
        entry.logits[:, eos_id].masked_fill_(entry.mask_eos, float("-inf"))
        # Sampler.apply_sampling_params, in its processor order.
        processed = torch.empty_like(entry.logits, dtype=torch.float32).copy_(entry.logits)
        # logits_processors[0]: LogitBiasState (min_tokens masks codec EOS, a stage stop id).
        apply_logit_bias(
            processed,
            idx,
            entry.pos,
            tables[5],
            bias.allowed_token_ids.gpu,
            tables[6],
            bias.logit_bias_token_ids.gpu,
            bias.logit_bias.gpu,
            tables[7],
            tables[8],
            tables[9],
            bias.stop_token_ids.gpu,
            False,
        )
        # logits_processors[1]: the Talker's window state (its stock base is
        # inactive on every admitted row, see ``admit``).
        _apply_codec_window_penalty_gpu(
            processed,
            idx,
            req_states.all_token_ids.gpu,
            req_states.total_len.gpu,
            prompt_len,
            repetition_penalty,
            window_size=window.window_size,
            prefix_history=window.prefix_history,
        )
        # logits_processors[2]: BadWordsState, inactive on every admitted row.
        apply_temperature(processed, idx, temperature)
        processed = apply_top_k_top_p(processed, top_k[idx], top_p[idx])
        # TopKTopPSampler.forward_native + random_sample with one generator per row.
        probs = processed.softmax(dim=-1, dtype=torch.float32)
        noise = torch.empty_like(probs)
        assert self._threads is not None
        launch_exponential_rows(noise, entry.rng_state[0], entry.rng_state[1], self.vocab_size, self._threads)
        sampled = probs.div_(noise).argmax(dim=-1).view(-1)
        # MiniCPMO45TalkerSampler: forced EOS after sampling.
        sampled.masked_fill_(entry.forced_eos, eos_id)
        return sampled

    @torch.inference_mode()
    def capture(self, device: torch.device) -> None:
        from vllm.v1.sample.ops import topk_topp_triton

        from vllm_omni.utils.seeded_exponential import torch_exponential_policy

        if self.graphs:
            return
        self._threads = torch_exponential_policy(self.vocab_size, device)[0]
        self._tables = torch.zeros(
            (len(self._table_sources()), int(self.sampler.req_states.max_num_reqs)), dtype=torch.int32, device=device
        )
        self._refresh_tables()
        pool = torch.cuda.graph_pool_handle()
        max_rows = int(self.sampler.req_states.max_num_reqs)
        # Largest first: vLLM's top-k scratch buffer grows to its final size
        # before any capture, and smaller batches reuse a prefix view of it.
        for rows in range(max_rows, 0, -1):
            entry = _SeededDecodeGraph(
                logits=torch.linspace(-8.0, 8.0, self.vocab_size, device=device).repeat(rows, 1),
                expanded_idx_mapping=torch.zeros(rows, dtype=torch.int32, device=device),
                mask_eos=torch.zeros(rows, dtype=torch.bool, device=device),
                forced_eos=torch.zeros(rows, dtype=torch.bool, device=device),
                rng_state=torch.zeros((2, rows), dtype=torch.int64, device=device),
                pos=torch.zeros(rows, dtype=torch.int64, device=device),
            )
            # Eager pass: compiles kernels and builds vLLM's lazy caches
            # outside capture. Disposable seeds leave every generator intact.
            self._body(entry)
            torch.accelerator.synchronize(device)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=pool, capture_error_mode="thread_local"):
                entry.sampled = self._body(entry)
            entry.graph = graph
            entry.keepalive = tuple(
                value
                for name in ("_TRITON_BUFFER_CACHE", "_TRITON_TABLE_CACHE", "_TRITON_SPLIT_CACHE")
                for value in getattr(topk_topp_triton, name, {}).values()
            )
            self.graphs[rows] = entry
        torch.accelerator.synchronize(device)
        logger.info("MiniCPM-o Talker: captured seeded codec sampler graphs for 1..%d rows", max_rows)

    def try_sample(
        self,
        logits: torch.Tensor,
        input_batch: InputBatch,
        mask_eos: torch.Tensor | None,
        forced_eos: torch.Tensor | None,
    ) -> SamplerOutput | None:
        """The graph result, or ``None`` when this step must take the eager path."""
        num_rows = int(logits.shape[0])
        entry = self.graphs.get(num_rows)
        if entry is None or mask_eos is None or forced_eos is None:
            return None
        if (
            logits.dtype != torch.float32
            or logits.shape[1] != self.vocab_size
            or not logits.is_contiguous()
            or int(mask_eos.shape[0]) != num_rows
            or int(forced_eos.shape[0]) != num_rows
            or int(input_batch.expanded_idx_mapping.shape[0]) != num_rows
        ):
            return None
        sampler = self.sampler
        base = sampler.base_sampler
        processors = base.logits_processors
        if len(processors) != len(self._processors) or any(
            live is not captured for live, captured in zip(processors, self._processors)
        ):
            # The pipeline changed after capture (e.g. a custom processor was added).
            return None
        idx_mapping_np = input_batch.idx_mapping_np
        if not self.slot_ok[idx_mapping_np].all():
            return None
        rows, accepted = sampler._seeded_rows(logits, input_batch)
        if len(rows) != num_rows or len(accepted) != num_rows:
            return None
        if base.get_logprobs_dims(idx_mapping_np) is not None:
            return None
        from vllm.v1.worker.gpu.input_batch import get_num_sampled_and_rejected

        from vllm_omni.utils.seeded_exponential import claim_exponential_draws

        device = logits.device
        generators = [sampler._request_generator(int(slot), device) for slot in idx_mapping_np]
        seeds, offsets, _ = claim_exponential_draws(generators, self.vocab_size, device)
        entry.logits.copy_(logits)
        entry.expanded_idx_mapping.copy_(input_batch.expanded_idx_mapping)
        entry.mask_eos.copy_(mask_eos)
        entry.forced_eos.copy_(forced_eos)
        if np.any(self._logit_bias.num_stop_token_ids.np[idx_mapping_np] > 0):
            # Sampler.__call__'s ``pos``; only the min_tokens branch reads it.
            entry.pos.copy_(input_batch.positions[input_batch.logits_indices])
        entry.rng_state.copy_(torch.tensor([seeds, offsets], dtype=torch.int64, pin_memory=True), non_blocking=True)
        self._refresh_tables()
        assert entry.graph is not None and entry.sampled is not None
        entry.graph.replay()
        # The next replay rewrites the graph output; the runner's async copy must not see that.
        sampled = entry.sampled.clone()
        num_sampled, num_rejected = get_num_sampled_and_rejected(
            input_batch.seq_lens.new_ones(input_batch.num_reqs),
            input_batch.seq_lens,
            input_batch.cu_num_logits,
            input_batch.idx_mapping,
            sampler.req_states.prefill_len.gpu,
        )
        return SamplerOutput(
            sampled_token_ids=sampled.view(-1, 1),
            logprobs_tensors=None,
            num_nans=None,
            num_sampled=num_sampled,
            num_rejected=num_rejected,
        )


class MiniCPMO45DuplexSampler(OmniSampler):
    """Keep listen/speak policy outside transformer graph capture.

    The shared policy owns boundary/candidate draws and text-based
    punctuation cuts. Reuse it, including its request-local RNG, while the
    upstream sampler owns counts and device token-state updates. Partial
    prefills never advance policy state.
    """

    def __init__(self, base_sampler: Sampler, model: MiniCPMO45OmniForConditionalGeneration) -> None:
        super().__init__(base_sampler)
        self.model = model
        self._requests: dict[str, tuple[int | None, list[int], torch.Generator | None]] = {}
        self._generators: dict[str, torch.Generator] = {}
        self._empty = torch.empty(0)
        self._pending_history: tuple[_MiniCPMO45PendingSamples, list[tuple[str, list[int], int]]] | None = None

    @torch.inference_mode()
    def warmup(self) -> None:
        """Prime the upstream Gumbel draw that commits each duplex decision.

        Every duplex step ends in ``Sampler._sample_random`` -> ``gumbel_sample``
        with ``apply_temperature=False``, no logits cache and the sampler's
        ``use_fp64_gumbel``. Its logits are the raw head output (head dtype)
        when no admitted request needs logits processing, otherwise the float32
        copy made by ``apply_sampling_params``; prime both. Seeds and
        temperatures come from disposable tables: the draw hashes (seed, pos)
        and never touches request or default generators.
        """
        device = _cuda_sampler_device(self)
        if device is None:
            return
        from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample

        model_config = getattr(getattr(self.model, "vllm_config", None), "model_config", None)
        head_dtype = getattr(model_config, "head_dtype", None) or getattr(model_config, "dtype", None)
        dtypes = dict.fromkeys(
            dtype for dtype in (head_dtype, torch.float32) if isinstance(dtype, torch.dtype) and dtype.is_floating_point
        )
        max_num_reqs = int(self.req_states.max_num_reqs)
        vocab_size = int(self.sampling_states.vocab_size)
        expanded_idx_mapping = torch.zeros(1, dtype=torch.int32, device=device)
        temperature = torch.zeros(max_num_reqs, dtype=torch.float32, device=device)
        seeds = torch.zeros(max_num_reqs, dtype=torch.int64, device=device)
        pos = torch.zeros(1, dtype=torch.int64, device=device)
        # Live logits are a view of the vocab-padded head output (row stride
        # rounded up to 64), so Triton specializes on that stride's 16-byte
        # divisibility; prime both the contiguous and the padded layouts.
        padded = -(-vocab_size // 64) * 64
        layouts = [(dtype, row_stride) for dtype in dtypes for row_stride in dict.fromkeys((vocab_size, padded))]
        for dtype, row_stride in layouts:
            gumbel_sample(
                torch.zeros((1, row_stride), dtype=dtype, device=device)[:, :vocab_size],
                expanded_idx_mapping,
                temperature,
                seeds,
                pos,
                apply_temperature=False,
                is_drafting=False,
                use_fp64=bool(getattr(self.base_sampler, "use_fp64_gumbel", False)),
            )

    def on_requests_finished(self, request_ids: set[str] | list[str]) -> None:
        for request_id in request_ids:
            self._generators.pop(request_id, None)
            self._requests.pop(request_id, None)

    def _finish_deferred_history(self) -> None:
        pending_history, self._pending_history = self._pending_history, None
        if pending_history is None:
            return
        pending, entries = pending_history
        # Reuse the shared commit's existing pinned copy and RNG rewinds.
        # Preprocess may already have committed the session latches.
        self.model._commit_minicpmo45_duplex_pending_samples()
        if pending.event is not None:
            pending.event.synchronize()
        tokens = pending.host[:, 0].tolist()
        for request_id, history, position in entries:
            current = self._requests.get(request_id)
            if current is not None and current[1] is history:
                history.append(int(tokens[position]))

    def __call__(self, logits: torch.Tensor, input_batch: InputBatch) -> SamplerOutput:
        self._finish_deferred_history()
        infos = getattr(self.model, "_mrv2_duplex_infos", ())
        rows = []
        histories = [[] for _ in range(input_batch.num_reqs)]
        generators = {}
        temperatures, top_ks, top_ps = [], [], []
        for i, info in enumerate(infos):
            params = info.get("sampling_params")
            temperatures.append(float(getattr(params, "temperature", 0.7)))
            top_ks.append(int(getattr(params, "top_k", 100)))
            top_ps.append(float(getattr(params, "top_p", 0.8)))
            duplex = info.get("duplex")
            if not isinstance(duplex, dict) or duplex.get("data_plane") is not True:
                continue
            # Kernel warmup registers requests with logprobs but no duplex
            # policy. Real duplex output diagnostics cannot describe the
            # two-pass boundary/punctuation policy with a one-hot logit row.
            if getattr(params, "logprobs", None) is not None or getattr(params, "logprob_token_ids", None) is not None:
                raise ValueError("MiniCPM-o MRv2 duplex sampling does not support output logprobs")
            # Includes only final prefill chunks and decode rows.
            if (
                input_batch.num_computed_prefill_tokens_np[i] + input_batch.num_scheduled_tokens[i]
                < input_batch.prefill_len_np[i]
            ):
                continue
            request_id, seq = info["req_id"], duplex.get("seq")
            state = self._requests.get(request_id)
            if state is None or state[0] != seq:
                generator = None
                seed = getattr(params, "seed", None)
                if seed is not None:
                    # V1 keeps the request's generator when a streaming append
                    # resets segment history. Slot reuse/preemption must not
                    # restart that random stream either.
                    generator = self._generators.get(info["req_id"])
                    if generator is None:
                        generator = torch.Generator(device=logits.device).manual_seed(seed)
                        self._generators[info["req_id"]] = generator
                # A preempted request can return in a different GPU slot.
                # Its accepted history survives; only a new segment resets it.
                state = (seq, [], generator)
                self._requests[request_id] = state
            histories[i] = state[1]
            if state[2] is not None:
                generators[i] = state[2]
            rows.append(
                DuplexSamplingRow(
                    row_idx=i,
                    request_id=info["req_id"],
                    session_id=duplex.get("session_id"),
                    seq=duplex.get("seq"),
                    payload=duplex.get("payload"),
                    max_tokens=getattr(params, "max_tokens", None),
                    temperature=temperatures[i],
                    top_k=top_ks[i],
                    top_p=top_ps[i],
                )
            )
        if not rows:
            return self.base_sampler(logits, input_batch)
        metadata = SamplingMetadata(
            output_token_ids=histories,
            generators=generators,
            temperature=torch.tensor(temperatures),
            top_k=torch.tensor(top_ks),
            top_p=torch.tensor(top_ps),
            all_greedy=all(temperatures[row.row_idx] == 0 for row in rows),
            all_random=all(temperatures[row.row_idx] > 0 for row in rows),
            max_num_logprobs=None,
            no_penalties=True,
            prompt_token_ids=None,
            frequency_penalties=self._empty,
            presence_penalties=self._empty,
            repetition_penalties=self._empty,
            allowed_token_ids_mask=None,
            bad_words_token_ids={},
            logitsprocs=LogitsProcessors(),
        )
        self.model.prepare_duplex_sampling(logits, metadata, tuple(rows))
        token_ids = self.model._minicpmo45_native_duplex_token_ids()
        terminators = self.model._minicpmo45_chunk_terminator_token_ids(token_ids)
        fresh_rows = [
            row for row in rows if not histories[row.row_idx] or histories[row.row_idx][-1] not in terminators
        ]
        sessions = [row.session_id for row in fresh_rows if row.session_id is not None]
        batch_sampled = {}
        # Shared session state requires row-order execution. Distinct sessions
        # can read each boundary/text draw together without changing the policy.
        row_params = list(zip(temperatures, top_ks, top_ps, strict=True))
        if fresh_rows and len(sessions) == len(set(sessions)):
            indices = [row.row_idx for row in fresh_rows]
            deferred = self.model._sample_minicpmo45_native_duplex_rows_deferred(
                logits, metadata, row_idxs=indices, token_ids=token_ids, row_params=row_params
            )
            if deferred is not None:
                pending = self.model._minicpmo45_duplex_pending_samples
                self._pending_history = (
                    pending,
                    [(row.request_id, histories[row.row_idx], pos) for pos, row in enumerate(fresh_rows)],
                )
                native_indices = [row.row_idx for row in rows]
                selected = index_to_device(
                    [histories[row.row_idx][-1] if histories[row.row_idx] else 0 for row in rows], logits.device
                )
                fresh_indices = set(indices)
                positions = index_to_device(
                    [pos for pos, row in enumerate(rows) if row.row_idx in fresh_indices], logits.device
                )
                selected.index_copy_(0, positions, deferred)
                native_indices_t = index_to_device(native_indices, logits.device)
                logits.index_fill_(0, native_indices_t, float("-inf"))
                # A Python scalar would be copied from pageable host memory,
                # which synchronizes the stream and defeats async scheduling.
                logits.index_put_((native_indices_t, selected), logits.new_zeros(()))
                return self.base_sampler(logits, input_batch)
            sampled_rows = self.model._sample_minicpmo45_native_duplex_rows(
                logits, metadata, row_idxs=indices, token_ids=token_ids, row_params=row_params
            )
            batch_sampled = dict(zip(indices, sampled_rows, strict=True))
        for row in rows:
            history = histories[row.row_idx]
            if history and history[-1] in terminators:
                sampled = history[-1]
            else:
                sampled = (
                    batch_sampled[row.row_idx]
                    if row.row_idx in batch_sampled
                    else self.model._sample_minicpmo45_native_duplex_rows(
                        logits,
                        metadata,
                        row_idxs=[row.row_idx],
                        token_ids=token_ids,
                        row_params=row_params,
                    )[0]
                )
                self.model._record_minicpmo45_duplex_terminator(row.row_idx, sampled, token_ids)
                history.append(sampled)
            # Upstream preserves partial-prefill counts, sampling diagnostics
            # and token-state writes. A determined duplex row has one outcome.
            logits[row.row_idx].fill_(float("-inf"))
            logits[row.row_idx, sampled].fill_(0)
        return self.base_sampler(logits, input_batch)
