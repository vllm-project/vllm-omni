# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Device-side MRv2 custom sampler for HunyuanImage3 AR forced-token rules.

Ports ``HunyuanImage3ForConditionalGeneration.sample()`` to Model Runner V2
without per-step host synchronization:

- Comprehension mode (engine-level ``_is_comprehension`` or per-request
  ``SamplingParams.extra_args["ar_task_mode"] == "comprehension"``):
  masks generation-only scaffold tokens (``<boi>``, ``<eoi>``, ``<img_size_*>``,
  ``<cfg>``, ``<img>``, ``<timestep>``, ``<joint_img_sep>``, and ratio tokens).
- Generation mode (IT2I / T2I):
  1. Stage-transition forcing (e.g. ``</think>`` -> ``<recaption>``,
     ``</recaption>`` -> ``<answer>``, ``<boi>``, ``<img_size_*>``).
  2. Ratio-token restriction (after ``<img_size_*>``, restrict vocabulary to
     valid ratio tokens and pick greedily).
  3. Forced ``<eos>`` immediately after any ratio token.

Token history is read directly from ``req_states.all_token_ids.gpu`` using a
fixed lookback window equal to the longest forced transition sequence (3 tokens
for HunyuanImage3), so slot reuse, batch reordering, chunked prefill, and
preemption replay are handled statelessly on device with zero D2H sync.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from vllm_omni.worker_v2.omni_sampler import OmniSampler

if TYPE_CHECKING:
    from vllm.sampling_params import SamplingParams
    from vllm.v1.worker.gpu.input_batch import InputBatch
    from vllm.v1.worker.gpu.sample.output import SamplerOutput
    from vllm.v1.worker.gpu.sample.sampler import Sampler


def _is_comprehension_request(sampling_params: Any) -> bool:
    extra_args = getattr(sampling_params, "extra_args", None)
    if not extra_args:
        return False
    return extra_args.get("ar_task_mode") == "comprehension"


class HunyuanImage3MRv2Sampler(OmniSampler):
    """Apply HunyuanImage3 forced-token and comprehension rules in MRv2."""

    omni_static_staged_writes = True

    def __init__(self, base_sampler: Sampler, model: Any) -> None:
        super().__init__(base_sampler)
        self.model = model
        self._is_comprehension_engine = bool(getattr(model, "_is_comprehension", False))
        eos_id = getattr(model, "_eos_token_id", None)
        self._eos_token_id = int(eos_id) if eos_id is not None else -1
        size_id = getattr(model, "_size_token_id", None)
        self._size_token_id = int(size_id) if size_id is not None else -1

        raw_transitions: dict[int, list[int]] = getattr(model, "_stage_transitions", None) or {}
        self._stage_transitions: tuple[tuple[int, tuple[int, ...]], ...] = tuple(
            (int(trigger), tuple(int(tok) for tok in seq)) for trigger, seq in raw_transitions.items() if seq
        )
        self._max_transition_len = max((len(seq) for _, seq in self._stage_transitions), default=0)

        self._ratio_token_ids: tuple[int, ...] = tuple(
            sorted(int(t) for t in (getattr(model, "_all_ratio_ids", None) or ()) if isinstance(t, int) and t >= 0)
        )
        self._blocked_token_ids: tuple[int, ...] = tuple(
            sorted(int(t) for t in (getattr(model, "_blocked_token_ids", None) or ()) if isinstance(t, int) and t >= 0)
        )

        max_num_reqs = int(base_sampler.req_states.max_num_reqs)
        self._is_comprehension_cpu = np.full(max_num_reqs, self._is_comprehension_engine, dtype=bool)
        self._is_comprehension_gpu: torch.Tensor | None = None
        self._dirty = True
        self._device_tensors: dict[torch.device, tuple[torch.Tensor | None, torch.Tensor | None]] = {}

    def _get_device_tensors(self, device: torch.device) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        cached = self._device_tensors.get(device)
        if cached is not None:
            return cached
        blocked = (
            torch.tensor(self._blocked_token_ids, dtype=torch.long, device=device) if self._blocked_token_ids else None
        )
        ratio = torch.tensor(self._ratio_token_ids, dtype=torch.long, device=device) if self._ratio_token_ids else None
        cached = (blocked, ratio)
        self._device_tensors[device] = cached
        return cached

    def _sync_comprehension_mask(self, device: torch.device) -> torch.Tensor:
        if self._is_comprehension_gpu is None or self._is_comprehension_gpu.device != device:
            self._is_comprehension_gpu = torch.empty(
                self._is_comprehension_cpu.shape[0], dtype=torch.bool, device=device
            )
            self._dirty = True
        if self._dirty:
            src = torch.from_numpy(self._is_comprehension_cpu)
            self._is_comprehension_gpu.copy_(src, non_blocking=device.type == "cuda")
            self._dirty = False
        return self._is_comprehension_gpu

    def add_request(self, req_idx: int, sampling_params: SamplingParams) -> None:
        self.base_sampler.add_request(req_idx, sampling_params)
        is_comp = True if self._is_comprehension_engine else _is_comprehension_request(sampling_params)
        self._is_comprehension_cpu[req_idx] = is_comp
        self._dirty = True

    def apply_staged_writes(self) -> None:
        self.base_sampler.apply_staged_writes()
        if self._dirty and not self._is_comprehension_engine:
            device = getattr(self.base_sampler.req_states, "device", None)
            if device is not None:
                self._sync_comprehension_mask(device)

    def _compute_forced_tokens(
        self,
        logits: torch.Tensor,
        slots: torch.Tensor,
        end_lens: torch.Tensor,
        prompt_lens: torch.Tensor,
        all_token_ids: torch.Tensor,
        is_gen: torch.Tensor,
        ratio_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        """Compute per-row forced target token ID on device (-1 = no forcing).

        Matches ``HunyuanImage3ForConditionalGeneration._get_forced_token``,
        ``_apply_ratio_restriction``, and ratio->EOS forcing with zero host sync.
        """
        n = slots.shape[0]
        device = logits.device
        if all_token_ids.shape[1] == 0:
            return torch.full((n,), -1, dtype=torch.long, device=device)
        num_gen = (end_lens - prompt_lens).clamp(min=0)
        max_col = all_token_ids.shape[1] - 1

        lookback = max(self._max_transition_len, 1)
        # recent[k] (0-indexed) holds token at offset -(k+1), i.e. t[-1], t[-2], ...
        recent: list[torch.Tensor] = []
        for offset in range(1, lookback + 1):
            valid = num_gen >= offset
            col = (end_lens - offset).clamp(min=0, max=max_col)
            tok = all_token_ids[slots, col].long()
            recent.append(torch.where(valid, tok, -1))

        forced = torch.full((n,), -1, dtype=torch.long, device=device)

        if self._stage_transitions:
            # Scan backwards (offset 1, 2, ..., max_transition_len). The most
            # recent trigger token determines whether a transition is active or
            # completed/diverged; earlier triggers are shadowed once resolved.
            unresolved = is_gen
            for offset in range(1, self._max_transition_len + 1):
                tok_at_offset = recent[offset - 1]
                hit_any_trigger = torch.zeros(n, dtype=torch.bool, device=device)
                for trigger_id, seq in self._stage_transitions:
                    is_trigger = unresolved & (tok_at_offset == trigger_id)
                    hit_any_trigger |= is_trigger
                    if offset <= len(seq):
                        # Tokens emitted after trigger are recent[offset-2], ..., recent[0],
                        # corresponding to seq[0], ..., seq[offset-2].
                        matched = is_trigger
                        for step_idx in range(offset - 1):
                            emitted_tok = recent[offset - 2 - step_idx]
                            matched = matched & (emitted_tok == seq[step_idx])
                        forced = torch.where(matched, seq[offset - 1], forced)
                unresolved = unresolved & ~hit_any_trigger

        t1 = recent[0]
        no_transition = is_gen & (forced < 0)

        if ratio_ids is not None and ratio_ids.numel() > 0:
            # After <img_size_*>, restrict to ratio tokens and select greedily.
            if self._size_token_id >= 0:
                need_ratio = no_transition & (t1 == self._size_token_id)
                ratio_logits = logits.index_select(-1, ratio_ids)
                best_ratio_pos = ratio_logits.argmax(dim=-1)
                best_ratio_tok = ratio_ids.index_select(0, best_ratio_pos)
                forced = torch.where(need_ratio, best_ratio_tok, forced)
            else:
                need_ratio = torch.zeros(n, dtype=torch.bool, device=device)

            # After any ratio token, force EOS.
            if self._eos_token_id >= 0:
                is_ratio_t1 = (t1.unsqueeze(-1) == ratio_ids.unsqueeze(0)).any(dim=-1)
                need_eos = no_transition & (~need_ratio) & is_ratio_t1
                forced = torch.where(need_eos, self._eos_token_id, forced)

        return forced

    def _apply_rules(self, logits: torch.Tensor, input_batch: Any) -> None:
        n = int(input_batch.num_reqs)
        if n <= 0:
            return
        active_logits = logits[:n]
        device = active_logits.device
        min_score = torch.finfo(active_logits.dtype).min
        blocked_ids, ratio_ids = self._get_device_tensors(device)

        if self._is_comprehension_engine:
            if blocked_ids is not None:
                active_logits.index_fill_(-1, blocked_ids, min_score)
            return

        slots = input_batch.idx_mapping[:n].to(device=device, dtype=torch.long)
        comp_table = self._sync_comprehension_mask(device)
        is_comp = comp_table.index_select(0, slots)
        is_gen = ~is_comp

        # Comprehension rows: mask DiT scaffold tokens without touching generation rows.
        if blocked_ids is not None:
            blocked_vals = active_logits.index_select(-1, blocked_ids)
            masked_vals = torch.where(is_comp.unsqueeze(-1), min_score, blocked_vals)
            active_logits.index_copy_(-1, blocked_ids, masked_vals)

        req_states = self.base_sampler.req_states
        all_token_ids = req_states.all_token_ids.gpu
        prompt_lens = req_states.prompt_len.gpu.index_select(0, slots).long()
        seq_lens = getattr(input_batch, "seq_lens", None)
        if seq_lens is not None:
            end_lens = seq_lens[:n].to(device=device, dtype=torch.long)
        else:
            end_lens = req_states.total_len.gpu.index_select(0, slots).long()

        forced_target = self._compute_forced_tokens(
            logits=active_logits,
            slots=slots,
            end_lens=end_lens,
            prompt_lens=prompt_lens,
            all_token_ids=all_token_ids,
            is_gen=is_gen,
            ratio_ids=ratio_ids,
        )

        # Apply forced targets in a single zero-sync vectorized scatter.
        has_forced = is_gen & (forced_target >= 0)
        safe_target = forced_target.clamp(min=0, max=active_logits.shape[-1] - 1).unsqueeze(-1)
        orig_at_target = active_logits.gather(-1, safe_target)
        active_logits.masked_fill_(has_forced.unsqueeze(-1), min_score)
        new_at_target = torch.where(has_forced.unsqueeze(-1), active_logits.new_zeros((n, 1)), orig_at_target)
        active_logits.scatter_(-1, safe_target, new_at_target)

    def __call__(self, logits: torch.Tensor, input_batch: InputBatch) -> SamplerOutput:
        if logits is None or logits.numel() == 0 or int(input_batch.num_reqs) == 0:
            return self.base_sampler(logits, input_batch)
        if getattr(input_batch, "num_draft_tokens", 0):
            raise NotImplementedError("HunyuanImage3 MRv2 sampling does not support speculative decoding")
        self._apply_rules(logits, input_batch)
        return self.base_sampler(logits, input_batch)
