# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Request-owned decode channels captured with the existing talker forward.

Host bookkeeping binds current token rows to persistent request slots. Tensor
values, delay transitions and feedback stay on the model device. An optional
live-request snapshot retains unscheduled leases across prefill-only steps.
Without that snapshot, the original current-decode-batch lease policy applies.
"""

from __future__ import annotations

import os
from collections import Counter
from typing import Any

import torch


class DeviceDecodeState:
    def __init__(self, model: Any, capacity: int, text_capacity: int, max_tokens: int) -> None:
        self.capacity = capacity
        self.text_capacity = text_capacity
        self.device = model._dec_audio_codes.device
        self.phoneme_width = model.arch.phoneme_stacking_factor if model.has_phoneme else 0
        self.phoneme_delay = model.phonemes_delay
        self.speech_delay = model.speech_delay
        self.audio_bos = model.arch.audio_bos_id
        self.phoneme_bos = getattr(model, "phoneme_bos_id", 0)
        self.phoneme_eos = getattr(model, "phoneme_eos_id", -1)
        # Every inactive token row has a distinct sink, so captured index_copy_
        # writes remain deterministic even when a graph pads the live batch.
        slots = capacity + max_tokens
        self.offset = torch.zeros(slots, dtype=torch.long, device=self.device)
        self.text = torch.zeros((capacity + 1, text_capacity), dtype=torch.long, device=self.device)
        self.text_length = torch.zeros_like(self.offset)
        self.audio = torch.zeros((slots, model.num_codebooks), dtype=torch.long, device=self.device)
        self.phoneme = torch.zeros((slots, self.phoneme_width), dtype=torch.long, device=self.device)
        self.has_audio = torch.zeros(slots, dtype=torch.bool, device=self.device)
        self.has_phoneme = torch.zeros_like(self.has_audio)
        self.ended = torch.zeros_like(self.has_audio)
        self.row_indices = torch.arange(max_tokens, device=self.device)
        self.inactive_slots = self.row_indices + capacity
        self.row_slots = self.inactive_slots.clone()
        self.leases: dict[Any, dict[str, Any]] = {}
        self.row_slots_host: dict[int, int] = {}
        self._routing_pool = None
        if os.environ.get("EASYMAGPIE_DDS_ROUTING_POOL", "0") == "1":
            from .dds_routing_pool import RoutingUploadPool

            self._routing_pool = RoutingUploadPool(self.device, capacity)
        self.lease_counters: Counter = Counter(
            dict.fromkeys(
                (
                    "prepare_calls",
                    "live_snapshot_calls",
                    "seeds",
                    "reused",
                    "evictions",
                    "dead_cleanup",
                    "ineligible_evictions",
                    "capacity_evictions",
                    "capacity_fallbacks",
                    "retained_unscheduled",
                    "not_live_fallbacks",
                    "peak_bound_count",
                ),
                0,
            )
        )

    def prepare(self, rows, request_keys, model, merge_text, *, live_request_keys=None):
        """Seed admissions/resumes; returning metadata preserves Omni's contract."""
        request_keys = tuple(request_keys)
        live = None if live_request_keys is None else set(live_request_keys)
        self.lease_counters["prepare_calls"] += 1
        self.row_slots.copy_(self.inactive_slots)
        self.row_slots_host.clear()
        eligible = []
        for (row, info), key in zip(rows, request_keys, strict=True):
            if live is not None and key not in live:
                self.lease_counters["not_live_fallbacks"] += 1
                continue
            if not self._eligible(info, model):
                continue
            text = info.get("text_tokens") or []
            incoming = info.get("text_token") or []
            appended = False
            if incoming:
                text, appended = merge_text(text, incoming, info.get("text_token_start"))
            if len(text) > self.text_capacity:
                continue
            eligible.append((row, info, key, tuple(text), appended))

        scheduled = set(request_keys)
        eligible_keys = {entry[2] for entry in eligible}
        keep = eligible_keys if live is None else live - (scheduled - eligible_keys)
        removed = set(self.leases) - keep
        dead = set() if live is None else removed - live
        self.lease_counters["dead_cleanup"] += len(dead)
        self.lease_counters["ineligible_evictions"] += len(removed & (scheduled - eligible_keys) - dead)
        self.lease_counters["evictions"] += len(removed)
        self.leases = {key: lease for key, lease in self.leases.items() if key in keep}
        if live is not None:
            self.lease_counters["live_snapshot_calls"] += 1
            self.lease_counters["retained_unscheduled"] += len(set(self.leases) - scheduled)
        used = {lease["slot"] for lease in self.leases.values()}
        free = iter(slot for slot in range(self.capacity) if slot not in used)
        # Only unscheduled leases may be recycled for a current admission.
        # Their request-owned Omni feedback remains intact and seeds a resume.
        victims = iter([key for key in self.leases if key not in eligible_keys])
        updates = {}
        for row, info, key, text, appended in eligible:
            offset = int(info.get("decode_offset", 0) or 0)
            lease = self.leases.get(key)
            fresh = lease is None
            if fresh:
                slot = next(free, None)
                if slot is None and live is not None:
                    victim = next(victims, None)
                    if victim is not None:
                        slot = self.leases.pop(victim)["slot"]
                        self.lease_counters["capacity_evictions"] += 1
                        self.lease_counters["evictions"] += 1
                if slot is None:
                    self.lease_counters["capacity_fallbacks"] += 1
                    continue
                lease = {"slot": slot, "expected_offset": offset, "text": ()}
                self.leases[key] = lease
            slot = lease["slot"]
            previous_text = lease["text"]
            appended_only = text[: len(previous_text)] == previous_text
            if fresh or offset != lease["expected_offset"] or not appended_only:
                self._seed(slot, offset, text, info)
                self.lease_counters["seeds"] += 1
            elif text != previous_text:
                self.lease_counters["reused"] += 1
                start = len(previous_text)
                self.text[slot, start : len(text)].copy_(torch.tensor(text[start:], device=self.device))
                self.text_length[slot] = len(text)
            else:
                self.lease_counters["reused"] += 1
            lease.update(expected_offset=offset + 1, text=text)
            self.row_slots_host[row] = slot
            update = {"decode_offset": offset + 1}
            if appended:
                update["text_tokens"] = list(text)
            # Sticky EOS is committed inside forward and returned by postprocess,
            # after it is ready; preprocessing must not clone its old value.
            updates[row] = update
        if self.row_slots_host:
            uploaded = self._routing_pool is not None and self._routing_pool.try_upload(
                self.row_slots, self.row_slots_host
            )
            if not uploaded:
                indices = torch.tensor(list(self.row_slots_host), device=self.device, dtype=torch.long)
                slots = torch.tensor(list(self.row_slots_host.values()), device=self.device, dtype=torch.long)
                self.row_slots.index_copy_(0, indices, slots)
        self.lease_counters["peak_bound_count"] = max(self.lease_counters["peak_bound_count"], len(self.leases))
        return updates

    def lease_status(self):
        return {
            "capacity": self.capacity,
            "bound_count": len(self.leases),
            "scheduled_bound_count": len(self.row_slots_host),
            **dict(self.lease_counters),
            "routing_pool": self._routing_pool.status() if self._routing_pool is not None else None,
        }

    def stage(self, model, rows):
        """Refresh only bound rows; keep existing eager/fallback rows intact."""
        slots = self.row_slots[rows]
        live = slots < self.capacity
        step = self.offset[slots]
        text_valid = step < self.text_length[slots]
        column = step.clamp(0, self.text_capacity - 1)
        text = self.text[slots.clamp(max=self.capacity), column]
        old_text = model._dec_text_tokens[rows]
        self._write(model._dec_text_tokens, rows, live & text_valid, text, old_text)
        self._write(model._dec_text_mask, rows, live, text_valid.long())

        audio_active = step >= self.speech_delay
        audio = torch.where(
            ((step == self.speech_delay) | ~self.has_audio[slots]).unsqueeze(1),
            self.audio_bos,
            self.audio[slots],
        )
        self._write(model._dec_audio_codes, rows, live & audio_active, audio)
        self._write(model._dec_audio_valid, rows, live, audio_active.long())

        ended = self.ended[slots]
        if self.phoneme_width:
            has_previous = self.has_phoneme[slots]
            phoneme_active = (step == self.phoneme_delay) | ((step > self.phoneme_delay) & has_previous)
            phoneme = torch.where((step == self.phoneme_delay).unsqueeze(1), self.phoneme_bos, self.phoneme[slots])
            self._write(model._dec_phoneme_tokens, rows, live & phoneme_active, phoneme)
            self._write(model._dec_phoneme_valid, rows, live, (phoneme_active & ~ended).long())
            ended = ended | ((step > self.phoneme_delay) & has_previous & (phoneme == self.phoneme_eos).any(dim=1))
        return rows, slots, step, ended, live

    def commit(self, model, context):
        """Commit predictions before next-step preparation or async output."""
        rows, slots, step, ended, live = context
        self.audio.index_copy_(0, slots, model._out_codes[rows])
        self.has_audio.index_copy_(0, slots, live)
        if self.phoneme_width:
            self.phoneme.index_copy_(0, slots, model._dec_phoneme_tokens[rows])
            self.has_phoneme.index_copy_(0, slots, live)
        self.offset.index_copy_(0, slots, torch.where(live, step + 1, 0))
        self.ended.index_copy_(0, slots, ended & live)

    def ended_for_row(self, row):
        slot = self.row_slots_host.get(row)
        return None if slot is None else self.ended[slot].detach()

    def _eligible(self, info, model):
        if int(info.get("decode_offset", 0) or 0) < 0:
            return False
        if isinstance(info.get("additional_information"), dict) or info.get("user_audio_prefilled"):
            return False
        for key, width in (("last_audio_codes", model.num_codebooks), ("last_phoneme_token", self.phoneme_width)):
            value = info.get(key)
            if (
                width
                and isinstance(value, torch.Tensor)
                and value.numel()
                and (value.device != self.device or value.dtype != torch.long or value.numel() < width)
            ):
                return False
        ended = info.get("phoneme_ended", False)
        return not isinstance(ended, torch.Tensor) or (
            ended.device == self.device and ended.dtype == torch.bool and ended.numel() == 1
        )

    def _seed(self, slot, offset, text, info):
        self.offset[slot] = offset
        self.text_length[slot] = len(text)
        if text:
            self.text[slot, : len(text)].copy_(torch.tensor(text, dtype=torch.long, device=self.device))
        for key, target, present in (
            ("last_audio_codes", self.audio, self.has_audio),
            ("last_phoneme_token", self.phoneme, self.has_phoneme),
        ):
            value = info.get(key)
            available = target.shape[1] > 0 and isinstance(value, torch.Tensor) and value.numel() > 0
            present[slot] = available
            if available:
                target[slot].copy_(value.reshape(-1)[: target.shape[1]])
            else:
                target[slot].zero_()
        self.ended[slot].copy_(torch.as_tensor(info.get("phoneme_ended", False), device=self.device).reshape(()))

    @staticmethod
    def _write(target, rows, enabled, value, previous=None):
        if previous is None:
            previous = target[rows]
        if value.ndim > 1:
            enabled = enabled.unsqueeze(1)
        target.index_copy_(0, rows, torch.where(enabled, value, previous))
