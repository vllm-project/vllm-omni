# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Persistent V1 codec state; CUDA updates never require a host token read."""

import torch
from vllm.triton_utils import tl, triton

from vllm_omni.utils.device_copy import index_to_device, to_device_nonblocking


@triton.jit
def _update_codec_state(
    ids_ptr,
    slots_ptr,
    decode_ptr,
    state_ptr,
    history_ptr,
    out_history_ptr,
    valid_ptr,
    done_ptr,
    masked_ptr,
    eos: tl.constexpr,
    window: tl.constexpr,
    cadence: tl.constexpr,
    boundary_steps: tl.constexpr,
):
    row = tl.program_id(0)
    slot = tl.load(slots_ptr + row)
    code = tl.load(ids_ptr + row)
    decode = tl.load(decode_ptr + row)
    base = state_ptr + slot * 5
    step = tl.load(base)
    finished = tl.load(base + 1) != 0
    limit = tl.load(base + 2)
    minimum = tl.load(base + 3)
    drain = tl.load(base + 4) != 0
    valid = decode & ~finished & (code != eos)
    step += valid.to(tl.int64)
    done = finished | (decode & (code == eos)) | (step >= limit)
    boundary = (step >= cadence) & ((step % cadence) < boundary_steps)
    masked = ~done & ((step < minimum) | (drain & boundary))
    offsets = tl.arange(0, window)
    # Load every old value before overwriting the history in place.
    old = tl.load(history_ptr + slot * window + tl.minimum(offsets + 1, window - 1))
    shifted = tl.where(offsets == window - 1, code, old)
    unchanged = tl.load(history_ptr + slot * window + offsets)
    updated = tl.where(valid, shifted, unchanged)
    tl.store(history_ptr + slot * window + offsets, updated)
    tl.store(out_history_ptr + row * window + offsets, updated)
    tl.store(base, step)
    tl.store(base + 1, done.to(tl.int64))
    tl.store(valid_ptr + row, valid)
    tl.store(done_ptr + row, done)
    tl.store(masked_ptr + row, masked)


@triton.jit
def _window_penalty(
    logits_ptr, histories_ptr, penalties_ptr, out_ptr, vocab: tl.constexpr, window: tl.constexpr, block: tl.constexpr
):
    row = tl.program_id(0)
    tokens = tl.program_id(1) * block + tl.arange(0, block)
    offsets = tl.arange(0, window)
    history = tl.load(histories_ptr + row * window + offsets)
    frequency = tl.sum((history[:, None] == tokens[None, :]).to(tl.int32), axis=0)
    penalty = tl.load(penalties_ptr + row).to(tl.float32)
    alpha = tl.exp(tl.log(penalty) * frequency.to(tl.float32))
    values = tl.load(logits_ptr + row * vocab + tokens, tokens < vocab, other=0).to(tl.float32)
    tl.store(out_ptr + row * vocab + tokens, tl.where(values < 0, values * alpha, values / alpha), tokens < vocab)


def apply_window_penalty(logits, histories, penalties):
    output = torch.empty_like(logits)
    _window_penalty[(logits.shape[0], triton.cdiv(logits.shape[1], 128))](
        logits,
        histories,
        penalties,
        output,
        logits.shape[1],
        histories.shape[1],
        128,
    )
    return output


class CodecState:
    """Stable request slots, including across duplex prefill/recompute resets.

    State columns are frame count, finished, frame limit, minimum frames and
    turn-end drain. Kernels and slot reuse run on the runner's current stream.
    Per-step outputs own their storage so asynchronous payload copies cannot
    observe a subsequent state update.
    """

    def __init__(self, device, vocab_size, capacity=4096, window=16):
        self.device = device
        self.vocab_size = vocab_size
        self.window = window
        self.capacity = capacity
        self.state = torch.empty((capacity, 5), dtype=torch.long, device=device)
        self.history = torch.empty((capacity, window), dtype=torch.long, device=device)
        self.slots = {}
        self._index_cache = {}
        self.free_slots = list(range(capacity - 1, -1, -1))

    def prepare(self, rows):
        new_slots, configs, histories = [], [], []
        for request_id, state in rows:
            if "_gpu_slot" in state:
                continue
            slot = self.slots.get(request_id)
            if slot is None:
                if not self.free_slots:
                    # Preempted requests retain state but no longer count
                    # against max_num_seqs. Grow without reading it on CPU.
                    capacity = max(1, self.capacity * 2)
                    state_storage = self.state.new_empty((capacity, 5))
                    history_storage = self.history.new_empty((capacity, self.window))
                    state_storage[: self.capacity].copy_(self.state)
                    history_storage[: self.capacity].copy_(self.history)
                    self.state, self.history = state_storage, history_storage
                    self.free_slots.extend(range(capacity - 1, self.capacity - 1, -1))
                    self.capacity = capacity
                slot = self.free_slots.pop()
                self.slots[request_id] = slot
            state["_gpu_slot"] = slot
            new_slots.append(slot)
            configs.append(
                [
                    int(state.get("step", 0)),
                    bool(state.get("finished")),
                    int(state.get("max_tokens") or (2**31 - 1)) - 1,
                    int(state.get("min_tokens") or 0),
                    bool(state.get("turn_end_drain")),
                ]
            )
            recent = state.get("recent_codes", [])
            recent = to_device_nonblocking(torch.as_tensor(recent, dtype=torch.long), self.device).reshape(-1)
            padding = torch.full((self.window,), self.vocab_size, device=self.device, dtype=torch.long)
            histories.append(torch.cat([padding, recent])[-self.window :])
        if new_slots:
            slots = index_to_device(new_slots, self.device)
            self.state.index_copy_(0, slots, index_to_device(configs, self.device))
            self.history.index_copy_(0, slots, torch.stack(histories))

    def update(self, rows, ids, *, eos, cadence, boundary):
        self.prepare([(request_id, state) for request_id, state, _ in rows])
        slots, decode = self.indices(
            tuple(state["_gpu_slot"] for _, state, _ in rows),
            tuple(code is not None for _, _, code in rows),
        )
        count = len(rows)
        history = torch.empty((count, self.window), dtype=torch.long, device=self.device)
        valid = torch.empty(count, dtype=torch.bool, device=self.device)
        done = torch.empty_like(valid)
        masked = torch.empty_like(valid)
        _update_codec_state[(count,)](
            ids,
            slots,
            decode,
            self.state,
            self.history,
            history,
            valid,
            done,
            masked,
            eos,
            self.window,
            cadence,
            boundary,
        )
        return history, valid, done, masked

    def indices(self, slots, flags):
        key = (slots, flags)
        cached = self._index_cache.get(key)
        if cached is None:
            if len(self._index_cache) >= 16:
                self._index_cache.clear()
            cached = (index_to_device(slots, self.device), index_to_device(flags, self.device, dtype=torch.bool))
            self._index_cache[key] = cached
        return cached

    def mask_embeddings(self, embeds, slots, rows):
        slot_ids, _ = self.indices(tuple(slots), (True,) * len(slots))
        row_ids, _ = self.indices(tuple(rows), (True,) * len(rows))
        _mask_embeddings[(len(slots),)](
            embeds,
            slot_ids,
            row_ids,
            self.state,
            embeds.shape[1],
            triton.next_power_of_2(embeds.shape[1]),
        )

    def release(self, request_id):
        slot = self.slots.pop(request_id, None)
        if slot is not None:
            self.free_slots.append(slot)


@triton.jit
def _mask_logits(logits_ptr, forced_ptr, masked_ptr, vocab: tl.constexpr, eos: tl.constexpr, block: tl.constexpr):
    row = tl.program_id(0)
    tokens = tl.program_id(1) * block + tl.arange(0, block)
    force = tl.load(forced_ptr + row)
    mask = tl.load(masked_ptr + row)
    values = tl.load(logits_ptr + row * vocab + tokens, tokens < vocab, other=0)
    values = tl.where(force, tl.where(tokens == eos, 0.0, float("-inf")), values)
    values = tl.where(mask & (tokens == eos), float("-inf"), values)
    tl.store(logits_ptr + row * vocab + tokens, values, tokens < vocab)


def mask_logits(logits, forced, masked, eos):
    _mask_logits[(logits.shape[0], triton.cdiv(logits.shape[1], 256))](
        logits,
        forced,
        masked,
        logits.shape[1],
        eos,
        256,
    )
    return logits


@triton.jit
def _mask_embeddings(embeds_ptr, slots_ptr, rows_ptr, state_ptr, hidden_size: tl.constexpr, block: tl.constexpr):
    row = tl.program_id(0)
    slot = tl.load(slots_ptr + row)
    dest = tl.load(rows_ptr + row)
    finished = tl.load(state_ptr + slot * 5 + 1) != 0
    cols = tl.arange(0, block)
    values = tl.load(embeds_ptr + dest * hidden_size + cols, cols < hidden_size, other=0)
    tl.store(embeds_ptr + dest * hidden_size + cols, tl.where(finished, 0.0, values), cols < hidden_size)
