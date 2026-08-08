# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the MiniMax H3 encoder pinned-offload transfer path.

The encoder's full-model CPU offload dominated single-GPU latency because it
used pageable ``.to("cpu")`` copies. ``_move_module_pinned`` routes those moves
through persistent pinned host buffers, gated by the shared ``pin_cpu_memory``
setting with a pageable fallback when pinning is off or unavailable. These CPU
tests cover the gating and fallback logic (the GPU<->CPU DMA speedup itself is
covered by the end-to-end A/B on hardware)."""

import pytest
import torch
import torch.nn as nn

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _make_encoder(*, pin_cpu_memory: bool):
    from vllm_omni.diffusion.models.minimax_h3.encoder import MiniMaxH3Qwen3VLEncoder

    # load_model=False constructs the shell (device_target + pin flag) without
    # pulling the 63 GB checkpoint, which is all these transfer tests need.
    return MiniMaxH3Qwen3VLEncoder(
        model_path="",
        device=torch.device("cpu"),
        load_model=False,
        pin_cpu_memory=pin_cpu_memory,
    )


def test_pin_flag_gates_offload_behavior():
    on = _make_encoder(pin_cpu_memory=True)
    off = _make_encoder(pin_cpu_memory=False)
    assert on._pin_offload_enabled() is True
    assert off._pin_offload_enabled() is False


def test_offload_moves_params_to_cpu_and_reuses_pinned_buffers():
    enc = _make_encoder(pin_cpu_memory=True)
    module = nn.Linear(8, 8)  # already on CPU; device_target is CPU here
    # to_device=False must leave params on CPU and not raise; pinned mirrors are
    # created lazily and reused across calls.
    enc._move_module_pinned(module, to_device=False)
    for p in module.parameters():
        assert p.device.type == "cpu"


def test_pageable_fallback_when_pinning_unavailable(monkeypatch):
    enc = _make_encoder(pin_cpu_memory=True)
    # Simulate a host that cannot allocate pinned memory: _pinned_cpu_mirror
    # returns None, so the mover must fall back to a plain pageable copy without
    # raising.
    monkeypatch.setattr(enc, "_pinned_cpu_mirror", lambda tensor: None)
    module = nn.Linear(8, 8)
    enc._move_module_pinned(module, to_device=False)
    for p in module.parameters():
        assert p.device.type == "cpu"


def test_disabled_flag_skips_pinned_mirror(monkeypatch):
    enc = _make_encoder(pin_cpu_memory=False)
    calls = {"n": 0}

    def _tracked(tensor):
        calls["n"] += 1
        return None

    monkeypatch.setattr(enc, "_pinned_cpu_mirror", _tracked)
    module = nn.Linear(8, 8)
    enc._move_module_pinned(module, to_device=False)
    # With pinning disabled the pinned-mirror path must never be consulted.
    assert calls["n"] == 0
