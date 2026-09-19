# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Checkpoint-loading contract for the SeedVR2 validation loader.

The released checkpoint stores the RoPE table one module deeper than this port
registers it, so the loader normalizes exactly that suffix and must then fail on
any remaining missing / unexpected / mismatched key instead of running inference.
These tests drive the real loader with a tiny fixture model.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tests.helpers.mark import hardware_marks

pytestmark = [
    pytest.mark.diffusion,
    pytest.mark.sp,
    pytest.mark.core_model,
    pytest.mark.cpu,
    *hardware_marks(res={"cuda": "L4"}, num_cards=1),
]

WORKER_PATH = Path(__file__).with_name("window_sp_worker.py")


def _worker():
    name = "seedvr2_window_sp_worker_contract"
    spec = importlib.util.spec_from_file_location(name, WORKER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolve the defining module through sys.modules.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


worker = _worker()
ROPE_SUFFIX = worker.ROPE_BUFFER_KEY_SUFFIX
ROPE_NORMALIZED = worker.ROPE_BUFFER_KEY_NORMALIZED


class _TinyBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = nn.Module()
        self.attn.rope = nn.Module()
        self.attn.rope.register_buffer("freqs", torch.arange(3, dtype=torch.float32))
        self.proj = nn.Linear(2, 2)


class _TinyPort(nn.Module):
    """Stands in for ``SeedVR2NaDiT`` with the same key shape."""

    def __init__(self, num_layers: int = 2) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([_TinyBlock() for _ in range(num_layers)])
        self.vid_in = nn.Module()
        self.vid_in.proj = nn.Linear(2, 2)


def _reference_state(num_layers: int = 2) -> dict:
    """The tiny model's weights with the released checkpoint's RoPE nesting."""
    state = {}
    for index in range(num_layers):
        state[f"blocks.{index}.attn.rope.rope.freqs"] = torch.arange(3, dtype=torch.float32)
        state[f"blocks.{index}.proj.weight"] = torch.eye(2)
        state[f"blocks.{index}.proj.bias"] = torch.zeros(2)
    state["vid_in.proj.weight"] = torch.eye(2)
    state["vid_in.proj.bias"] = torch.zeros(2)
    return state


def _args(**overrides) -> SimpleNamespace:
    values = {
        "ckpt": "unused",
        "num_layers": 2,
        "varlen": False,
        "allow_truncated_layers": True,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _load(state: dict, args: SimpleNamespace | None = None):
    args = args or _args()
    return worker._load_port_model(
        args,
        torch.device("cpu"),
        torch.float32,
        model_factory=lambda: _TinyPort(num_layers=args.num_layers),
        state_loader=lambda path: state,
    )


def test_reference_rope_suffix_is_normalized():
    normalized, stats = worker.normalize_reference_state_dict(_reference_state())
    assert all(not key.endswith(ROPE_SUFFIX) for key in normalized)
    assert any(key.endswith(ROPE_NORMALIZED) for key in normalized)
    assert stats["rope_buffer_keys_normalized"] == 2
    assert ROPE_SUFFIX == ".rope.rope.freqs" and ROPE_NORMALIZED == ".rope.freqs"


def test_already_normalized_keys_are_left_alone():
    normalized, _ = worker.normalize_reference_state_dict(_reference_state())
    again, stats = worker.normalize_reference_state_dict(normalized)
    assert again == normalized
    assert stats["rope_buffer_keys_normalized"] == 0


def test_duplicate_source_keys_are_rejected():
    state = _reference_state()
    state["blocks.0.attn.rope.freqs"] = torch.ones(3)
    with pytest.raises(RuntimeError, match="collision"):
        worker.normalize_reference_state_dict(state)


def test_real_loader_accepts_the_reference_layout():
    model, missing, unexpected, stats = _load(_reference_state())
    assert missing == [] and unexpected == []
    assert stats["rope_buffer_keys_normalized"] == 2
    assert isinstance(model, _TinyPort)


def test_loader_rejects_missing_keys():
    state = _reference_state()
    del state["blocks.1.proj.weight"]
    with pytest.raises(RuntimeError, match="missing key"):
        _load(state)


def test_loader_rejects_unexpected_keys():
    state = _reference_state()
    state["blocks.0.not_a_parameter"] = torch.zeros(1)
    with pytest.raises(RuntimeError, match="unexpected key"):
        _load(state)


def test_loader_rejects_shape_mismatch():
    state = _reference_state()
    state["vid_in.proj.weight"] = torch.zeros(3, 3)
    with pytest.raises(RuntimeError):
        _load(state)


def test_full_depth_is_required_unless_truncation_is_explicit():
    state = _reference_state(num_layers=2)
    with pytest.raises(RuntimeError, match="full-depth"):
        _load(state, _args(allow_truncated_layers=False))
    model, missing, unexpected, stats = _load(state, _args(num_layers=1, allow_truncated_layers=True))
    assert missing == [] and unexpected == []
    assert stats["truncated_block_keys"] == 3, "the dropped block's buffer + weight + bias must be reported"


def test_worker_failure_exits_without_entering_another_collective(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["window_sp_worker", "--case", "transport", "--report-dir", "/unused"])
    monkeypatch.setattr(worker.dist, "init_process_group", lambda **kwargs: None)

    def fail(*args):
        raise RuntimeError("worker failed")

    def unexpected_barrier():
        pytest.fail("failed worker entered a collective")

    monkeypatch.setattr(worker, "run_transport", fail)
    monkeypatch.setattr(worker.dist, "barrier", unexpected_barrier)
    with pytest.raises(RuntimeError, match="worker failed"):
        worker.main()
