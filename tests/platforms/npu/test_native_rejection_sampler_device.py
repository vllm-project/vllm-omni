# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Real-device checks for the NPU native rejection-sampler restore gate.

``test_native_rejection_sampler.py`` pins the decision matrix with a faked
SoC name. This file runs the same gate against the device it is executed on,
so the two contracts that a stub cannot prove are checked on hardware:

* the SoC probe reads a real, stable device name (an empty read means
  "unknown", which is what the latch must not treat as a decision), and
* the gate reads that name once per call.

The expectations are derived from the probed name instead of a hard-coded
part, so the file stays valid on every 910B (A2) host the lane runs on.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch_npu")

import vllm_omni.platforms.npu.native_rejection_sampler as gate
from tests.helpers.mark import hardware_marks
from vllm_omni.platforms.npu.ascend_warmup_patch import _probe_soc_name
from vllm_omni.platforms.npu.native_rejection_sampler import restore_native_rejection_sampler

pytestmark = [pytest.mark.core_model, pytest.mark.omni, *hardware_marks(res={"npu": "A2"}, num_cards=1)]


def _armed_config(num_speculative_tokens: int = 4):
    from types import SimpleNamespace

    spec = SimpleNamespace(method="ngram", num_speculative_tokens=num_speculative_tokens)
    return SimpleNamespace(speculative_config=spec)


def test_the_device_name_is_readable_and_stable() -> None:
    """A readable, stable name is what lets the gate decide at all."""
    names = [_probe_soc_name() for _ in range(5)]

    assert names[0], "the device name must be readable on a live device"
    assert len(set(names)) == 1, f"the device name must be stable, got {names}"
    assert names[0].lower().startswith("ascend"), f"unexpected device name: {names[0]!r}"


def test_the_gate_decides_from_the_device_name() -> None:
    """On this part the decision follows the probed name, not a hard-coded one."""
    soc_name = _probe_soc_name()

    # A 910B without the Talker K-step keeps the sampler it is measured with.
    assert gate._target_soc(soc_name) is False
    # With the K-step armed it needs the torch-native sampler.
    assert gate._target_soc(soc_name, _armed_config()) is True


def test_the_gate_reads_the_device_name_once_per_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """One read per call decides the target check and the latch."""
    monkeypatch.setattr(gate, "_RESTORED", False)
    reads: list[str] = []
    real_probe = gate._probe_soc_name

    def counting_probe() -> str:
        name = real_probe()
        reads.append(name)
        return name

    monkeypatch.setattr(gate, "_probe_soc_name", counting_probe)

    restore_native_rejection_sampler()

    assert len(reads) == 1


def test_an_unreadable_device_name_does_not_latch(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unreadable name is "unknown", so the gate may still decide later."""
    monkeypatch.setattr(gate, "_RESTORED", False)
    monkeypatch.setattr(gate, "_probe_soc_name", lambda: "")

    restore_native_rejection_sampler()

    assert gate._RESTORED is False

    # The next call sees the real device again and decides.
    monkeypatch.undo()
    monkeypatch.setattr(gate, "_RESTORED", False)
    restore_native_rejection_sampler()

    assert gate._RESTORED is True
