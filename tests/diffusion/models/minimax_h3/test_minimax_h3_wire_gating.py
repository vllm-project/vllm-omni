# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Gating and fallback contracts for the H3 int8 wire modules.

The safety argument for these modules is "off by default, and a failure falls back to
the original bf16 path, so a boot with the arms off is byte-identical". That argument
is only worth anything if it is enforced, so these tests pin it:

- the arm comes only from its own environment variable, and the control-file environment
  variables these modules used to read are inert;
- which mode strings actually enable int8 (the a2a module also accepts the fused spellings;
  the all-reduce module accepts exactly ``int8``);
- a failure **latches** the fallback, so a half-broken run cannot silently keep using a wire
  that already raised;
- the disabled path calls the caller's own all-reduce instead of reimplementing it;
- the intermediate-buffer pool defaults to **off**. The pooled variant corrupts
  the payload while reporting success (deterministic 26,410,165 B colour mosaic, ``exit 0``,
  ``a2a_failed=0``), so it must never be the default.

Also pins the packet size the whole speed argument rests on: 144 bytes per 128 bf16 values
(0.5625x), the byte ratio quoted in the module docstring.

No GPU is required.
"""

from __future__ import annotations

import pytest
import torch

from vllm_omni.diffusion.models.minimax_h3 import a2a_wire as h3_a2a_wire
from vllm_omni.diffusion.models.minimax_h3 import ar_wire as h3_ar_wire

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture(autouse=True)
def _clean_env_and_state(monkeypatch):
    for var in ("H3_A2A_WIRE", "H3_A2A_WIRE_BUFCACHE", "H3_AR_WIRE"):
        monkeypatch.delenv(var, raising=False)
    h3_a2a_wire._STATE["failed"] = False
    h3_ar_wire._STATE["failed"] = False
    yield


# --- all-to-all gating ---------------------------------------------------------------


def test_a2a_defaults_to_bf16_when_nothing_is_set():
    assert h3_a2a_wire._wire() == "bf16"
    assert h3_a2a_wire.wire_enabled() is False


def test_control_file_envs_are_inert(monkeypatch, tmp_path):
    """The control-file plane is gone: these variables must not select an arm."""
    for name in ("H3_A2A_WIRE_CONTROL", "H3_AR_WIRE_CONTROL"):
        control = tmp_path / name
        control.write_text("int8\n")
        monkeypatch.setenv(name, str(control))
    monkeypatch.setenv("H3_A2A_WIRE", "bf16")
    monkeypatch.setenv("H3_AR_WIRE", "bf16")
    assert h3_a2a_wire.wire_enabled() is False
    assert h3_ar_wire.enabled() is False


@pytest.mark.parametrize("mode", ["int8", "int8-fused", "int8_fused", "fused", "INT8"])
def test_a2a_int8_mode_spellings_all_enable_the_wire(monkeypatch, mode):
    monkeypatch.setenv("H3_A2A_WIRE", mode)
    assert h3_a2a_wire.wire_enabled() is True


@pytest.mark.parametrize("mode", ["bf16", "", "fp8", "none"])
def test_a2a_non_int8_modes_leave_the_wire_disabled(monkeypatch, mode):
    monkeypatch.setenv("H3_A2A_WIRE", mode)
    assert h3_a2a_wire.wire_enabled() is False


def test_a2a_buffer_pool_defaults_to_off(monkeypatch):
    """Regression guard: the pooled variant corrupted the payload, so it is opt-in."""
    assert h3_a2a_wire._bufcache() is False
    monkeypatch.setenv("H3_A2A_WIRE_BUFCACHE", "1")
    assert h3_a2a_wire._bufcache() is True
    monkeypatch.setenv("H3_A2A_WIRE_BUFCACHE", "0")
    assert h3_a2a_wire._bufcache() is False


# --- all-reduce gating ---------------------------------------------------------------


def test_ar_defaults_to_bf16_when_nothing_is_set():
    assert h3_ar_wire._wire() == "bf16"
    assert h3_ar_wire.enabled() is False


def test_ar_accepts_only_exact_int8(monkeypatch):
    monkeypatch.setenv("H3_AR_WIRE", "int8-fused")
    assert h3_ar_wire.enabled() is False


def test_ar_failure_latches_the_fallback(monkeypatch):
    """Once the wire has raised, int8 stays off for the process (no retry loop)."""
    monkeypatch.setenv("H3_AR_WIRE", "int8")
    assert h3_ar_wire.enabled() is True
    h3_ar_wire._STATE["failed"] = True
    assert h3_ar_wire.enabled() is False


def test_ar_disabled_path_calls_the_callers_own_all_reduce(monkeypatch):
    """The bf16 path must be the caller's function, not a reimplementation."""
    monkeypatch.setenv("H3_AR_WIRE", "bf16")
    value = torch.ones(4, 8, dtype=torch.bfloat16)
    seen = []

    def original(x):
        seen.append(x)
        return x + 1

    out = h3_ar_wire.tp_all_reduce(value, original)
    assert seen == [value], "the caller's all-reduce was not called exactly once"
    assert torch.equal(out, original(value))


def test_ar_disabled_path_is_taken_when_the_wire_has_failed(monkeypatch):
    monkeypatch.setenv("H3_AR_WIRE", "int8")
    h3_ar_wire._STATE["failed"] = True
    value = torch.zeros(2, 4, dtype=torch.bfloat16)
    out = h3_ar_wire.tp_all_reduce(value, lambda x: x + 7)
    assert torch.equal(out, value + 7)


# --- the byte ratio the speed argument rests on --------------------------------------


def test_packet_is_0_5625x_of_the_bf16_payload():
    from vllm_omni.diffusion.models.minimax_h3.comm.comm_quant import OUTPUT_PACKET, VECTOR

    assert VECTOR == 128
    assert OUTPUT_PACKET == 144
    assert OUTPUT_PACKET / (VECTOR * 2) == 0.5625


# --- input contract of the encoder (checked before any kernel launch) ----------------


def test_encode_pack_rejects_wrong_dtype_and_shape():
    with pytest.raises(ValueError):
        h3_a2a_wire.encode_pack(torch.zeros(4, 128, dtype=torch.float32))
    with pytest.raises(ValueError):
        h3_a2a_wire.encode_pack(torch.zeros(4, 64, dtype=torch.bfloat16))
    with pytest.raises(ValueError):
        h3_a2a_wire.encode_pack(torch.zeros(4, 256, dtype=torch.bfloat16))
