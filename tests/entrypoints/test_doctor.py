# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest

from vllm_omni.entrypoints.cli.doctor import render_doctor
from vllm_omni.entrypoints.cli.main import _is_omni_invocation
from vllm_omni.utils.hardware import HardwareInfo

pytestmark = [pytest.mark.cpu, pytest.mark.core_model]


def test_doctor_dispatches_without_omni_flag():
    assert _is_omni_invocation(["vllm-omni", "doctor"])
    assert _is_omni_invocation(["vllm-omni", "serve", "Qwen/Qwen2.5-Omni-7B", "--omni"])
    assert not _is_omni_invocation(["vllm-omni", "serve", "Qwen/Qwen2.5-Omni-7B"])


def test_render_doctor_marks_below_tier_as_not_measured():
    info = HardwareInfo(
        backend="cuda",
        device="NVIDIA RTX 4060 Laptop GPU",
        memory_gb=8.0,
        os_name="Linux-WSL2",
        detail="CUDA capability 8.9",
    )

    output = render_doctor(info)

    assert "NVIDIA RTX 4060 Laptop GPU" in output
    assert "below every documented recipe" in output
    assert "not yet measured" in output
    assert "unsupported" in output


def test_render_doctor_meets_documented_minimum():
    info = HardwareInfo(
        backend="cuda",
        device="NVIDIA RTX 4090",
        memory_gb=24.0,
        os_name="Linux",
        detail="CUDA capability 8.9",
    )

    output = render_doctor(info)

    assert "meets the documented minimum" in output
    assert "end-to-end run is still required" in output
