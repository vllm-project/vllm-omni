# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import json

import pytest

from vllm_omni.entrypoints.cli.doctor import render_doctor, render_doctor_json
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


def test_render_doctor_json_is_machine_readable():
    info = HardwareInfo(
        backend="cuda",
        device="NVIDIA RTX 4060 Laptop GPU",
        memory_gb=8.0,
        os_name="Linux-WSL2",
        detail="CUDA capability 8.9",
    )

    payload = json.loads(render_doctor_json(info))

    assert payload["hardware"]["backend"] == "cuda"
    assert payload["hardware"]["memory_gb"] == 8.0
    assert payload["tier"] == "not_yet_measured"
    assert payload["recipes"]


def test_render_doctor_does_not_match_cpu_host_memory():
    info = HardwareInfo(
        backend="cpu",
        device="CPU",
        memory_gb=64.0,
        os_name="Linux",
        detail="no accelerator detected",
    )

    output = render_doctor(info)

    assert "No documented recipe targets CPU" in output
    assert "meets the documented minimum" not in output


def test_render_doctor_does_not_match_unified_memory():
    info = HardwareInfo(
        backend="mps",
        device="Apple M2 Ultra",
        memory_gb=192.0,
        os_name="macOS",
        detail="unified memory",
    )

    output = render_doctor(info)

    assert "No documented recipe targets MPS" in output
    assert "meets the documented minimum" not in output


def test_doctor_payload_marks_high_memory_cpu_and_mps_unmeasured():
    for backend in ("cpu", "mps"):
        info = HardwareInfo(
            backend=backend,
            device="generic",
            memory_gb=64.0,
            os_name="generic",
            detail="generic",
        )

        payload = json.loads(render_doctor_json(info))

        assert payload["tier"] == "not_yet_measured"
        assert {recipe["backend"] for recipe in payload["recipes"]} == {"cuda"}
        assert all(recipe["memory_kind"] == "device" for recipe in payload["recipes"])


def test_doctor_payload_marks_matching_cuda_as_meeting():
    info = HardwareInfo(
        backend="cuda",
        device="NVIDIA RTX 4090",
        memory_gb=24.0,
        os_name="Linux",
        detail="CUDA capability 8.9",
    )

    payload = json.loads(render_doctor_json(info))

    assert payload["tier"] == "meets_documented_minimum"
    assert all(recipe["backend"] == "cuda" for recipe in payload["recipes"])
    assert all(recipe["memory_kind"] == "device" for recipe in payload["recipes"])
