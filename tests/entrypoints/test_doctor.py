# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""``vllm-omni doctor`` reports devices against the documented tiers.

A documented minimum is a capacity bound on dedicated device memory. These
tests pin the extra gates: a variant is only a fit when its recipe declares the
device's backend and, where the recipe lists hardware, includes the card.
"""

import argparse
import json

import pytest

from vllm_omni.entrypoints.cli import doctor
from vllm_omni.entrypoints.cli.main import wants_omni_cli
from vllm_omni.utils.hardware import Device

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

VARIANTS = [
    # No platform declared: capacity only, never a fit.
    {"model_id": "acme/tiny-tts", "variant": "default", "minimum_gib": 8, "precision": "bf16"},
    # Declared CUDA with no hardware allowlist: fits any CUDA card at the tier.
    {
        "model_id": "acme/cuda-tts",
        "variant": "default",
        "minimum_gib": 16,
        "precision": "bf16",
        "backends": ["cuda"],
    },
    # Declared CUDA but restricted to Blackwell parts.
    {
        "model_id": "acme/blackwell-video",
        "variant": "nvfp4",
        "minimum_gib": 13,
        "precision": "nvfp4",
        "backends": ["cuda"],
        "hardware": ["b200", "gb200", "b300", "gb300"],
    },
    # Declared ROCm only.
    {
        "model_id": "acme/rocm-tts",
        "variant": "default",
        "minimum_gib": 8,
        "precision": "bf16",
        "backends": ["rocm"],
    },
    {"model_id": "acme/big-video", "variant": "default", "minimum_gib": 80, "precision": "bf16"},
]


def _device(total_mib, kind="cuda", name="fake", unified_memory=False):
    return Device(kind=kind, name=name, total_mib=total_mib, unified_memory=unified_memory)


def _statuses(device, variants=None):
    rows = doctor.classify_variants(device, VARIANTS if variants is None else variants)
    return {(row[0]["model_id"], row[0]["variant"]): row[1] for row in rows}


def test_marketed_memory_size_rounds_into_its_own_tier():
    """A 16 GiB GPU reports ~15.9 GiB; it must not be floored out of the 16 GiB band."""
    laptop_16_gib = _device(16303)

    assert doctor.device_tier_gib(laptop_16_gib) == 16
    assert set(_statuses(laptop_16_gib)) >= {("acme/tiny-tts", "default"), ("acme/cuda-tts", "default")}


def test_capacity_filter_uses_the_documented_minimum():
    at_8_gib = doctor.classify_variants(_device(8 * 1024), VARIANTS)
    assert {row[0]["minimum_gib"] for row in at_8_gib} == {8}
    assert len(doctor.classify_variants(_device(80 * 1024), VARIANTS)) == len(VARIANTS)


def test_unknown_device_memory_is_not_reported_as_unsupported():
    report = doctor.render_report([_device(None)], VARIANTS)

    assert "memory could not be read" in report
    assert "not yet measured, not unsupported" in report


def test_device_below_every_documented_minimum_says_unmeasured():
    report = doctor.render_report([_device(6 * 1024)], VARIANTS)

    assert "no variant is documented for this device at or below 6 GiB" in report
    assert "not yet measured, not unsupported" in report


def test_cuda_positive_is_a_fit():
    """A CUDA card whose recipe declares CUDA is the positive control."""
    device = _device(16 * 1024, name="NVIDIA GeForce RTX 4060 Ti")

    assert [variant["model_id"] for variant in doctor.fitting_variants(device, VARIANTS)] == ["acme/cuda-tts"]
    assert "1 documented variant across 1 model fit" in doctor.render_report([device], VARIANTS)


@pytest.mark.parametrize(
    ("device", "reason"),
    [
        (_device(32 * 1024, kind="cpu", name="AMD EPYC 32 cores"), "system-memory"),
        (_device(32 * 1024, kind="mps", name="Apple M2 Max", unified_memory=True), "unified-memory"),
    ],
)
def test_cpu_and_unified_memory_are_never_fits(device, reason):
    """System or unified memory is not device memory, so nothing is compatible."""
    rows = doctor.classify_variants(device, VARIANTS)

    assert rows, "32 GiB still holds the smallest documented minimum"
    assert {status for _, status, _ in rows} == {doctor.CAPACITY_ONLY}
    assert {why for _, _, why in rows} == {reason}

    report = doctor.render_report([device], VARIANTS)
    assert " no variant is documented for this device" in report
    assert " fit: " not in report


def test_rocm_device_does_not_match_cuda_only_recipes():
    device = _device(16 * 1024, kind="rocm", name="AMD Instinct MI300X")
    statuses = _statuses(device, [variant for variant in VARIANTS if variant["model_id"] != "acme/rocm-tts"])

    assert set(statuses.values()) == {doctor.CAPACITY_ONLY}
    assert "recipe documents another backend" in doctor.render_report([device], VARIANTS)


def test_declared_backend_mismatch_is_reported_with_its_reason():
    device = _device(8 * 1024, kind="rocm", name="AMD Radeon RX 7900")
    rows = doctor.classify_variants(device, VARIANTS)
    statuses = {row[0]["model_id"]: row[1] for row in rows}
    reasons = {row[0]["model_id"]: row[2] for row in rows}

    assert statuses["acme/rocm-tts"] == doctor.COMPATIBLE
    assert reasons["acme/tiny-tts"] == "backend-not-declared"


def test_cuda_card_outside_the_allowlist_needs_specific_hardware():
    """A 24 GiB consumer card must not claim a B200-only NVFP4 variant."""
    device = _device(24564, name="NVIDIA GeForce RTX 4090")

    assert _statuses(device)[("acme/blackwell-video", "nvfp4")] == doctor.NEEDS_HARDWARE
    assert ("acme/blackwell-video", "nvfp4") not in {
        (variant["model_id"], variant["variant"]) for variant in doctor.fitting_variants(device, VARIANTS)
    }

    report = doctor.render_report([device], VARIANTS)
    assert "needs specific hardware" in report
    assert "b200, gb200, b300, gb300" in report


def test_cuda_card_inside_the_allowlist_is_a_fit():
    device = _device(183 * 1024, name="NVIDIA B200")

    assert _statuses(device)[("acme/blackwell-video", "nvfp4")] == doctor.COMPATIBLE


def test_unlisted_hardware_is_not_assumed_to_match():
    """An allowlisted recipe plus an unknown card name stays out of the fits."""
    device = _device(48 * 1024, name="NVIDIA GeForce RTX 6000 Ada")

    assert doctor.device_hardware_slugs(device) == set()
    assert _statuses(device)[("acme/blackwell-video", "nvfp4")] == doctor.NEEDS_HARDWARE


def test_report_names_the_fitting_models():
    report = doctor.render_report([_device(16 * 1024, name="NVIDIA GeForce RTX 4060 Ti")], VARIANTS)

    assert "1 documented variant across 1 model fit" in report
    assert "`acme/cuda-tts`" in report
    assert "`acme/tiny-tts`" in report  # capacity only, never in the fit list
    assert "acme/big-video" not in report


def test_report_uses_singular_counts():
    report = doctor.render_report([_device(8 * 1024, kind="rocm", name="AMD Radeon RX 7900")], VARIANTS)

    assert "1 documented variant across 1 model fit" in report
    assert "1 more variant fits on capacity alone" in report


def test_report_caps_the_model_list():
    variants = [
        {"model_id": f"acme/model-{index:02d}", "variant": "default", "minimum_gib": 8, "precision": "bf16"}
        for index in range(doctor._MAX_SHOWN_MODELS + 2)
    ]

    report = doctor.render_report([_device(16 * 1024)], variants)

    assert "and 2 more" in report


def test_packaged_index_is_loadable_and_documented():
    variants = doctor.load_documented_variants()

    assert variants, "recipes/models/ must contribute at least one documented minimum"
    assert {"model_id", "variant", "minimum_gib", "backends", "memory_kind"} <= set(variants[0])
    # The index must not claim a backend the recipes never declared.
    assert all(isinstance(variant["backends"], list) for variant in variants)


@pytest.mark.parametrize(
    ("device", "statuses", "reason"),
    [
        (_device(32 * 1024, kind="cpu", name="AMD EPYC 32 cores"), {"capacity_only"}, "system-memory"),
        (
            _device(32 * 1024, kind="mps", name="Apple M2 Max", unified_memory=True),
            {"capacity_only"},
            "unified-memory",
        ),
        (_device(8 * 1024, kind="rocm", name="AMD Radeon RX 7900"), {"capacity_only", "compatible"}, None),
        (
            _device(16 * 1024, name="NVIDIA GeForce RTX 4060 Ti"),
            {"capacity_only", "compatible", "needs_hardware"},
            None,
        ),
        (_device(24564, name="NVIDIA GeForce RTX 4090"), {"capacity_only", "compatible", "needs_hardware"}, None),
    ],
)
def test_json_output_carries_a_status_per_variant(device, statuses, reason):
    payload = doctor.report_json([device], VARIANTS)
    rows = payload["reports"][0]["variants"]

    assert {row["status"] for row in rows} == statuses
    if reason is not None:
        assert {row["reason"] for row in rows} == {reason}


def test_json_output_is_machine_readable(monkeypatch, capsys):
    device = _device(24564, name="NVIDIA GeForce RTX 4090")
    monkeypatch.setattr(doctor, "detect_devices", lambda: [device])
    monkeypatch.setattr(doctor, "load_documented_variants", lambda: VARIANTS)
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")

    doctor.OmniDoctorSubcommand.cmd(parser.parse_args(["--json"]))

    payload = json.loads(capsys.readouterr().out)
    assert payload["devices"][0]["total_mib"] == 24564
    report = payload["reports"][0]
    assert report["tier_gib"] == 24
    nvfp4 = next(v for v in report["variants"] if v["model_id"] == "acme/blackwell-video")
    assert nvfp4["status"] == "needs_hardware"
    assert nvfp4["hardware"] == ["b200", "gb200", "b300", "gb300"]


def test_doctor_does_not_need_the_omni_marker():
    """vLLM has no `doctor` subcommand, so the shim must keep the call in-tree."""
    assert wants_omni_cli(["vllm-omni", "doctor"])
    assert wants_omni_cli(["vllm-omni", "doctor", "--json"])
    assert wants_omni_cli(["vllm-omni", "serve", "some/model", "--omni"])
    assert not wants_omni_cli(["vllm-omni", "serve", "some/model"])
    assert not wants_omni_cli(["vllm-omni"])


def test_subcommand_registers_doctor_with_a_json_flag():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="subcommand")

    (command,) = doctor.cmd_init()
    command.subparser_init(subparsers)

    assert command.name == "doctor"
    assert parser.parse_args(["doctor", "--json"]).json is True
