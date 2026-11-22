# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""``vllm-omni doctor`` reports devices against the documented tiers."""

import argparse
import json

import pytest

from vllm_omni.entrypoints.cli import doctor
from vllm_omni.entrypoints.cli.main import wants_omni_cli
from vllm_omni.utils.hardware import Device

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

VARIANTS = [
    {"model_id": "acme/tiny-tts", "variant": "default", "minimum_gib": 8, "precision": "bf16"},
    {"model_id": "acme/mid-tts", "variant": "default", "minimum_gib": 16, "precision": "bf16"},
    {"model_id": "acme/big-video", "variant": "default", "minimum_gib": 80, "precision": "bf16"},
]


def _device(total_mib, kind="cuda"):
    return Device(kind=kind, name="fake", total_mib=total_mib)


def test_marketed_memory_size_rounds_into_its_own_tier():
    """A 16 GiB GPU reports ~15.9 GiB; it must not be floored out of the 16 GiB band."""
    laptop_16_gib = _device(16303)

    assert doctor.device_tier_gib(laptop_16_gib) == 16
    assert len(doctor.fitting_variants(laptop_16_gib, VARIANTS)) == 2


def test_fitting_variants_uses_the_documented_minimum():
    at_8_gib = doctor.fitting_variants(_device(8 * 1024), VARIANTS)
    assert [variant["model_id"] for variant in at_8_gib] == ["acme/tiny-tts"]
    assert len(doctor.fitting_variants(_device(16 * 1024), VARIANTS)) == 2
    assert len(doctor.fitting_variants(_device(80 * 1024), VARIANTS)) == 3


def test_unknown_device_memory_is_not_reported_as_unsupported():
    report = doctor.render_report([_device(None)], VARIANTS)

    assert "memory could not be read" in report
    assert "not yet measured, not unsupported" in report


def test_device_below_every_documented_minimum_says_unmeasured():
    report = doctor.render_report([_device(6 * 1024)], VARIANTS)

    assert "no recipe documents a minimum at or below 6 GiB" in report
    assert "not yet measured, not unsupported" in report


def test_report_names_the_fitting_models():
    report = doctor.render_report([_device(16 * 1024)], VARIANTS)

    assert "2 documented variants across 2 models fit" in report
    assert "`acme/tiny-tts`" in report
    assert "acme/big-video" not in report


def test_report_uses_singular_counts():
    report = doctor.render_report([_device(8 * 1024)], VARIANTS)

    assert "1 documented variant across 1 model fit" in report
    assert "`acme/tiny-tts`" in report


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
    assert {"model_id", "variant", "minimum_gib"} <= set(variants[0])


def test_json_output_is_machine_readable(monkeypatch, capsys):
    monkeypatch.setattr(doctor, "detect_devices", lambda: [_device(16 * 1024)])
    monkeypatch.setattr(doctor, "load_documented_variants", lambda: VARIANTS)
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")

    doctor.OmniDoctorSubcommand.cmd(parser.parse_args(["--json"]))

    payload = json.loads(capsys.readouterr().out)
    assert payload["devices"][0]["total_mib"] == 16 * 1024
    assert len(payload["documented_fits"][0]["variants"]) == 2


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
