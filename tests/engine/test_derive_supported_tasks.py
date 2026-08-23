# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import dataclasses
import types

import pytest

from vllm_omni.engine import async_omni_engine as async_omni_engine_module
from vllm_omni.engine.async_omni_engine import (
    _MODEL_CAPABILITY_TASKS,
    _CapabilityTask,
    _derive_supported_tasks,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _pool(*, is_comprehension=False, final_output_type="text", stage_client=True, vllm_config=None):
    client = (
        types.SimpleNamespace(is_comprehension=is_comprehension, final_output_type=final_output_type)
        if stage_client
        else None
    )
    return types.SimpleNamespace(stage_client=client, stage_vllm_config=vllm_config)


def _vllm_config(model="m", architectures=("SomeArch",)):
    return types.SimpleNamespace(model_config=types.SimpleNamespace(architectures=list(architectures), model=model))


def _is_comprehension_stage(client):
    return getattr(client, "is_comprehension", False)


def _is_audio_output_stage(client):
    return getattr(client, "final_output_type", None) == "audio"


def _transcription_rule_with_probe(probe):
    """The real transcription entry (task and eligibility intact) with the registry probe stubbed."""
    return dataclasses.replace(_MODEL_CAPABILITY_TASKS[0], probe=probe)


def test_no_stages_defaults_to_generate():
    assert _derive_supported_tasks([]) == ("generate",)


def test_comprehension_stage_yields_generate():
    assert set(_derive_supported_tasks([_pool(is_comprehension=True)])) == {"generate"}


def test_audio_output_stage_yields_speech():
    tasks = _derive_supported_tasks(
        [
            _pool(is_comprehension=True),
            _pool(final_output_type="audio"),
        ]
    )
    assert set(tasks) == {"generate", "speech"}


def test_transcription_capable_comprehension_stage_yields_transcription(monkeypatch):
    rule = _transcription_rule_with_probe(lambda _architectures, _model_config: True)
    monkeypatch.setattr(async_omni_engine_module, "_MODEL_CAPABILITY_TASKS", (rule,))
    tasks = _derive_supported_tasks(
        [
            _pool(is_comprehension=True, vllm_config=_vllm_config()),
            _pool(final_output_type="audio"),
        ]
    )
    assert set(tasks) == {"generate", "speech", "transcription"}


def test_transcription_requires_an_eligible_stage(monkeypatch):
    # Transcription declares comprehension stages eligible; an audio-only
    # pipeline never advertises it, however capable its model claims to be.
    rule = _transcription_rule_with_probe(lambda _architectures, _model_config: True)
    monkeypatch.setattr(async_omni_engine_module, "_MODEL_CAPABILITY_TASKS", (rule,))
    tasks = _derive_supported_tasks([_pool(final_output_type="audio", vllm_config=_vllm_config())])
    assert "transcription" not in tasks


def test_transcription_absent_when_model_lacks_support(monkeypatch):
    rule = _transcription_rule_with_probe(lambda _architectures, _model_config: False)
    monkeypatch.setattr(async_omni_engine_module, "_MODEL_CAPABILITY_TASKS", (rule,))
    tasks = _derive_supported_tasks(
        [
            _pool(is_comprehension=True, vllm_config=_vllm_config()),
            _pool(final_output_type="audio"),
        ]
    )
    assert "transcription" not in tasks


def test_capability_table_drives_derivation(monkeypatch):
    monkeypatch.setattr(
        async_omni_engine_module,
        "_MODEL_CAPABILITY_TASKS",
        (
            _CapabilityTask(
                task="transcription",
                probe=lambda _architectures, _model_config: False,
                eligible=_is_comprehension_stage,
            ),
            _CapabilityTask(
                task="hypothetical",
                probe=lambda _architectures, _model_config: True,
                eligible=_is_comprehension_stage,
            ),
        ),
    )
    tasks = _derive_supported_tasks([_pool(is_comprehension=True, vllm_config=_vllm_config())])
    assert set(tasks) == {"generate", "hypothetical"}


def test_eligibility_is_per_entry(monkeypatch):
    # A capability may belong to a non-comprehension stage: an entry eligible
    # on audio-output stages probes the talker's model, not the thinker's.
    probed = []

    def probe(_architectures, model_config):
        probed.append(model_config.model)
        return True

    monkeypatch.setattr(
        async_omni_engine_module,
        "_MODEL_CAPABILITY_TASKS",
        (_CapabilityTask(task="hypothetical", probe=probe, eligible=_is_audio_output_stage),),
    )
    tasks = _derive_supported_tasks(
        [
            _pool(is_comprehension=True, vllm_config=_vllm_config(model="thinker")),
            _pool(final_output_type="audio", vllm_config=_vllm_config(model="talker")),
        ]
    )
    assert "hypothetical" in tasks
    assert probed == ["talker"]


def test_check_skips_ineligible_stages_without_probing():
    probed = []

    def probe(_architectures, model_config):
        probed.append(model_config.model)
        return True

    rule = _CapabilityTask(task="transcription", probe=probe, eligible=_is_comprehension_stage)
    assert rule.check(_pool(stage_client=False, vllm_config=_vllm_config(model="orphan"))) is False
    assert rule.check(_pool(final_output_type="audio", vllm_config=_vllm_config(model="talker"))) is False
    assert probed == []
    assert rule.check(_pool(is_comprehension=True, vllm_config=_vllm_config(model="thinker"))) is True
    assert probed == ["thinker"]


def test_check_false_without_vllm_config():
    # Diffusion-stage pools carry no vLLM config; an eligible stage without
    # one has nothing to probe.
    rule = _CapabilityTask(
        task="transcription",
        probe=lambda _architectures, _model_config: True,
        eligible=_is_comprehension_stage,
    )
    assert rule.check(_pool(is_comprehension=True, vllm_config=None)) is False


def test_check_passes_architectures_and_model_config_to_probe():
    calls = []

    def probe(architectures, model_config):
        calls.append((architectures, model_config))
        return True

    rule = _CapabilityTask(task="transcription", probe=probe, eligible=_is_comprehension_stage)
    config = _vllm_config()
    assert rule.check(_pool(is_comprehension=True, vllm_config=config)) is True
    assert calls == [(["SomeArch"], config.model_config)]


def test_check_degrades_failing_probe_to_false():
    # A capability probe must degrade to False on any inspection failure
    # rather than take down engine boot.
    def probe(_architectures, _model_config):
        raise RuntimeError("uninspectable model")

    rule = _CapabilityTask(task="transcription", probe=probe, eligible=_is_comprehension_stage)
    assert rule.check(_pool(is_comprehension=True, vllm_config=_vllm_config())) is False
