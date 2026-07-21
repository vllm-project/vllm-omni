# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the generic diffusion->diffusion transition processor."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class _FakeDiffusionOutput:
    """Stand-in for DiffusionOutput carrying a payload in custom_output."""

    custom_output: dict[str, Any] = field(default_factory=dict)


def _make_payload(interface_mod, **overrides):
    kwargs = dict(
        request_id="req-7",
        boundary=interface_mod.StageBoundary.ENCODE_TO_DIT,
        scalar_fields={"session_id": "sess-A"},
    )
    kwargs.update(overrides)
    return interface_mod.StagePayload.create(**kwargs)


def test_transition_moves_payload_into_prompt_extra(diffusion_processor, interface_mod):
    payload = _make_payload(interface_mod)
    src = _FakeDiffusionOutput(custom_output={diffusion_processor.STAGE_PAYLOAD_OUTPUT_KEY: payload})

    prompt = diffusion_processor.diffusion_stage_transition([src], {"prompt": "hi"}, False)

    assert prompt is not None
    assert prompt["extra"][diffusion_processor.STAGE_PAYLOAD_PROMPT_KEY] is payload
    # session id mirrored to prompt for the denoise runner (from public scalar_fields)
    assert prompt["session_id"] == "sess-A"


def test_transition_returns_none_when_no_outputs(diffusion_processor):
    assert diffusion_processor.diffusion_stage_transition([], None, False) is None


def test_transition_returns_none_when_no_payload(diffusion_processor):
    src = _FakeDiffusionOutput(custom_output={})
    assert diffusion_processor.diffusion_stage_transition([src], None, False) is None


def test_transition_accepts_bare_payload(diffusion_processor, interface_mod):
    payload = _make_payload(interface_mod, boundary=interface_mod.StageBoundary.DIT_TO_DECODE)
    prompt = diffusion_processor.diffusion_stage_transition([payload], None, False)
    assert prompt["extra"][diffusion_processor.STAGE_PAYLOAD_PROMPT_KEY] is payload


def test_transition_passthrough_fields(diffusion_processor, interface_mod):
    payload = _make_payload(interface_mod)
    src = _FakeDiffusionOutput(custom_output={diffusion_processor.STAGE_PAYLOAD_OUTPUT_KEY: payload})
    prompt = diffusion_processor.diffusion_stage_transition(
        [src],
        {"prompt": "p", "seed": 42, "num_inference_steps": 4, "guidance_scale": 1.5},
        False,
    )
    assert prompt["seed"] == 42
    assert prompt["num_inference_steps"] == 4
    assert prompt["guidance_scale"] == 1.5


def test_transition_accepts_sampling_params_kwarg(diffusion_processor, interface_mod):
    # Orchestrator probes the signature and passes sampling_params=...; the
    # generic processor must accept it without error.
    payload = _make_payload(interface_mod)
    src = _FakeDiffusionOutput(custom_output={diffusion_processor.STAGE_PAYLOAD_OUTPUT_KEY: payload})
    prompt = diffusion_processor.diffusion_stage_transition(
        [src], None, True, sampling_params=object()
    )
    assert prompt is not None


def test_transition_rejects_invalid_payload(diffusion_processor, interface_mod):
    # A payload with a bad payload_version fails validate() -> None routed.
    bad = interface_mod.StagePayload(
        request_id="r",
        boundary=interface_mod.StageBoundary.ENCODE_TO_DIT,
        payload_version=999,
    )
    src = _FakeDiffusionOutput(custom_output={diffusion_processor.STAGE_PAYLOAD_OUTPUT_KEY: bad})
    assert diffusion_processor.diffusion_stage_transition([src], None, False) is None
