# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU tests for the AR->DiT think-text bridge gating.

``bridge_think_text_to_image`` lifts the AR (Thinker) decoded text into the
diffusion request's ``extra_args["text_output"]`` so ``_merge_mixed_task_text``
can surface it under the ``{image, text}`` output-modality contract.  With the
per-request AR ``max_tokens=1`` clamp, non-think DiT-routed tasks
(``dense_perception`` / ``recon3d``) decode a junk 1-token artifact that must
NOT be lifted.  Only thinking modes (``caption_generate`` / ``think_*``)
surface AR text.

These tests are CPU-only; no model weights or GPU are required.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from vllm.outputs import CompletionOutput, RequestOutput

from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.model_executor.models.sensenova_vision.prompt_utils import (
    THINK_TEXT_MODES,
    _mode_surfaces_text,
    bridge_think_text_to_image,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _source_outputs(text: str) -> list[SimpleNamespace]:
    completion = CompletionOutput(
        index=0,
        text=text,
        token_ids=[1, 2, 3],
        cumulative_logprob=None,
        logprobs=None,
        finish_reason="length",
        stop_reason=None,
    )
    return [
        RequestOutput(
            request_id="req-ar",
            prompt="prompt",
            prompt_token_ids=[101],
            prompt_logprobs=None,
            outputs=[completion],
            finished=True,
            metrics=None,
            lora_request=None,
        )
    ]


def _diffusion_params() -> OmniDiffusionSamplingParams:
    return OmniDiffusionSamplingParams(num_inference_steps=1)


@pytest.mark.parametrize("mode", sorted(THINK_TEXT_MODES))
def test_bridge_lifts_text_for_thinking_modes(mode: str) -> None:
    """caption_generate / think_* modes surface the AR text as text_output."""
    params = _diffusion_params()
    prompt = {"mode": mode}
    out = bridge_think_text_to_image(_source_outputs("a red car"), prompt=prompt, sampling_params=params)
    assert out is prompt
    assert params.extra_args.get("text_output") == "a red car"


@pytest.mark.parametrize(
    "mode", ["dense_perception", "recon3d", "edit", "generate", "understanding", "dense_detection"]
)
def test_bridge_skips_text_for_non_think_modes(mode: str) -> None:
    """Non-think DiT-routed modes must NOT lift the (possibly junk) AR text."""
    params = _diffusion_params()
    prompt = {"mode": mode}
    out = bridge_think_text_to_image(_source_outputs("<|im_start|>"), prompt=prompt, sampling_params=params)
    assert out is prompt
    assert params.extra_args.get("text_output") is None


def test_bridge_skips_when_prompt_is_not_a_dict() -> None:
    """A non-dict prompt (no mode) never surfaces text."""
    params = _diffusion_params()
    out = bridge_think_text_to_image(_source_outputs("text"), prompt="bare string", sampling_params=params)
    assert out == "bare string"
    assert params.extra_args.get("text_output") is None


def test_bridge_skips_without_sampling_params() -> None:
    """Without sampling params the bridge is a no-op pass-through."""
    prompt = {"mode": "caption_generate"}
    out = bridge_think_text_to_image(_source_outputs("text"), prompt=prompt, sampling_params=None)
    assert out is prompt


def test_bridge_skips_when_no_stage0_text() -> None:
    """No stage-0 text means nothing is staged."""
    params = _diffusion_params()
    prompt = {"mode": "caption_generate"}
    out = bridge_think_text_to_image([], prompt=prompt, sampling_params=params)
    assert out is prompt
    assert params.extra_args.get("text_output") is None


def test_mode_surfaces_text_helper() -> None:
    """The gating helper mirrors the think-mode set."""
    assert _mode_surfaces_text({"mode": "caption_generate"}) is True
    assert _mode_surfaces_text({"mode": "dense_perception"}) is False
    assert _mode_surfaces_text({"mode": "recon3d"}) is False
    assert _mode_surfaces_text({"sensenova_vision_mode": "think_generate"}) is True
    assert _mode_surfaces_text(None) is False
    assert _mode_surfaces_text("not a dict") is False
