# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for :mod:`vllm_omni.model_executor.models.sensenova_vision.prompt_utils`.

The think topology wraps raw user content with the BAGEL think system prompt so
the AR (Thinker) stage decodes ``<thinking>`` tokens before KV transfer, and it
lifts the AR stage's decoded text into the DiT request's ``extra_args`` so the
mixed ``{image, text}`` output-modality contract is preserved.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vllm_omni.model_executor.models.sensenova_vision.prompt_utils import (
    bridge_think_text_to_image,
    build_think_prompt,
)
from vllm_omni.model_executor.stage_input_processors.bagel import (
    GEN_THINK_SYSTEM_PROMPT,
    VLM_THINK_SYSTEM_PROMPT,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_IM_START = "<|im_start|>"
_IM_END = "<|im_end|>"


def test_build_think_prompt_default_image_mode() -> None:
    """Image-output modes use the GEN think system prompt with system+user turns."""
    content = "a cute corgi astronaut"
    out = build_think_prompt(content)

    assert out.startswith(f"{_IM_START}system\n")
    assert GEN_THINK_SYSTEM_PROMPT in out
    assert f"{_IM_END}\n" in out
    assert out.endswith(f"{_IM_START}user\n{content}")
    # The system prompt must be the GEN (image-planning) variant, not the VLM one.
    assert VLM_THINK_SYSTEM_PROMPT not in out


def test_build_think_prompt_think_understanding_mode() -> None:
    """``think_understanding`` uses the VLM think system prompt."""
    content = "what is in this image?"
    out = build_think_prompt(content, mode="think_understanding")

    assert VLM_THINK_SYSTEM_PROMPT in out
    assert GEN_THINK_SYSTEM_PROMPT not in out
    assert out.endswith(f"{_IM_START}user\n{content}")


def test_build_think_prompt_has_expected_markers() -> None:
    """The helper emits exactly the chat markers the tokenizer maps to control ids."""
    out = build_think_prompt("hello")
    assert out.count(_IM_START) == 2  # system + user
    assert out.count(_IM_END) == 1  # closes the system turn
    assert "\n<|im_start|>user\n" in out


def _source_output(text: str) -> SimpleNamespace:
    return SimpleNamespace(outputs=[SimpleNamespace(text=text)])


def test_bridge_wires_stage0_text_into_extra_args_and_passes_prompt() -> None:
    """The DIFFUSION custom_process_input_func returns prompt unchanged and records text_output."""
    sampling_params = SimpleNamespace(extra_args={})
    original_prompt = {"prompt": "a cute corgi astronaut", "modalities": ["image"]}

    result = bridge_think_text_to_image(
        [_source_output("thinking about the corgi")],
        prompt=original_prompt,
        sampling_params=sampling_params,
    )

    assert result is original_prompt
    assert sampling_params.extra_args["text_output"] == "thinking about the corgi"


def test_bridge_creates_extra_args_when_missing() -> None:
    """If sampling_params has no extra_args, the bridge creates it."""
    sampling_params = SimpleNamespace(extra_args=None)
    result = bridge_think_text_to_image(
        [_source_output("thought text")],
        prompt="prompt",
        sampling_params=sampling_params,
    )
    assert result == "prompt"
    assert sampling_params.extra_args == {"text_output": "thought text"}


def test_bridge_preserves_explicit_text_output() -> None:
    """Caller-supplied text_output must win over the AR stage text."""
    sampling_params = SimpleNamespace(extra_args={"text_output": "explicit"})
    result = bridge_think_text_to_image(
        [_source_output("ar text")],
        prompt="prompt",
        sampling_params=sampling_params,
    )
    assert result == "prompt"
    assert sampling_params.extra_args == {"text_output": "explicit"}


def test_bridge_noop_without_text_output() -> None:
    """An AR output without text must not clobber anything and pass prompt through."""
    sampling_params = SimpleNamespace(extra_args={"seed": 52})
    result = bridge_think_text_to_image(
        [SimpleNamespace(outputs=[SimpleNamespace(text=None)])],
        prompt={"prompt": "p", "modalities": ["image"]},
        sampling_params=sampling_params,
    )
    assert result == {"prompt": "p", "modalities": ["image"]}
    assert sampling_params.extra_args == {"seed": 52}


def test_bridge_noop_when_no_sampling_params() -> None:
    """Without sampling_params the bridge is a passthrough."""
    result = bridge_think_text_to_image(
        [_source_output("text")],
        prompt="prompt",
    )
    assert result == "prompt"
