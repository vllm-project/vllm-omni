# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prompt construction + AR->DiT text bridging helpers for the SenseNova-Vision think topology.

The ``sensenova_vision_think`` pipeline splits raw text into a system (think)
segment and a user segment so the AR (Thinker) stage decodes ``<thinking>``
tokens before the KV cache is handed to the DiT stage.  This module centralises
the two glue pieces that topology needs but that do not belong in the frozen
BAGEL core:

* :func:`build_think_prompt` — wrap raw user content with the think system
  prompt so the chat template sees a single ``system`` turn followed by the
  ``user`` turn.
* :func:`bridge_think_text_to_image` — the stage-1 ``custom_process_input_func``
  that lifts the AR stage's decoded text into the diffusion request's
  ``extra_args["text_output"]`` so :meth:`SenseNovaVisionPipeline._merge_mixed_task_text`
  surfaces it under the existing ``{image, text}`` output-modality contract.

System prompts
--------------
The system prompt text is imported verbatim from
``vllm_omni.model_executor.stage_input_processors.bagel``
(``GEN_THINK_SYSTEM_PROMPT`` / ``VLM_THINK_SYSTEM_PROMPT``) so the two stages
share a single source of truth and there is no copy drift.

Single-bos/eos limitation (empirical GPU check needed)
-------------------------------------------------------
Upstream SenseNova-Vision emits a separate ``bos``/``eos`` pair around *every*
prompt segment (``SenseNova-Vision/modeling/bagel/bagel.py:303-304``), i.e. one
``<|im_start|>``/``<|im_end|>`` pair for the system turn and another for the
user turn.  The vllm-omni port's :meth:`~vllm_omni.diffusion.models.bagel.bagel_transformer.Bagel.prepare_prompts`
(``bagel_transformer.py:1390-1391``) wraps the prompt with a single ``bos``/``eos``
pair per call, so it can only express **one** system turn inside one chat template.

To stay within that contract the helper emits a single chat string with the
system and user turns concatenated inside one ``<|im_start|>…<|im_end|>``
span.  Whether the checkpoint/DiT conditioning treats this identically to the
upstream per-segment bos/eos layout must be verified on GPU before relying on
it for parity — this is flagged for an empirical check rather than assumed.
"""

from __future__ import annotations

from typing import Any

from vllm.logger import init_logger

# Single source of truth for the think system prompts (shared with the AR
# stage's prompt-expansion module so the two stages never drift apart).
from vllm_omni.model_executor.stage_input_processors.bagel import (
    GEN_THINK_SYSTEM_PROMPT,
    VLM_THINK_SYSTEM_PROMPT,
)

logger = init_logger(__name__)

# Chat markers used by SenseNova-Vision / BAGEL.  These must appear verbatim in
# the built prompt so the tokenizer maps them to the checkpoint's control ids.
_IM_START = "<|im_start|>"
_IM_END = "<|im_end|>"

# Mode marker that selects the VLM (understanding) system prompt instead of the
# image-generation one.
_THINK_UNDERSTANDING_MODE = "think_understanding"


def build_think_prompt(content: str, mode: str = "generate") -> str:
    """Wrap ``content`` with the think system prompt for image-output modes.

    Produces a single chat string of the form::

        <|im_start|>system
        {system_prompt}<|im_end|>
        <|im_start|>user
        {content}

    ``system_prompt`` is :data:`GEN_THINK_SYSTEM_PROMPT` for image-output modes
    (default) and :data:`VLM_THINK_SYSTEM_PROMPT` when ``mode ==
    "think_understanding"``.  ``content`` is used as-is (no extra bos/eos is
    added here; ``prepare_prompts`` adds the port's single pair later).

    See the module docstring for the single-bos/eos caveat vs upstream.
    """
    system_prompt = (
        VLM_THINK_SYSTEM_PROMPT if mode == _THINK_UNDERSTANDING_MODE else GEN_THINK_SYSTEM_PROMPT
    )
    return (
        f"{_IM_START}system\n{system_prompt}{_IM_END}\n"
        f"{_IM_START}user\n{content}"
    )


def bridge_think_text_to_image(
    source_outputs: list[Any],
    prompt: Any | None = None,
    requires_multimodal_data: bool = False,  # noqa: ARG001
    sampling_params: Any | None = None,
) -> Any:
    """Stage-1 ``custom_process_input_func``: surface AR think text + pass prompt through.

    The orchestrator invokes this with ``(diffusion_source_outputs, prompt,
    requires_multimodal_data, sampling_params=diffusion_stage_params)`` before
    submitting the DiT request.  Two jobs:

    1. Decode the stage-0 (AR Thinker) generated text and record it on the
       diffusion stage's ``sampling_params.extra_args["text_output"]``.  The
       DiT pipeline's :meth:`SenseNovaVisionPipeline._merge_mixed_task_text`
       lifts exactly this key into ``payload["text"]`` so the final output stays
       ``{image, text}`` without new payload keys.
    2. Return ``prompt`` unchanged so the DiT still conditions on the original
       user prompt (the stage-0 KV cache carries the thinking).

    If no stage-0 text can be extracted, or ``sampling_params`` has no
    ``extra_args``, this is a no-op that still passes ``prompt`` through.
    """
    text = _extract_stage0_text(source_outputs)
    if text and sampling_params is not None:
        extra = getattr(sampling_params, "extra_args", None)
        if extra is None:
            extra = {}
            sampling_params.extra_args = extra  # type: ignore[attr-defined]
        # Do not clobber an explicitly supplied value.
        extra.setdefault("text_output", text)
        logger.debug(
            "bridge_think_text_to_image: staged stage-0 text (len=%d) into "
            "diffusion extra_args['text_output']",
            len(text),
        )
    return prompt


def _extract_stage0_text(source_outputs: list[Any]) -> str | None:
    """Return the decoded text of the first AR stage output, or ``None``.

    ``source_outputs`` are :class:`~vllm_omni.outputs.OmniRequestOutput`
    instances; the AR stage exposes its generated text via ``outputs[0].text``
    (a :class:`~vllm.transformers_utils...CompletionOutput`).
    """
    if not source_outputs:
        return None
    output = getattr(source_outputs[0], "outputs", None)
    if not output:
        return None
    text = getattr(output[0], "text", None)
    if isinstance(text, str) and text.strip():
        return text
    return None


__all__ = [
    "build_think_prompt",
    "bridge_think_text_to_image",
    "GEN_THINK_SYSTEM_PROMPT",
    "VLM_THINK_SYSTEM_PROMPT",
]
