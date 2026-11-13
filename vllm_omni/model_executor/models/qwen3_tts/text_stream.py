# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Qwen3-TTS streamed text and the text lead.

A CustomVoice request in streaming mode (``non_streaming_mode`` false) may set
``additional_information["text_lead"] = k``: its prompt places the first ``k``
text ids before codec BOS, the layout of a checkpoint trained on text leads.
It may also stream its text (``additional_information["text_stream"]``, then
``AsyncOmni.append_text``) when its Talker stage sets
``text_stream_policy: vllm_omni.model_executor.models.qwen3_tts.text_stream.text_stream_spec``.
Either way the prompt reads text ids ``[0, k]`` (``k`` lead rows, then the BOS
row), and each later codec frame reads the next id.

This module stays light: the scheduler imports it by name.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from vllm_omni.core.sched.text_stream import TextStreamSpec, first_value

if TYPE_CHECKING:
    from vllm.config import ModelConfig

TEXT_LEAD_KEY = "text_lead"


def non_streaming_mode(info: Mapping[str, Any], task_type: str) -> bool:
    """The request's prompt mode; unset follows the official defaults (CustomVoice/VoiceDesign: on, Base: off)."""
    value = first_value(info.get("non_streaming_mode"))
    if isinstance(value, bool):
        return value
    return task_type in ("CustomVoice", "VoiceDesign")


def text_lead(info: Mapping[str, Any] | None) -> int:
    """Text ids the request places before codec BOS (0: the stock streaming layout)."""
    value = first_value(info.get(TEXT_LEAD_KEY)) if info else None
    if value is None:
        return 0
    # An integral float (2.0) is accepted; bool, NaN, infinities and fractions are not.
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value != int(value)
        or value < 0
    ):
        raise ValueError(f"text_lead must be an integer >= 0, got {value!r}")
    return int(value)


class TextLeadError(ValueError):
    """A request's ``text_lead`` that its prompt cannot lay out: a client error, raised before the request runs."""


def text_lead_error(
    info: Mapping[str, Any] | None, num_text_ids: int | None = None, *, task_type: str | None = None
) -> str | None:
    """Why the request's ``text_lead`` cannot be laid out, or ``None``.

    A lead applies to the CustomVoice prompt in streaming mode only, where it is
    otherwise ignored. With ``num_text_ids`` (a whole text's token count) it is
    also bounded: the lead rows may hold every id and the text END, no more.
    ``task_type`` is the caller's resolved task type (default: the request's own).
    """
    try:
        lead = text_lead(info)
    except ValueError as exc:
        return str(exc)
    if lead == 0 or info is None:
        return None
    task_type = task_type or first_value(info.get("task_type")) or "CustomVoice"
    if task_type != "CustomVoice":
        return "'text_lead' is only valid for CustomVoice task"
    if non_streaming_mode(info, task_type):
        return "'text_lead' requires non_streaming_mode=false (the streaming prompt layout)"
    if num_text_ids is not None and lead > num_text_ids + 1:
        return f"'text_lead' {lead} exceeds the text's {num_text_ids} tokens plus its end"
    return None


def text_stream_spec(info: Mapping[str, Any], model_config: ModelConfig | None = None) -> TextStreamSpec:
    """``text_stream_policy`` of a Qwen3-TTS Talker stage: how a request reads its streamed text.

    Raises ``ValueError`` for a request whose prompt layout cannot stream.
    Appended ids must index the Talker's text embedding.
    """
    task_type = first_value(info.get("task_type")) or "CustomVoice"
    if task_type != "CustomVoice":
        raise ValueError(f"Qwen3-TTS streams text only for CustomVoice requests, not {task_type}.")
    if non_streaming_mode(info, "CustomVoice"):
        raise ValueError("Qwen3-TTS streams text only in streaming mode: set non_streaming_mode to false.")
    lead = text_lead(info)
    talker_config = getattr(getattr(model_config, "hf_config", None), "talker_config", None)
    text_vocab_size = getattr(talker_config, "text_vocab_size", None)
    # A text that ends before its prompt is full still fits when its END fits
    # in the lead rows: k lead rows hold k - 1 ids and the END, then BOS on PAD.
    return TextStreamSpec(
        prompt_ids=lead + 1,
        min_ids=max(0, lead - 1),
        id_limit=int(text_vocab_size) if text_vocab_size else None,
    )
