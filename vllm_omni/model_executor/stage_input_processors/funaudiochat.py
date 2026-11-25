# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""FunAudioChat codec-token transfer to the CosyVoice3 decoder stage."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from vllm.logger import init_logger

from vllm_omni.data_entry_keys import CodesStruct, MetaStruct, OmniPayloadStruct
from vllm_omni.inputs.data import OmniTokensPrompt

logger = init_logger(__name__)

_CODEC_VOCAB_SIZE = 6561
_ASYNC_STATE_KEY = "_funaudiochat_async_chunk_state"


def _codec_tokens(value: Any, vocab_size: int) -> list[int]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)) and any(isinstance(item, (list, tuple, torch.Tensor)) for item in value):
        return [token for item in value for token in _codec_tokens(item, vocab_size)]
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value, dtype=torch.long)
    value = value.detach().to(device="cpu", dtype=torch.long)
    if value.ndim == 2:
        value = value[(value >= 0).any(dim=-1)]
    value = value.reshape(-1)
    return [int(token) for token in value.tolist() if 0 <= int(token) < vocab_size]


def _read_codec_tokens(multimodal_output: Any, vocab_size: int) -> tuple[list[int], bool]:
    if not isinstance(multimodal_output, Mapping):
        return [], False
    tokens = multimodal_output.get("audio_token_ids")
    if tokens is None:
        tokens = multimodal_output.get("speech_ids")
    if tokens is None:
        return [], False
    is_snapshot = (
        isinstance(tokens, (list, tuple)) and len(tokens) > 1 and any(isinstance(item, torch.Tensor) for item in tokens)
    )
    return _codec_tokens(tokens, vocab_size), is_snapshot


def _chunk_config(transfer_manager: Any) -> tuple[int, int, int]:
    connector = getattr(transfer_manager, "connector", None)
    raw_config = getattr(connector, "config", {}) or {}
    extra = raw_config.get("extra", raw_config) if isinstance(raw_config, dict) else {}
    chunk_frames = int(extra.get("codec_chunk_frames", 25))
    lookahead_frames = int(extra.get("codec_pre_lookahead_frames", 3))
    vocab_size = int(extra.get("codec_vocab_size", _CODEC_VOCAB_SIZE))
    if chunk_frames <= 0 or lookahead_frames < 0 or vocab_size <= 0:
        raise ValueError(
            "Invalid FunAudioChat codec chunk configuration: "
            f"codec_chunk_frames={chunk_frames}, "
            f"codec_pre_lookahead_frames={lookahead_frames}, "
            f"codec_vocab_size={vocab_size}"
        )
    return chunk_frames, lookahead_frames, vocab_size


def funaudiochat2code2wav_async_chunk(
    transfer_manager: Any,
    multimodal_output: Mapping[str, Any] | None,
    request: Any,
    is_finished: bool = False,
    **_: Any,
) -> OmniPayloadStruct | None:
    """Create a bounded codec-token prefix for the next decoder step.

    The transfer adapter invokes this hook on its background save thread, so
    connector I/O does not block the AR worker. ``audio_token_ids`` is a
    per-step delta produced by FunAudioChat's speech sidecar. The emitted
    ``codes.audio`` tensor is a cumulative prefix; ``left_context_size`` tells
    the CosyVoice3 decoder which prefix has already been synthesized.
    """
    chunk_frames, lookahead_frames, vocab_size = _chunk_config(transfer_manager)
    request_id = str(request.external_req_id)
    finished = bool(is_finished or (callable(getattr(request, "is_finished", None)) and request.is_finished()))

    request_payload = transfer_manager.request_payload
    request_state = request_payload.setdefault(request_id, {})
    if not isinstance(request_state, dict):
        raise TypeError(f"Unexpected FunAudioChat transfer state for request {request_id!r}")
    state = request_state.setdefault(
        _ASYNC_STATE_KEY,
        {"tokens": [], "emitted_tokens": 0, "chunk_seq": 0, "terminal_sent": False},
    )
    if state["terminal_sent"]:
        return None

    new_tokens, is_snapshot = _read_codec_tokens(multimodal_output, vocab_size)
    if (
        is_snapshot
        and len(new_tokens) >= len(state["tokens"])
        and new_tokens[: len(state["tokens"])] == state["tokens"]
    ):
        state["tokens"] = new_tokens
    else:
        state["tokens"].extend(new_tokens)
    tokens: list[int] = state["tokens"]
    emitted_tokens = int(state["emitted_tokens"])
    available = len(tokens) - emitted_tokens
    if not finished and available < chunk_frames + lookahead_frames:
        return None

    if finished:
        prefix_end = len(tokens)
        state["emitted_tokens"] = len(tokens)
        state["terminal_sent"] = True
    else:
        prefix_end = emitted_tokens + chunk_frames + lookahead_frames
        state["emitted_tokens"] = emitted_tokens + chunk_frames

    payload = OmniPayloadStruct(
        codes=CodesStruct(audio=torch.tensor(tokens[:prefix_end], dtype=torch.long)),
        meta=MetaStruct(
            finished=torch.tensor(finished, dtype=torch.bool),
            stream_finished=torch.tensor(finished, dtype=torch.bool),
            req_id=[request_id],
            chunk_seq=int(state["chunk_seq"]),
            left_context_size=emitted_tokens,
            codec_chunk_frames=prefix_end - emitted_tokens,
        ),
    )
    state["chunk_seq"] += 1
    return payload


def funaudiochat2code2wav(
    source_outputs: list[Any],
    prompt: Any = None,
    requires_multimodal_data: bool = False,
) -> list[OmniTokensPrompt]:
    """Build full codec-token prompts for non-streaming decoder execution."""
    del prompt, requires_multimodal_data
    inputs: list[OmniTokensPrompt] = []
    for source_output in source_outputs:
        if not getattr(source_output, "finished", False):
            continue
        outputs = getattr(source_output, "outputs", None)
        if not outputs:
            raise ValueError(f"FunAudioChat output {source_output.request_id!r} has no completion")
        multimodal_output = getattr(outputs[0], "multimodal_output", None)
        if multimodal_output is None:
            raise ValueError(f"FunAudioChat output {source_output.request_id!r} has no multimodal output")
        tokens, _ = _read_codec_tokens(multimodal_output, _CODEC_VOCAB_SIZE)
        inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=tokens,
                multi_modal_data=None,
                mm_processor_kwargs=None,
            )
        )
    return inputs


__all__ = ["funaudiochat2code2wav", "funaudiochat2code2wav_async_chunk"]
