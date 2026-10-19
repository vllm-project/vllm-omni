# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Stage input processors for Ming-3.0-audio-flash."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import torch
from vllm.inputs import TextPrompt
from vllm.logger import init_logger
from vllm.tokenizers import cached_tokenizer_from_config

from vllm_omni.data_entry_keys import CodesStruct, MetaStruct, OmniPayloadStruct
from vllm_omni.engine import OmniEngineCoreRequest
from vllm_omni.engine.serialization import deserialize_additional_information
from vllm_omni.inputs.data import OmniTokensPrompt
from vllm_omni.model_executor.models.ming_flash_omni.native_text_frontend import NativeMingTextFrontend

from .ming_talker import (
    build_ming_talker_additional_info,
    build_ming_talker_inputs,
)

logger = init_logger(__name__)


def thinker2talker_token_only(
    source_outputs: list[Any],
    prompt: OmniTokensPrompt | TextPrompt | list[Any] | None = None,
    _requires_multimodal_data: bool = False,
) -> list[OmniTokensPrompt]:
    """Build Ming talker inputs after the thinker has completed its answer."""
    return build_ming_talker_inputs(source_outputs, prompt)


@dataclass
class _MingChunkState:
    frontend: NativeMingTextFrontend
    max_length: int
    scrub_markup: bool
    prefix: str = ""
    synthesis_info: dict[str, Any] | None = None


def thinker2talker_async_chunk(
    transfer_manager: Any,
    multimodal_output: dict[str, Any] | None,
    request: OmniEngineCoreRequest,
    is_finished: bool = False,
) -> OmniPayloadStruct | None:
    """Preprocess stable thinker deltas before handing fragments to the talker.

    Frontend state is local to this request and is cleaned up by the existing
    transport cleanup paths. No model-specific receiver protocol is required.
    """
    del multimodal_output
    request_id = request.external_req_id
    states = getattr(transfer_manager, "_ming3_chunk_state", None)
    if states is None:
        states = {}
        transfer_manager._ming3_chunk_state = states
    controls = deserialize_additional_information(request.additional_information)
    max_length = int(controls.get("max_text_length", 50))
    scrub_markup = controls.get("scrub_markup", True)
    if not isinstance(scrub_markup, bool):
        raise ValueError("scrub_markup must be a boolean")
    state = states.get(request_id)
    if state is None:
        state = _MingChunkState(
            frontend=NativeMingTextFrontend(max_length, scrub_markup=scrub_markup),
            max_length=max_length,
            scrub_markup=scrub_markup,
        )
        states[request_id] = state
    elif state.max_length != max_length or state.scrub_markup != scrub_markup:
        raise ValueError("max_text_length and scrub_markup cannot change within a Ming text stream")

    token_ids = list(request.all_token_ids)[len(request.prompt_token_ids) :]
    model_config = getattr(transfer_manager, "model_config", None) or getattr(transfer_manager, "config", None)
    if model_config is None and hasattr(transfer_manager, "_get_model_config"):
        model_config = transfer_manager._get_model_config()
    tokenizer = getattr(transfer_manager, "_ming3_text_tokenizer", None)
    if tokenizer is None:
        tokenizer = cached_tokenizer_from_config(model_config)
        if tokenizer is None:
            raise RuntimeError("Ming3 async chunk requires the thinker tokenizer")
        transfer_manager._ming3_text_tokenizer = tokenizer
    text = tokenizer.decode(token_ids, skip_special_tokens=True)
    if not text.startswith(state.prefix):
        raise RuntimeError("Ming3 thinker text changed after a text delta was consumed")

    # Byte-level tokens can temporarily decode to U+FFFD. Feed only stable
    # UTF-8 text; the final snapshot still flushes the frontend when empty.
    end = len(text) if is_finished else text.find("\ufffd", len(state.prefix))
    if end < 0:
        end = len(text)
    delta = text[len(state.prefix) : end]
    state.prefix = text[:end]
    fragments = state.frontend.feed(delta, final=is_finished)
    # Optional sender-side diagnostics before Talker linguistic normalization.
    if fragments and os.environ.get("MING_TEXT_FRAG_LOG") == "1":
        logger.info("Ming text fragments req=%s fragments=%r", request_id, fragments)
    if is_finished:
        states.pop(request_id, None)
    if not fragments and not is_finished:
        return None

    spoken_text = "".join(fragments)
    if state.synthesis_info is None:
        # Language-dependent prompt and automatic IP selection are locked by
        # the first speakable fragment, rather than reselected for every chunk.
        info = build_ming_talker_additional_info(spoken_text, controls)
        if fragments:
            state.synthesis_info = info
    else:
        info = dict(state.synthesis_info)
    info.update(text=spoken_text, ming_text_fragments=fragments)
    return OmniPayloadStruct(
        codes=CodesStruct(audio=torch.tensor([0] if fragments else [], dtype=torch.long)),
        additional_information=info,
        meta=MetaStruct(replace_runtime_additional_information=True),
    )


# Opt into dispatch on every decode step, even without an inter-stage payload.
# The v2 data plane also keeps the full token history for each snapshot,
# since the frontend consumes deltas from each complete generated prefix.
setattr(thinker2talker_async_chunk, "_requires_full_token_history", True)


__all__ = ["thinker2talker_async_chunk", "thinker2talker_token_only"]
