# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Lychee codec snapshots to the native response-owned Token2Wav stage."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from vllm_omni.inputs.data import OmniTokensPrompt

LYCHEE_CODEC_OFFSET = 151696
LYCHEE_CODEC_VOCAB_SIZE = 6561


def _tokens(value: Any) -> list[int]:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().flatten().tolist()
    return list(value)


def lychee2token2wav(
    source_outputs: list[Any], prompt: Any = None, requires_multimodal_data: bool = False
) -> list[OmniTokensPrompt]:
    """Convert plugin-fenced partial outputs; never infer response EOF from parking."""
    prompts: list[OmniTokensPrompt] = []
    for source in source_outputs:
        output = source.outputs[0]
        multimodal = getattr(output, "multimodal_output", None) or getattr(source, "multimodal_output", None) or {}
        metadata = multimodal.get("lychee_t2w")
        if not isinstance(metadata, Mapping):
            raise ValueError("Lychee Stage1 input requires plugin-owned lychee_t2w response metadata")
        metadata = dict(metadata)
        if "codec_token_ids" in metadata:
            codec = _tokens(metadata.pop("codec_token_ids"))
        else:
            speech = _tokens(multimodal.get("lychee_speech_token_ids", []))
            codec = [
                token - LYCHEE_CODEC_OFFSET
                for token in speech
                if LYCHEE_CODEC_OFFSET <= token < LYCHEE_CODEC_OFFSET + LYCHEE_CODEC_VOCAB_SIZE
            ]
        if any(
            isinstance(token, bool) or not isinstance(token, int) or not 0 <= token < LYCHEE_CODEC_VOCAB_SIZE
            for token in codec
        ):
            raise ValueError("Lychee native Flow checkpoint accepts codec IDs [0, 6561), not all tokenizer6656 IDs")
        if not codec and not metadata.get("final", False) and not metadata.get("cancel", False):
            continue
        metadata["empty"] = not codec
        prompts.append(OmniTokensPrompt(prompt_token_ids=codec or [0], additional_information={"lychee_t2w": metadata}))
    return prompts
