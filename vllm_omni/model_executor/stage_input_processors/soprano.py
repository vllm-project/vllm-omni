# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Offline continuous predictor-state handoff; token IDs are not codec codes."""

from collections.abc import Mapping

import torch
from vllm.logger import init_logger

from vllm_omni.inputs.data import OmniTokensPrompt
from vllm_omni.model_executor.models.soprano.contracts import select_predictors

logger = init_logger(__name__)


def ar_to_decoder_token_only(source_outputs, prompt=None, _requires_multimodal_data=False):
    """One control slot per complete waveform; real data comes via connector."""
    return [OmniTokensPrompt(prompt_token_ids=[0]) for output in source_outputs if output.finished]


def ar_to_decoder_full_payload(transfer_manager, pooling_output, request, is_finished=True):
    """Snapshot the complete consumed trajectory, then exclude EOS predictors.

    Validation errors cross the same full-payload path explicitly, rather than
    becoming a missing payload that leaves the decoder waiting forever.
    """
    try:
        if not isinstance(pooling_output, Mapping) or not isinstance(pooling_output.get("hidden"), torch.Tensor):
            raise ValueError("Missing Soprano hidden trajectory")
        sampled = list(request.output_token_ids)
        hidden = select_predictors(pooling_output["hidden"], len(request.prompt_token_ids), sampled)
        logger.info(
            "Soprano handoff req=%s prompt=%d sampled=%d retained=%d eos=%s",
            request.req_id,
            len(request.prompt_token_ids),
            len(sampled),
            hidden.shape[0],
            sampled[-1] == 3,
        )
        return {
            "hidden_states": {"output": hidden, "output_shape": list(hidden.shape)},
            "ids": {"output": sampled},
            "meta": {"finished": True},
        }
    except (ValueError, AttributeError) as exc:
        return {"additional_information": {"soprano_error": str(exc)}, "meta": {"finished": True}}
