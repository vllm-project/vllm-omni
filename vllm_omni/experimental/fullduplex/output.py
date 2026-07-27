# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from typing import Any

from vllm_omni.outputs import OmniRequestOutput

DUPLEX_OUTPUT_DECISION_KEY = "_vllm_omni.experimental.fullduplex.duplex_output_decision"


def attach_duplex_output_decision(
    output: OmniRequestOutput,
    decision: object,
) -> OmniRequestOutput:
    custom_output = dict(output._custom_output)
    custom_output[DUPLEX_OUTPUT_DECISION_KEY] = decision
    output._custom_output = custom_output
    return output


def get_duplex_output_decision(output: object) -> Any | None:
    if not isinstance(output, OmniRequestOutput):
        return None
    custom_output = output._custom_output
    if not isinstance(custom_output, dict):
        return None
    return custom_output.get(DUPLEX_OUTPUT_DECISION_KEY)


__all__ = [
    "DUPLEX_OUTPUT_DECISION_KEY",
    "attach_duplex_output_decision",
    "get_duplex_output_decision",
]
