# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Canonical keys for runner-owned per-request model metadata.

The GPU runner publishes these values through a model's preprocess/postprocess
``**kwargs`` compatibility seam. Keeping the keys here prevents silent drift
between the generic runner and model implementations until that seam can move
to a typed payload.
"""

from typing import Final

OMNI_REQUEST_ID_KEY: Final = "_omni_req_id"
OMNI_INPUT_TOKEN_IDS_CPU_KEY: Final = "_omni_input_token_ids_cpu"
OMNI_PROMPT_LEN_KEY: Final = "_omni_prompt_len"
OMNI_NUM_COMPUTED_TOKENS_KEY: Final = "_omni_num_computed_tokens"
OMNI_IS_PREFILL_KEY: Final = "_omni_is_prefill"

__all__ = [
    "OMNI_INPUT_TOKEN_IDS_CPU_KEY",
    "OMNI_IS_PREFILL_KEY",
    "OMNI_NUM_COMPUTED_TOKENS_KEY",
    "OMNI_PROMPT_LEN_KEY",
    "OMNI_REQUEST_ID_KEY",
]
