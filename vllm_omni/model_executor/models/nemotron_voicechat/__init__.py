# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""``nemotron_voicechat`` Omni pipeline (NemotronDuplexH → EarTTS).

This package exposes the topology declaration
(:data:`NEMOTRON_VOICECHAT_PIPELINE`) and the subdir naming
constants used by the deploy YAML to address each component
checkpoint inside the user's wrapper directory. See :mod:`.pipeline`
for full details on the pipeline contract and the expected wrapper
directory layout.
"""

from vllm_omni.model_executor.models.nemotron_voicechat.pipeline import (
    EARTTS_SUBDIR,
    NEMOTRON_SUBDIR,
    NEMOTRON_VOICECHAT_PIPELINE,
)

__all__ = [
    "EARTTS_SUBDIR",
    "NEMOTRON_SUBDIR",
    "NEMOTRON_VOICECHAT_PIPELINE",
]
