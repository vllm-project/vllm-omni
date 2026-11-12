# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Platform view of the Qwen3-Omni codec connector extras."""

from collections.abc import Mapping
from typing import Any

from vllm_omni.platforms import current_omni_platform

# Validated on CUDA only. Without them other platforms keep the fixed
# ``initial_codec_chunk_frames`` schedule and its Code2Wav graph shapes.
CUDA_ONLY_CODEC_EXTRAS = ("codec_chunk_ramp", "codec_graph_batch_sizes")


def qwen3_omni_codec_extra(extra: Mapping[str, Any]) -> dict[str, Any]:
    """Return the connector extras the Talker chunker and Code2Wav apply here."""
    if current_omni_platform.is_cuda():
        return dict(extra)
    return {key: value for key, value in extra.items() if key not in CUDA_ONLY_CODEC_EXTRAS}
