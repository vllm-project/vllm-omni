# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""vllm-omni integration for Haru-4B (higurashi-ai), a full-duplex spoken dialogue model.

Each conversation is one resumable request on a single AR stage served over the
duplex realtime WebSocket path. Adapted from the authors' vLLM-Omni plugin
(https://github.com/higurashi-ai/Haru, Apache-2.0).
"""

from vllm_omni.model_executor.models.haru.configuration_haru import (
    HaruCodecConfig,
    HaruConfig,
    HaruHeadConfig,
    HaruSpec,
)

__all__ = ["HaruCodecConfig", "HaruConfig", "HaruHeadConfig", "HaruSpec"]
