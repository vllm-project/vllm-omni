# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Thin websocket serving for engine-resident duplex sessions.

The session logic lives in ``vllm_omni.engine.duplex`` and is reached through
``vllm_omni.entrypoints.duplex_omni.DuplexOmni``; this package only speaks the
OpenAI Realtime wire protocol over ``/v1/realtime?duplex=1``.
"""

from .serving import OmniDuplexSessionHandler

__all__ = ["OmniDuplexSessionHandler"]
