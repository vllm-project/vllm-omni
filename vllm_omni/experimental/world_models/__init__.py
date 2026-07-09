# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""World-model session memory (RFC #4480).

Model-agnostic session memory for AR-diffusion world models: a typed
``MemoryObject`` contract and a ``SessionMemoryManager`` that owns the objects
per session, plus per-model adapters that present a model's own per-session
cache over that contract. APIs may change without notice.
"""

from __future__ import annotations
