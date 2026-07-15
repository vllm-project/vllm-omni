# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""World-model session state (RFC #4480).

Model-agnostic session state for AR-diffusion world models: a typed
``StateObject`` contract and a ``SessionStateManager`` that owns the objects
per session, plus per-model adapters that present a model's own per-session
cache over that contract. APIs may change without notice.
"""

from __future__ import annotations
