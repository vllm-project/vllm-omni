# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Shared building blocks for ``DuplexModelPlugin`` implementations.

Everything here depends on the duplex framework contracts
(``vllm_omni.engine.duplex.plugin``), so it is imported only from a model's
``duplex`` package, never from the turn-based stack.
"""
