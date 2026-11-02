# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Lazy MiniMax-H3 component config resolution."""


def minimax_h3_transformer_subfolder(*, model: str | None, revision: str | None = None) -> str:
    from vllm_omni.model_executor.models.minimax_h3.checkpoint import minimax_h3_transformer_subfolder as resolve

    return resolve(model=model, revision=revision)
