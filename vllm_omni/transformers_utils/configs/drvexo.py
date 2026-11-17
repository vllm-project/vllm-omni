# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Shared admission contracts, with model-specific trained token budgets."""

DRVEXO_MODEL_TYPES = frozenset({"drvexo_speech_8b", "drvexo_speech_qwen35_4b"})


def decision_token_limits(model_type: str) -> tuple[int, int]:
    if model_type == "drvexo_speech_qwen35_4b":
        return 4096, 384
    if model_type == "drvexo_speech_8b":
        return 1536, 384
    raise ValueError(f"Unsupported DRVEXO model type: {model_type}")
