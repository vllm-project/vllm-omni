# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from typing import Any


class OmniWorkerMixin:
    """Mixin to ensure Omni plugins are loaded in worker processes."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        from vllm_omni.plugins import load_omni_general_plugins

        load_omni_general_plugins()


def maybe_unpad_input_ids(model, input_ids, num_tokens_unpadded):
    """Trim cudagraph-bucket padding for models that split a flat ``input_ids``
    by per-request ``seq_token_counts``. See #6712."""
    if input_ids is not None and getattr(model, "requires_exact_input_shape", False):
        return input_ids[:num_tokens_unpadded]
    return input_ids
