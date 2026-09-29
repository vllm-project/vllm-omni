# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from typing import Any

from vllm_omni.plugins import load_omni_general_plugins


class OmniWorkerMixin:
    """Shared Omni plugin and native KV connector setup for workers."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        load_omni_general_plugins()

    def initialize_from_config(self, kv_cache_config) -> None:
        # The concrete GPU/NPU/XPU worker supplies this method through the MRO.
        super().initialize_from_config(kv_cache_config)  # type: ignore[misc]
        from vllm.distributed.kv_transfer import get_kv_transfer_group, has_kv_transfer_group

        if has_kv_transfer_group():
            from vllm_omni.diffusion.diffusion_kv.kv_connector import install_mooncake_cfg_fanout

            install_mooncake_cfg_fanout(get_kv_transfer_group())


def maybe_unpad_input_ids(model, input_ids, num_tokens_unpadded):
    """Trim cudagraph-bucket padding for models that split a flat ``input_ids``
    by per-request ``seq_token_counts``. See #6712."""
    if input_ids is not None and getattr(model, "requires_exact_input_shape", False):
        return input_ids[:num_tokens_unpadded]
    return input_ids
