# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vllm_omni.lora.inventory import LoRAAdapterInfo


class OmniWorkerMixin:
    """Shared Omni plugin and native KV connector setup for workers."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        from vllm_omni.plugins import load_omni_general_plugins

        load_omni_general_plugins()

    def initialize_from_config(self, kv_cache_config) -> None:
        # The concrete GPU/NPU/XPU worker supplies this method through the MRO.
        super().initialize_from_config(kv_cache_config)  # type: ignore[misc]
        from vllm.distributed.kv_transfer import get_kv_transfer_group, has_kv_transfer_group

        if has_kv_transfer_group():
            from vllm_omni.diffusion.diffusion_kv.kv_connector import install_mooncake_cfg_fanout

            install_mooncake_cfg_fanout(get_kv_transfer_group())

    def list_lora_adapters(self) -> list[LoRAAdapterInfo]:
        """Return runtime adapter identities for AR worker inventory queries."""
        engine = getattr(self, "weight_transfer_engine", None)
        manager = getattr(engine, "lora_manager", None)
        if manager is not None:
            list_metadata = getattr(manager, "list_adapter_metadata", None)
            if callable(list_metadata):
                return list_metadata()
        if getattr(getattr(self, "vllm_config", None), "lora_config", None) is None:
            return []
        list_loras = getattr(self, "list_loras", None)
        if not callable(list_loras):
            return []
        return [{"id": adapter_id, "name": None} for adapter_id in sorted(list_loras())]


def maybe_unpad_input_ids(model, input_ids, num_tokens_unpadded):
    """Trim cudagraph-bucket padding for models that split a flat ``input_ids``
    by per-request ``seq_token_counts``. See #6712."""
    if input_ids is not None and getattr(model, "requires_exact_input_shape", False):
        return input_ids[:num_tokens_unpadded]
    return input_ids
