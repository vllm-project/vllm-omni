# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Runner-side adapters for payload metadata that affects scheduling."""

from __future__ import annotations

import importlib
from typing import Any, Protocol, runtime_checkable

from vllm.logger import init_logger

from vllm_omni.data_entry_keys import OmniPayload
from vllm_omni.outputs import SchedulingMetadataUpdate

logger = init_logger(__name__)


@runtime_checkable
class SchedulingMetadataAdapter(Protocol):
    """Translate runner payloads into generic scheduler-visible effects.

    The current production consumer is the runner-owned full-payload receive
    path. Async-chunk scheduling remains owned by OmniChunkTransferAdapter.
    """

    def extract(
        self,
        payload: OmniPayload,
        *,
        model_mode: str,
    ) -> SchedulingMetadataUpdate | None: ...


class DefaultSchedulingMetadataAdapter:
    """Adapter for the established ``meta`` and ``codes.audio`` payload schema."""

    def extract(
        self,
        payload: OmniPayload,
        *,
        model_mode: str,
    ) -> SchedulingMetadataUpdate | None:
        meta = payload.get("meta") if isinstance(payload, dict) else None
        meta = meta if isinstance(meta, dict) else {}

        resize_prompt_to = self._extract_prompt_length(payload, meta)
        prompt_token_ids = self._extract_prompt_token_ids(payload) if model_mode != "ar" else None
        if resize_prompt_to is None and prompt_token_ids is None:
            return None
        return SchedulingMetadataUpdate(
            prompt_token_ids=prompt_token_ids,
            resize_prompt_to=resize_prompt_to,
        )

    @staticmethod
    def _extract_prompt_length(payload: OmniPayload, meta: dict[str, Any]) -> int | None:
        value = meta.get("next_stage_prompt_len")
        if value is None and "next_stage_prompt_len" in payload:
            logger.warning_once(
                "legacy flat 'next_stage_prompt_len' key in payload; expected 'meta.next_stage_prompt_len'"
            )
            value = payload["next_stage_prompt_len"]
        return value if isinstance(value, int) and value > 0 else None

    @classmethod
    def _extract_prompt_token_ids(cls, payload: OmniPayload) -> tuple[int, ...] | None:
        codes = payload.get("codes") if isinstance(payload, dict) else None
        value = codes.get("audio") if isinstance(codes, dict) else None
        if value is None:
            return None
        flattened = tuple(cls._flatten(value))
        return flattened or None

    @classmethod
    def _flatten(cls, value: Any) -> list[int]:
        if hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "tolist"):
            value = value.detach().cpu().tolist()
        elif hasattr(value, "tolist") and not isinstance(value, (list, tuple)):
            value = value.tolist()
        if isinstance(value, (list, tuple)):
            return [token_id for item in value for token_id in cls._flatten(item)]
        return [int(value)]


def resolve_scheduling_metadata_adapter(
    adapter: str | SchedulingMetadataAdapter | None,
) -> SchedulingMetadataAdapter:
    """Resolve a configured adapter without exposing payload keys to schedulers."""
    if adapter is None:
        return DefaultSchedulingMetadataAdapter()
    if isinstance(adapter, str):
        module_name, separator, attribute_name = adapter.rpartition(".")
        if not separator:
            raise ValueError("scheduling_metadata_adapter must be a fully qualified import path")
        adapter = getattr(importlib.import_module(module_name), attribute_name)
    if isinstance(adapter, type):
        adapter = adapter()
    if not isinstance(adapter, SchedulingMetadataAdapter):
        raise TypeError("scheduling_metadata_adapter must implement extract(payload, *, model_mode)")
    return adapter
