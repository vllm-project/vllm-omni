"""Shared serialization helpers for omni engine request payloads."""

from __future__ import annotations

from typing import Any

from vllm.logger import init_logger

from vllm_omni.data_entry_keys import (
    OmniInputStruct,
    OmniPayloadStruct,
    to_dict,
    to_input_struct,
    to_struct,
)
from vllm_omni.engine import AdditionalInformationField

logger = init_logger(__name__)


def serialize_additional_information(
    raw_info: dict[str, Any] | OmniPayloadStruct | OmniInputStruct | None,
    *,
    log_prefix: str | None = None,
) -> AdditionalInformationField:
    """Serialize omni request metadata for EngineCore transport.

    Returns the value as one of the typed envelope variants on
    ``OmniEngineCoreRequest.additional_information``:
    ``OmniInputStruct`` / ``OmniPayloadStruct`` / None. Tagged-union dispatch
    via msgspec selects the right variant on the consumer side.
    """
    if raw_info is None:
        return None
    if isinstance(raw_info, (OmniInputStruct, OmniPayloadStruct)):
        return raw_info
    if isinstance(raw_info, dict):
        # Distinguish input-shaped vs payload-shaped dicts by field
        # membership. Input fields and payload fields are disjoint by design
        # (see ``OmniInputStruct`` / ``OmniPayloadStruct`` docstrings).
        payload_fields = set(OmniPayloadStruct.__struct_fields__)
        if any(k in payload_fields for k in raw_info):
            return to_struct(raw_info)
        return to_input_struct(raw_info)

    raise TypeError(f"Unsupported additional_information type: {type(raw_info)!r}")


def deserialize_additional_information(
    payload: AdditionalInformationField | dict,
) -> dict:
    """Deserialize an *additional_information* payload into a plain dict."""
    if payload is None:
        return {}
    if isinstance(payload, dict):
        return payload
    return to_dict(payload)
