# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Shared runtime LoRA inventory records and normalization helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TypedDict


class LoRAAdapterInfo(TypedDict):
    """Serializable identity for one runtime-registered LoRA adapter."""

    id: int
    name: str | None


LORA_ADAPTERS_RPC = "list_lora_adapters"


def merge_lora_adapter_records(value: object) -> list[LoRAAdapterInfo]:
    """Flatten and deduplicate adapter records returned by worker RPCs.

    Worker backends may return one list per rank, while older vLLM paths may
    still return bare integer IDs. Names are retained only when all sources
    agree; conflicting names are represented as ``None`` rather than exposing
    an arbitrary rank's value.

    Args:
        value: A worker result, nested rank results, or a bare adapter ID.

    Returns:
        Adapter records sorted by adapter ID.

    Raises:
        RuntimeError: If a worker reports that the inventory RPC is unsupported.
        ValueError: If the worker returns an invalid adapter record.
    """
    names_by_id: dict[int, set[str]] = {}
    seen_ids: set[int] = set()

    def visit(item: object) -> None:
        if item is None:
            return
        if isinstance(item, bool):
            raise ValueError("LoRA inventory adapter IDs must be integers")
        if isinstance(item, int):
            seen_ids.add(item)
            return
        if isinstance(item, Mapping):
            if item.get("supported") is False:
                raise RuntimeError("LoRA inventory RPC is unsupported")
            if "id" not in item:
                raise ValueError("LoRA inventory record is missing 'id'")
            adapter_id = item["id"]
            if isinstance(adapter_id, bool) or not isinstance(adapter_id, int):
                raise ValueError("LoRA inventory adapter IDs must be integers")
            seen_ids.add(adapter_id)
            name = item.get("name")
            if name is not None:
                if not isinstance(name, str):
                    raise ValueError("LoRA inventory adapter names must be strings or null")
                names_by_id.setdefault(adapter_id, set()).add(name)
            return
        if isinstance(item, (list, tuple, set)):
            for part in item:
                visit(part)
            return
        raise ValueError(f"Unsupported LoRA inventory result type: {type(item).__name__}")

    visit(value)
    records: list[LoRAAdapterInfo] = []
    for adapter_id in sorted(seen_ids):
        names = names_by_id.get(adapter_id, set())
        records.append({"id": adapter_id, "name": next(iter(names)) if len(names) == 1 else None})
    return records


__all__ = ["LORA_ADAPTERS_RPC", "LoRAAdapterInfo", "merge_lora_adapter_records"]
