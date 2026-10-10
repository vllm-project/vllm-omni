# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Available selection methods, imported lazily for configuration parsing."""

from collections.abc import Mapping
from importlib import import_module

from .abstract import BlockSelector

# New methods implement BlockSelector and add their import path here.
BLOCK_SELECTORS: dict[str, str] = {
    "block_topk": "vllm_omni.diffusion.attention.block_selection.subblock_topk.SubBlockTopK",
}


def get_block_selector(name: str | None) -> type[BlockSelector]:
    if not isinstance(name, str) or name not in BLOCK_SELECTORS:
        raise ValueError(f"Unknown block selector: {name!r}; available: {sorted(BLOCK_SELECTORS)}")
    module, class_name = BLOCK_SELECTORS[name].rsplit(".", 1)
    return getattr(import_module(module), class_name)


def normalize_selection(selection=None):
    selection = {"name": "block_topk"} if selection is None else selection
    if not isinstance(selection, Mapping) or selection.keys() - {"name", "config"}:
        raise ValueError("selection requires name and optional config")
    selector = get_block_selector(selection.get("name"))
    options = selection.get("config", {})
    if not isinstance(options, Mapping):
        raise ValueError("selection.config must be a mapping")
    return {"name": selection["name"], "config": selector.normalize_config(dict(options))}
