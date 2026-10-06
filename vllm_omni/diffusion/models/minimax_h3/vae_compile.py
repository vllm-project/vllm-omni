# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Preserve checkpoint RoPE outputs across separate compiled invocations."""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from types import FunctionType

import torch
import torch.nn as nn


def _preserve_compiled_rope_output(original: FunctionType) -> Callable[..., torch.Tensor]:
    namespace = original.__globals__
    get_impl = namespace["_get_apply_rotary_pos_emb_impl"]

    @wraps(original)
    def guarded(value: torch.Tensor, rotary_pos_emb: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        output = original(value, rotary_pos_emb)
        if get_impl() is namespace.get("_COMPILED_APPLY_ROTARY_POS_EMB"):
            # Q must survive the next invocation for K. Clone outside the
            # compiled region so CUDA Graph replay cannot overwrite its storage.
            return output.clone()
        return output

    return guarded


def install_compiled_rope_output_guard(decoder: nn.Module) -> None:
    """Guard the known remote-code binding once; leave other implementations alone."""
    blocks = getattr(decoder, "transformer_blocks", None)
    if not isinstance(blocks, nn.ModuleList):
        return
    for block in blocks:
        forward = getattr(type(getattr(block, "attn", None)), "forward", None)
        attention_globals = getattr(forward, "__globals__", {})
        original = attention_globals.get("apply_rotary_pos_emb")
        namespace = getattr(original, "__globals__", {})
        if "_COMPILED_APPLY_ROTARY_POS_EMB" in namespace and callable(namespace.get("_get_apply_rotary_pos_emb_impl")):
            attention_globals["apply_rotary_pos_emb"] = _preserve_compiled_rope_output(original)
