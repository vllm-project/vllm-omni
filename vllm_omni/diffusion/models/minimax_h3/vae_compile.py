# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Preserve native checkpoint RoPE outputs across compiled invocations.

``install_compiled_rope_output_guard`` wraps the known attention binding while
leaving other checkpoint implementations unchanged.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from types import FunctionType

import torch
import torch.nn as nn


def _preserve_compiled_rope_output(original: FunctionType) -> Callable[..., torch.Tensor]:
    """Return an output-preserving wrapper for a checkpoint RoPE function.

    ``original`` must expose the checkpoint's compile selector and cache in its
    globals. The returned callable accepts the same tensor and cosine/sine pair.
    Raise ``KeyError`` if the required compile selector is missing.
    """
    namespace = original.__globals__
    get_impl = namespace["_get_apply_rotary_pos_emb_impl"]

    @wraps(original)
    def guarded(value: torch.Tensor, rotary_pos_emb: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Apply RoPE to ``value`` using the ``rotary_pos_emb`` cosine/sine pair.

        Clone the result outside the compiled region when the compiled helper
        is selected, so Q survives the subsequent K call and later CUDA Graph
        replays. Eager and fallback results retain their original identity.
        Exceptions from RoPE, helper selection, or cloning propagate.
        """
        output = original(value, rotary_pos_emb)
        if get_impl() is namespace.get("_COMPILED_APPLY_ROTARY_POS_EMB"):
            return output.clone()
        return output

    return guarded


def install_compiled_rope_output_guard(decoder: nn.Module) -> None:
    """Install the compiled RoPE output guard in a checkpoint decoder.

    ``decoder`` must have an ``nn.ModuleList`` named ``transformer_blocks`` with
    attention ``forward`` functions whose globals bind the known checkpoint
    RoPE helper. Replace that shared global binding in place, affecting all
    attention blocks that use it. Unsupported decoders and unknown or already
    guarded bindings are left unchanged, making repeated installation a no-op.
    Return ``None``; installation errors propagate.
    """
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
