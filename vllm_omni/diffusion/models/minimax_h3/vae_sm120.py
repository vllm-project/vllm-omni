# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Port of SGLang's SM12x fix for the MiniMax-H3 video VAE decoder: avoid the fused-bias epilogue.

Reference: sgl-project/sglang `python/sglang/multimodal_gen/runtime/models/vaes/minimax_h3_video_vae/`
(`base_module.py::_unfused_bias_linear`, `_is_sm120`) and `vae_vit.py::prepare_decoder_autocast_weights`.

Why
---
Our checkpoint's decoder `FeedForward` calls `self.w2(hidden_states)` - an `addmm`, i.e. a GEMM with a
fused bias epilogue. On compute capability 12.x that epilogue makes cuBLAS pick a 16x16 wmma kernel:
SGLang measured 14 TFLOPS fused versus 76-91 TFLOPS with the bias added separately (4.0 ms -> 0.8 ms
per `w2` call, ~3780 calls per decode). The target part is SM120 (capability 12.0).

vLLM-Omni's own fused VAE ops cannot help: `ops/vae/dispatch.py`'s `H3_VAE_OPERATOR_TABLE` covers
sm90/sm100/sm103 only, so `resolve_h3_vae_operators()` returns None on sm12x and
`install_h3_vae_optimizations()` returns False - which is also why the fp16 block-linear persistence
that function performs elsewhere never happens here.

Modes (`H3_VAE_SM120`; unset or unparsable always means stock):
  0  stock (wrappers may be installed but always delegate)
  1  unfused `w2`: `matmul(x, w.t()) + bias` on CUDA when dtypes line up
  2  mode 1 plus one-time fp16 materialization of the decoder-block Linear weights

Discovery is name-agnostic on purpose: the previous version assumed `decoder.transformer_blocks[*].ff.w2`
and failed silently. This version walks every submodule whose class is named `FeedForward` (or that has
both `w1` and `w2`), and logs unconditionally what it found, so a failed install can never be silent.

Numerics: mode 1 is the same product in the same dtype with the bias added separately (close to, not
bit-identical with, the fused form); mode 2 additionally changes weight storage dtype. Eye-gate clips.
"""

from __future__ import annotations

import os
import weakref

import torch

try:  # vLLM provides the logger in-container; plain logging keeps this module testable outside it
    from vllm.logger import init_logger

    logger = init_logger(__name__)
except Exception:  # pragma: no cover
    import logging

    logger = logging.getLogger(__name__)

# Opt-in via H3_VAE_SM120; no machine-specific default.
_LOGGED_MODES: set[int] = set()
_OWNER_OF: dict[int, weakref.ReferenceType[torch.nn.Module]] = {}
_LINEAR_ATTRS = ("to_qkv", "to_out", "w1", "w2")


def _is_sm12x(device) -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        if isinstance(device, str):
            device = torch.device(device)
        if device is not None and getattr(device, "type", None) not in (None, "cuda"):
            # A non-CUDA target (cpu/meta) at load time: ask the current device instead.
            if getattr(device, "type", None) != "cuda":
                device = None
        major, _ = torch.cuda.get_device_capability(None if device is None else device)
    except (RuntimeError, AssertionError, TypeError, ValueError):
        return False
    return major == 12


def _resolve_mode() -> int:
    """Decoder mode from ``H3_VAE_SM120`` (0 = untouched, 1/2 = the two fix sets)."""
    try:
        mode = int(os.environ.get("H3_VAE_SM120", "0"))
    except ValueError:
        mode = 0
    return mode if mode in (1, 2) else 0


def _find_feed_forwards(root: torch.nn.Module) -> list[torch.nn.Module]:
    """Every submodule that looks like the decoder's FeedForward, by class name or by shape."""
    found: list[torch.nn.Module] = []
    seen: set[int] = set()
    stack = [root]
    while stack:
        module = stack.pop()
        if id(module) in seen:
            continue
        seen.add(id(module))
        looks_like = type(module).__name__ == "FeedForward" or (hasattr(module, "w1") and hasattr(module, "w2"))
        if looks_like and isinstance(getattr(module, "w2", None), torch.nn.Linear):
            found.append(module)
        try:
            stack.extend(module.children())
        except RecursionError:  # paranoia: never let discovery kill a model load
            continue
    return found


def _materialize_fp16(root: torch.nn.Module) -> int:
    changed = 0
    for module in root.modules():
        for attr in _LINEAR_ATTRS:
            linear = getattr(module, attr, None)
            if isinstance(linear, torch.nn.Linear) and linear.weight is not None:
                if linear.weight.dtype != torch.float16:
                    with torch.no_grad():
                        linear.weight.data = linear.weight.data.to(torch.float16)
                        if linear.bias is not None:
                            linear.bias.data = linear.bias.data.to(torch.float16)
                    changed += 1
    return changed


def note_engagement(owner: torch.nn.Module | None) -> None:
    mode = _resolve_mode()
    if mode in _LOGGED_MODES:
        return
    if mode >= 2 and owner is not None and not getattr(owner, "_h3_vae_sm120_fp16_done", True):
        changed = _materialize_fp16(owner)
        owner._h3_vae_sm120_fp16_done = True  # type: ignore[attr-defined]
        logger.info("[h3_vae_sm120] mode 2: materialized %d decoder block Linear weights in fp16", changed)
    _LOGGED_MODES.add(mode)
    logger.info("[h3_vae_sm120] engaged mode=%d", mode)


class _UnfusedBiasLinear(torch.nn.Module):
    """`nn.Linear` whose forward avoids the fused-bias epilogue on sm12x, else delegates."""

    def __init__(self, linear: torch.nn.Linear, owner: torch.nn.Module | None) -> None:
        super().__init__()
        self.weight = linear.weight
        self.bias = linear.bias
        self._stock = linear
        # NOTE: the owner is deliberately *not* stored as a child module (module-graph cycle).
        if owner is not None:
            _OWNER_OF[id(self)] = weakref.ref(owner)
        self._sm12x = _is_sm12x(getattr(linear.weight, "device", None))

    def _owner(self) -> torch.nn.Module | None:
        ref = _OWNER_OF.get(id(self))
        return ref() if ref is not None else None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if (
            self._sm12x
            and hidden_states.is_cuda
            and self.bias is not None
            and hidden_states.dtype == self.weight.dtype
            and _resolve_mode() >= 1
        ):
            note_engagement(self._owner())
            return torch.matmul(hidden_states, self.weight.t()) + self.bias
        return self._stock(hidden_states)


def install_vae_sm120_fixes(target, *, device=None) -> bool:
    """Install the sm12x unfused-w2 wrappers on a VAE adapter or a decoder module. Always logs."""
    if target is None:
        logger.info("[h3_vae_sm120] skip: no target passed")
        return False

    # Accept either the vLLM-Omni adapter (has .model) or the decoder module itself.
    root = getattr(target, "model", None) if not isinstance(target, torch.nn.Module) else target
    if root is None:
        root = getattr(target, "remote", None)
        root = getattr(root, "model", None) if root is not None else None
    if root is None and isinstance(target, torch.nn.Module):
        root = target
    if root is None:
        logger.info("[h3_vae_sm120] skip: could not resolve a module root from %s", type(target).__name__)
        return False

    resolved_device = device if device is not None else getattr(root, "device", None)
    if not _is_sm12x(resolved_device):
        logger.info("[h3_vae_sm120] skip: not sm12x (device=%s)", resolved_device)
        return False
    if getattr(root, "_h3_vae_sm120_installed", False):
        return True

    feed_forwards = _find_feed_forwards(root)
    wrapped = 0
    for module in feed_forwards:
        linear = module.w2  # type: ignore[attr-defined]
        if not isinstance(linear, _UnfusedBiasLinear):
            module.w2 = _UnfusedBiasLinear(linear, module)  # type: ignore[attr-defined]
            wrapped += 1

    root._h3_vae_sm120_installed = True  # type: ignore[attr-defined]
    root._h3_vae_sm120_fp16_done = False  # type: ignore[attr-defined]
    logger.info(
        "[h3_vae_sm120] installed: %d/%d FeedForward w2 wrappers on %s (device=%s, control=%s, mode 0 = inert)",
        wrapped,
        len(feed_forwards),
        type(root).__name__,
        resolved_device,
        "<env H3_VAE_SM120>",
    )
    return wrapped > 0


__all__ = ["install_vae_sm120_fixes"]
