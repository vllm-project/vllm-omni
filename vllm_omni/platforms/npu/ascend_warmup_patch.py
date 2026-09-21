# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Skip the vllm-ascend Triton warmups that fault the 910_93 vector core.

vllm-ascend warms three Triton kernel families on the worker before graph
capture (``vllm_ascend/model_executor/warmup/kernel_warmup.py``). On 910C/A3
the second one, ``penalties_triton_warmup``, faults inside its dummy-token
setup before any Triton kernel runs:

    tokens[:, -1:] = vocab_size

lowers to ACLNN InplaceCopy over the legacy BroadcastTo kernel, and on
ascend910_93 that kernel raises a vector-core exception -- "The address for
the scalar to access the internal buffer of AICore is out of bounds", aclnn
error 507035 -- which kills the stage worker during initialization and takes
the whole orchestrator down with it.

The identical line runs clean on 910B3 under the same CANN 9.1.0 and
torch_npu 2.10.0.post4 (verified directly, with the same shapes), so this is
a 910_93 operator fault rather than a shape, dtype or memory problem -- the
warmup input is only ``[max_num_seqs, 257]`` token ids. Rewriting the line
would merely relocate the fault into an operator nobody can verify on the
target box, so the warmup is skipped instead.

Skipping is safe by construction: the warmup only pre-compiles Triton kernels
(JIT), and the sampling path runs those kernels over real token tensors, never
through this dummy construction. The cost is a one-off compile on the first
penalised sample of the first request. Correctness is unaffected.

Scoped so the 910B baseline keeps the stock warmup byte-for-byte: the guards
delegate to the original warmup there. The skip set is resolved when a warmup
actually runs rather than when the platform is constructed, because the
constructor can run before the device is set -- and a probe that failed there
would silently strip a warmup the baseline relies on. A probe that still
cannot name the SoC skips the faulting warmup: a missing warmup costs latency,
a faulting one costs the whole run.
``VLLM_OMNI_NPU_SKIP_WARMUPS`` overrides the set (comma separated substrings
of the warmup names, or "none" to restore stock behaviour).

``rejection_sampler_triton_warmup`` joined the default skip set after the A3
run of 02:38 pinned the acl 507035 aivec fault on it: the dummy
(batch, spec_len, vocab) sweep "completed" on the host half a second before
the pre-capture synchronize raised, and nothing else was enqueued in between
(penalties was already skipped, rms is a no-op there). Skipping is lossless
the same way: on this stack no 910C runtime path launches the reject_sample
kernels -- the Talker's K-step sample() and its SoC guards fall back to
torch argmax, so the warmup only ever compiled kernels nobody runs.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)

_PATCHED = False
_ENV = "VLLM_OMNI_NPU_SKIP_WARMUPS"
# Substrings matched against the warmup function names in kernel_warmup.py.
_DEFAULT_SKIP = ("penalties", "rejection_sampler")
_WARMUP_NAMES = (
    "rejection_sampler_triton_warmup",
    "penalties_triton_warmup",
    "triton_rms_warmup",
)


def _probe_soc_name() -> str:
    """torch_npu device name, or "" when it cannot be read yet."""
    try:
        import torch_npu

        try:
            device = torch_npu.npu.current_device()
        except Exception:
            device = 0
        return str(torch_npu.npu.get_device_name(device))
    except Exception:
        return ""


def _kstep_armed() -> bool:
    """Whether the Talker K-step is explicitly armed (frames > 1).

    Single source of truth is the same env the deploy config reads; a loader
    that cannot import it is treated as "not armed" so the baseline never
    changes by accident.
    """
    try:
        from vllm_omni.config.stage_config import talker_frames_per_step

        return talker_frames_per_step() > 1
    except Exception:
        return False


def _skipped_names() -> set[str]:
    raw = os.environ.get(_ENV, "").strip().lower()
    if raw:
        if raw in ("none", "off", "0", "false", "no"):
            return set()
        return {part.strip() for part in raw.split(",") if part.strip()}
    soc = _probe_soc_name()
    if soc.startswith("Ascend910B"):
        if _kstep_armed():
            # The torch-native sampler replaces the reject kernels whenever
            # the K-step is armed, so this warmup would only compile kernels
            # nobody runs. Everything else keeps the stock warmup.
            return {"rejection_sampler"}
        # Verified clean here; keep the stock warmup untouched.
        return set()
    return set(_DEFAULT_SKIP)


def _make_guard(name: str, original: Callable[[Any], Any]) -> Callable[[Any], Any]:
    """Run the real warmup unless this call is on a host that must skip it.

    The decision is taken per call, not at install time: the platform
    constructor can run before the device is set, and a probe that fails there
    would otherwise silently skip a warmup the 910B baseline relies on.
    """

    def _guarded(worker: Any) -> Any:
        if any(fragment in name for fragment in _skipped_names()):
            logger.info(
                "[npu] %s skipped: these Triton warmups fault the 910_93 "
                "vector core during their dummy-token setup (acl 507035), "
                "and no runtime path on this stack launches the kernels",
                name,
            )
            return None
        return original(worker)

    _guarded._vllm_omni_guarded = True  # type: ignore[attr-defined]
    return _guarded


def apply_ascend_warmup_patch() -> None:
    """Guard the ascend warmups so a faulting one can be skipped per call."""
    global _PATCHED
    if _PATCHED:
        return
    _PATCHED = True

    try:
        import importlib

        # Import the module, not the function: ``vllm_ascend.model_executor.warmup``
        # re-exports ``kernel_warmup`` as a package attribute, so
        # ``from ... import kernel_warmup`` yields the function and every
        # ``getattr`` below would silently miss.
        warmup_module = importlib.import_module("vllm_ascend.model_executor.warmup.kernel_warmup")
    except Exception as error:  # pragma: no cover - non-ascend builds
        logger.warning("[npu] Ascend warmup patch not applied: %s", error)
        return

    guarded: list[str] = []
    for name in _WARMUP_NAMES:
        original = getattr(warmup_module, name, None)
        if original is None or getattr(original, "_vllm_omni_guarded", False):
            continue
        setattr(warmup_module, name, _make_guard(name, original))
        guarded.append(name)

    if guarded:
        logger.info(
            "[npu] ascend Triton warmups guarded (%s); the skip set is resolved "
            "per call, so the SoC probe sees a live device. Override with %s.",
            ", ".join(guarded),
            _ENV,
        )
