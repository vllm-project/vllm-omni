# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Restore the torch-native rejection sampler on the 910_93 part.

vllm-ascend module-patches vllm's ``rejection_sample`` (and its two helpers)
onto Triton kernels (``vllm_ascend/patch/worker/patch_rejection_sampler.py``).
On the 910_93 vector core those kernels fault: the 02:38 run pinned acl
507035 on their warmup, which is why ``rejection_sampler`` sits in the
warmup skip set. The skip moves the fault to the first request -- every
verify then JIT-compiles the kernel family on the fly, and both stages
stall there: stage 0 verifies the n-gram drafts of every decode step, and
stage 1 verifies the always-``continue`` drafts behind the K-step
block-table growth.

The torch-native implementation the patch replaces is pure tensor ops and
exact for both callers, so this module reloads vllm's own sampler source
into a fresh module object and re-points the three names at the pre-patch
functions. Scoped to the same SoC that skips the warmup; the 910B baseline
keeps the Triton kernels it has been measured with -- except when the Talker
K-step is explicitly armed there. That is the one path whose spec width the
910B kernels cannot take (aivec 507035 on every one of the 8 combos tried),
so the restore on a 910B is tied to exactly that arming, never to the SoC
alone: the stock baseline (no arming env) keeps its sampler byte-for-byte.
"""

import importlib.util
import logging

from vllm_omni.platforms.npu.ascend_warmup_patch import _kstep_armed, _probe_soc_name

logger = logging.getLogger(__name__)

_RESTORED = False
_RESTORE_SOC_PREFIXES = ("ascend910_93", "ascend910c")
# See the module docstring: on these parts the restore applies only when the
# K-step is explicitly armed, because their stock baseline runs the patched
# kernels fine at the widths it was measured with.
_KSTEP_SOC_PREFIXES = ("ascend910b",)


def _target_soc() -> bool:
    name = _probe_soc_name().strip().lower()
    if any(name.startswith(prefix) for prefix in _RESTORE_SOC_PREFIXES):
        return True
    return any(name.startswith(prefix) for prefix in _KSTEP_SOC_PREFIXES) and _kstep_armed()


def restore_native_rejection_sampler() -> None:
    """Point vllm's rejection_sampler module back at its torch functions."""
    global _RESTORED
    if _RESTORED:
        return
    _RESTORED = True  # probe once; a non-target SoC never retries
    if not _target_soc():
        return
    try:
        import vllm.v1.sample.rejection_sampler as rs
    except ImportError:  # pragma: no cover - vllm core moved the module
        return
    spec = importlib.util.spec_from_file_location(
        "_vllm_rejection_sampler_native", rs.__file__
    )
    if spec is None or spec.loader is None:  # pragma: no cover
        return
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    restored = []
    for name in ("apply_sampling_constraints", "expand_batch_to_tokens", "rejection_sample"):
        fn = getattr(native, name, None)
        if callable(fn) and getattr(rs, name, None) is not fn:
            setattr(rs, name, fn)
            restored.append(name)
    if restored:
        logger.info(
            "[npu] rejection sampler restored to the torch-native %s on "
            "%s: the Triton kernels fault this vector core (acl 507035, "
            "their warmup is already skipped) and JIT-compiling them on "
            "the first request stalls every stage mid-verify",
            ", ".join(restored),
            _probe_soc_name(),
        )
