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

The identical line runs clean on 910B3 on the same CANN and torch_npu stack,
so this is a 910_93 operator fault rather than a shape, dtype or memory
problem -- the warmup input is only ``[max_num_seqs, 257]`` token ids.
Rewriting the line would merely relocate the fault into an operator nobody
can verify on the target box, so the warmup is skipped instead.

Skipping is safe by construction: the warmup only pre-compiles Triton kernels
(JIT), and the sampling path runs those kernels over real token tensors, never
through this dummy construction. The cost is a one-off compile on the first
penalised sample of the first request. Correctness is unaffected.

Scoped per family: 910B3 skips only ``rejection_sampler`` -- its dummy
``(batch, spec_len, vocab)`` sweep faults the vector core before any kernel
runs, the armed path uses the torch-native sampler anyway, and the stock path
merely JITs on its first request. Every other warmup stays stock there. The
skip set is resolved when a warmup actually runs rather than when the platform
is constructed, because the constructor can run before the device is set --
and a probe that failed there would silently strip a warmup the baseline
relies on. A probe that still cannot name the SoC skips the faulting warmup: a
missing warmup costs latency, a faulting one costs the whole run.

``rejection_sampler_triton_warmup`` is in the default skip set for the same
reason: its dummy (batch, spec_len, vocab) sweep faults the 910_93 vector
core, and the fault surfaces at the pre-capture synchronize rather than
inside the sweep. Skipping is lossless here in the same way as above: on this
stack no 910C runtime path launches the reject_sample kernels, because the
Talker's K-step sample() and its SoC guards fall back to torch argmax.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)

_PATCHED = False
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


def _kstep_armed(vllm_config: Any = None) -> bool:
    """Whether this worker runs the Talker multi-frame decode.

    Callers inside the worker (the model runner) pass the config they were
    built with, which is the only source that is guaranteed to be populated:
    the engine's ``set_current_vllm_config`` context wraps vLLM's own
    ``load_model`` call, so by the time a caller sits *after*
    ``model_runner.load_model()`` the context has already exited and
    ``get_current_vllm_config_or_none()`` reads None. A context lookup is kept
    as a fallback for callers that run inside that window.
    """
    cfg = vllm_config
    if cfg is None:
        try:
            from vllm.config import get_current_vllm_config_or_none

            cfg = get_current_vllm_config_or_none()
        except Exception:
            cfg = None
    spec = getattr(cfg, "speculative_config", None) if cfg is not None else None
    if spec is None:
        return False
    method = getattr(spec, "method", None)
    num_spec = getattr(spec, "num_speculative_tokens", 0) or 0
    return method == "ngram" and num_spec > 0


def _skipped_names() -> set[str]:
    soc = _probe_soc_name()
    if soc.startswith("Ascend910B"):
        # This warmup faults the vector core (acl 507035) on this family, and the
        # worker cannot tell at warmup time whether the K-step is armed (spawned
        # process, engine config not handed over yet). Skip either way: the armed
        # path uses the torch-native sampler, the stock path only JITs the kernel
        # on its first request.
        return {"rejection_sampler"}
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


def _apply_contiguous_kv_patch() -> None:
    """Force the contiguous KV layout for the K-step Talker's layers.

    vllm-ascend's block-major strided allocation regressed this model's
    static-shape decode scan by ~19x (12.7 ms per frame vs 0.66 ms on the
    previous build). The upstream fix that keeps K/V contiguous for paged
    attention (``requires_contiguous_pa_kv_cache``) explicitly excludes
    speculative configs, and the K-step Talker runs under one, so its layers
    kept landing in the strided set. The strided layout only pays off for
    paged attention; the one-query FIA graph reads every KV slot of its
    bucket each frame, so it only ever pays for the stride indirection.

    Patched on the consumer module: ``model_runner_v1`` binds the function
    with a from-import, so patching ``vllm_ascend.attention.utils`` alone
    would not be seen at the call site. Gated on the K-step config so any
    other worker (stage 0, non-speculative deployments) keeps its stock
    layout choice.
    """
    try:
        import importlib

        runner_module = importlib.import_module("vllm_ascend.worker.model_runner_v1")
    except Exception as error:  # pragma: no cover - non-ascend builds
        logger.warning("[npu] contiguous KV patch not applied: %s", error)
        return

    original = getattr(runner_module, "requires_contiguous_pa_kv_cache", None)
    if original is None or getattr(original, "_vllm_omni_contiguous", False):
        return

    def _contiguous_for_kstep(layer, vllm_config, spec, *args, **kwargs):
        try:
            # Reproduce the original guard verbatim minus the one condition
            # this patch overrides (using_paged_attention, which is
            # unconditionally False under speculative configs).
            from vllm_ascend.attention.attention_v1 import AscendAttentionBackendImpl

            impl = getattr(layer, "impl", None)
            armed = _kstep_armed(vllm_config)
            # isinstance, not type() is: the K-step Talker layers run the
            # OmniStaticShapeAttentionBackendImpl subclass of the ascend
            # backend, and the upstream strict-type check would exclude them.
            backend_ok = isinstance(impl, AscendAttentionBackendImpl)
            sliding = getattr(impl, "sliding_window", "MISSING")
            runner = getattr(vllm_config.model_config, "runner_type", None)
            page_eq = spec.page_size_bytes == spec.real_page_size_bytes
            passed = backend_ok and sliding is None and runner != "pooling" and page_eq
            logger.info(
                "[npu] contiguous KV probe: layer=%s armed=%s impl=%s backend_ok=%s "
                "sliding_window=%s runner=%s page_eq=%s -> return=%s",
                getattr(layer, "layer_name", "?"),
                armed,
                type(impl).__name__,
                backend_ok,
                sliding,
                runner,
                page_eq,
                bool(armed and passed),
            )
            if armed and passed:
                return True
        except Exception:
            logger.exception("[npu] contiguous KV probe failed")
        return original(layer, vllm_config, spec, *args, **kwargs)

    _contiguous_for_kstep._vllm_omni_contiguous = True  # type: ignore[attr-defined]
    runner_module.requires_contiguous_pa_kv_cache = _contiguous_for_kstep  # type: ignore[attr-defined]
    logger.info(
        "[npu] contiguous KV patch applied: the K-step Talker's layers leave "
        "the strided block-major set (its one-query FIA decode reads the "
        "whole bucket every frame and only loses on the indirection)",
    )


def apply_ascend_warmup_patch() -> None:
    """Guard the ascend warmups so a faulting one can be skipped per call."""
    global _PATCHED
    if _PATCHED:
        return
    _PATCHED = True

    _apply_contiguous_kv_patch()

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
            "per call, so the SoC probe sees a live device.",
            ", ".join(guarded),
        )
