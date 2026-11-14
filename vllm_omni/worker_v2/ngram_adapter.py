# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Enable synchronous MiniCPM-o Thinker Ngram on the Ascend MRv2 runner.

The scheduler proposes from verified request history. The worker uses the
upstream Ngram speculator interface for MRv2 initialization and verifies the
scheduler's drafts on device.
"""

from __future__ import annotations

import logging
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

logger = logging.getLogger("vllm_omni.patch.ngram")

MRV2_NGRAM_ENV = "VLLM_OMNI_MRV2_NGRAM"
MINICPMO_THINKER_ARCH = "MiniCPMO45OmniForConditionalGeneration"
NGRAM_METHODS = frozenset({"ngram", "ngram_gpu"})

_active_runner: Any | None = None
_speculator_cls: type[Any] | None = None


def mrv2_ngram_enabled() -> bool:
    """Whether the opt-in MRv2 ngram path is requested."""
    return os.environ.get(MRV2_NGRAM_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


def spec_method(spec: Any) -> str | None:
    """Read ``method`` from either a dict or a vLLM SpeculativeConfig object."""
    if isinstance(spec, dict):
        return spec.get("method")
    return getattr(spec, "method", None)


def is_ngram_spec(spec: Any) -> bool:
    return spec_method(spec) == "ngram"


def use_scheduler_ngram(vllm_config: Any) -> bool:
    """Use host proposals when the scheduler owns the verified token history."""
    if not mrv2_ngram_enabled() or not is_ngram_spec(getattr(vllm_config, "speculative_config", None)):
        return False

    model = vllm_config.model_config
    parallel = vllm_config.parallel_config
    return (
        vllm_config.device_config.device_type == "npu"
        and getattr(model, "use_v2_model_runner", False)
        and getattr(model, "model_stage", None) == "llm"
        and model.architecture == MINICPMO_THINKER_ARCH
        and not vllm_config.scheduler_config.async_scheduling
        and parallel.pipeline_parallel_size == 1
        and parallel.prefill_context_parallel_size == 1
        and parallel.decode_context_parallel_size == 1
        and vllm_config.kv_transfer_config is None
    )


def ngram_speculator_available() -> bool:
    """Whether this vLLM build has a usable V2 ngram speculator.

    ``NgramGPUSpeculator`` is Triton-based and arrived with vLLM 0.32. Older
    builds (and builds without Triton) cannot serve ngram on MRv2, so the
    caller must keep the V1 fallback.
    """
    try:
        from vllm.triton_utils import HAS_TRITON
        from vllm.v1.worker.gpu.spec_decode.ngram.speculator import NgramGPUSpeculator
    except (ImportError, AttributeError):
        return False
    return bool(HAS_TRITON) and NgramGPUSpeculator is not None


def resolve_use_v2_model_runner(
    *,
    use_v2_model_runner: bool,
    is_npu: bool,
    speculative_config: Any,
    kstep_frames: int,
    supports_scheduler_ngram: bool = False,
    ngram_enabled: bool | None = None,
    ngram_available: bool | None = None,
) -> bool:
    """Decide the runner for one stage; pure and independently testable.

    The NPU Ngram fallback covers ngram and ngram_gpu. The Talker K-step
    contract and supported Thinker opt-in use MRv2. Other methods follow
    the configured runner.
    """
    if not (use_v2_model_runner and is_npu and spec_method(speculative_config) in NGRAM_METHODS):
        return use_v2_model_runner
    if kstep_frames != 0:
        return use_v2_model_runner
    if supports_scheduler_ngram and is_ngram_spec(speculative_config):
        enabled = mrv2_ngram_enabled() if ngram_enabled is None else ngram_enabled
        available = ngram_speculator_available() if ngram_available is None else ngram_available
        if enabled and available:
            return True
    return False


@contextmanager
def ngram_runner_scope(runner: Any) -> Iterator[None]:
    """Expose the runner being constructed to the patched speculator factory.

    Ascend's ``NPUModelRunner.__init__`` calls ``init_speculator(vllm_config,
    device)`` without a runner argument, and rebuilds ``req_states`` *after*
    that call. The scope supplies the runner so the speculator can resolve the
    live request state lazily.
    """
    global _active_runner
    if not mrv2_ngram_enabled():
        yield
        return
    previous = _active_runner
    _active_runner = runner
    try:
        yield
    finally:
        _active_runner = previous


def _speculator_class() -> type[Any]:
    global _speculator_cls
    if _speculator_cls is None:
        from vllm.v1.worker.gpu.spec_decode.ngram.speculator import NgramGPUSpeculator

        class _OmniNgramV2Speculator(NgramGPUSpeculator):
            """Ngram speculator that reads Ascend's rebuilt ``req_states``.

            ``NgramGPUSpeculator.propose`` reads ``self.req_states`` at call
            time. Ascend deletes and recreates ``RequestState`` after
            ``init_speculator`` returns, so a captured reference would be
            stale. Resolve it from the runner on every access instead.
            """

            def __init__(self, vllm_config: Any, device: Any, runner: Any) -> None:
                self._omni_runner = runner
                super().__init__(vllm_config, device, None)

            @property
            def req_states(self) -> Any:
                return self._omni_runner.req_states

            @req_states.setter
            def req_states(self, value: Any) -> None:
                # Ignore NgramGPUSpeculator.__init__'s eager assignment.
                return

        _speculator_cls = _OmniNgramV2Speculator
    return _speculator_cls


def build_ngram_speculator(vllm_config: Any, device: Any, runner: Any) -> Any:
    if not ngram_speculator_available():
        raise RuntimeError(
            f"MRv2 ngram requires vLLM's Triton-based NgramGPUSpeculator; unset {MRV2_NGRAM_ENV} to use the V1 runner."
        )
    return _speculator_class()(vllm_config, device, runner)


def _patch_ascend_init_speculator() -> None:
    try:
        import vllm_ascend.worker.v2.spec_decode as ascend_spec_decode
    except ImportError:
        return
    original = getattr(ascend_spec_decode, "init_speculator", None)
    if original is None or getattr(original, "_omni_ngram", False):
        return

    # Signature differs across vllm-ascend builds: v0.30/0.32 pass
    # (vllm_config, device), while 0.33 adds req_states. Accept and forward
    # whatever the caller supplies so non-ngram methods are unaffected.
    def init_speculator(*args: Any, **kwargs: Any) -> Any:
        vllm_config = args[0] if args else kwargs.get("vllm_config")
        device = args[1] if len(args) > 1 else kwargs.get("device")
        if use_scheduler_ngram(vllm_config):
            runner = _active_runner
            if runner is None:
                raise RuntimeError(
                    "MRv2 ngram speculator needs the runner being constructed; "
                    "NPUOmniARModelRunner must wrap super().__init__ in "
                    "ngram_runner_scope()."
                )
            return build_ngram_speculator(vllm_config, device, runner)
        return original(*args, **kwargs)

    init_speculator._omni_ngram = True  # type: ignore[attr-defined]
    vars(ascend_spec_decode)["init_speculator"] = init_speculator
    # Ascend's model_runner binds ``from ... import init_speculator`` at import
    # time; replace that already-bound reference too, whichever import ran first.
    model_runner = sys.modules.get("vllm_ascend.worker.v2.model_runner")
    if model_runner is not None and vars(model_runner).get("init_speculator") is original:
        vars(model_runner)["init_speculator"] = init_speculator
    logger.info("MRv2 ngram: patched vllm-ascend V2 init_speculator.")


def _patch_ascend_v2_blacklist() -> None:
    try:
        import vllm_ascend.mrv2_utils as mrv2_utils
    except ImportError:
        return
    original = getattr(mrv2_utils, "_v2_blacklist", None)
    if original is None or getattr(original, "_omni_ngram", False):
        return

    def _v2_blacklist(vllm_config: Any) -> list[str]:
        reasons = list(original(vllm_config))
        if use_scheduler_ngram(vllm_config):
            reasons = [reason for reason in reasons if "ngram" not in str(reason).lower()]
        return reasons

    _v2_blacklist._omni_ngram = True  # type: ignore[attr-defined]
    mrv2_utils._v2_blacklist = _v2_blacklist
    logger.info("MRv2 ngram: removed ngram from vllm-ascend V2 blacklist.")


def _patch_vllm_v2_unsupported_features() -> None:
    try:
        from vllm.config import VllmConfig
    except ImportError:
        return
    original = getattr(VllmConfig, "_get_v2_model_runner_unsupported_features", None)
    if original is None or getattr(original, "_omni_ngram", False):
        return

    def _get_v2_model_runner_unsupported_features(self: Any) -> list[str]:
        reasons = list(original(self))
        if use_scheduler_ngram(self):
            reasons = [reason for reason in reasons if "ngram" not in str(reason).lower()]
        return reasons

    _get_v2_model_runner_unsupported_features._omni_ngram = True  # type: ignore[attr-defined]
    VllmConfig._get_v2_model_runner_unsupported_features = _get_v2_model_runner_unsupported_features
    logger.info("MRv2 ngram: filtered ngram from vLLM V2 unsupported features.")


def install_ascend_ngram_patches() -> None:
    """Install every adapter patch, or do nothing when the opt-in is off."""
    if not mrv2_ngram_enabled():
        return
    if not ngram_speculator_available():
        logger.warning(
            "%s is set but this vLLM build has no Triton V2 ngram speculator; keeping the V1 fallback.",
            MRV2_NGRAM_ENV,
        )
        return
    _patch_ascend_init_speculator()
    _patch_ascend_v2_blacklist()
    _patch_vllm_v2_unsupported_features()
