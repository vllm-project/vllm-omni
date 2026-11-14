# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Connect scheduler-owned MiniCPM-o Ngram drafts to Ascend MRv2."""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vllm.config import VllmConfig


def use_scheduler_ngram(config: VllmConfig) -> bool:
    spec = config.speculative_config
    if spec is None or spec.method != "ngram" or config.device_config.device_type != "npu":
        return False
    model = config.model_config
    if not (
        getattr(model, "use_v2_model_runner", False)
        and getattr(model, "model_stage", None) == "llm"
        and model.architecture == "MiniCPMO45OmniForConditionalGeneration"
    ):
        return False

    parallel = config.parallel_config
    if (
        config.scheduler_config.async_scheduling
        or parallel.pipeline_parallel_size != 1
        or parallel.prefill_context_parallel_size != 1
        or parallel.decode_context_parallel_size != 1
        or config.kv_transfer_config is not None
    ):
        raise ValueError("MiniCPM-o Thinker MRv2 Ngram requires synchronous scheduling without PP, CP, or KV transfer.")
    return True


class _SchedulerNgramSpeculator:
    """Satisfy MRv2 warmup and verification while the scheduler proposes tokens."""

    supports_mm_inputs = False
    draft_logits = None
    num_query_per_req = 1
    num_extra_query_per_req = 0

    def init_cudagraph_manager(self, cudagraph_mode: object) -> None:
        pass

    def capture(self) -> None:
        pass

    def prepare_watermarking(self, sampler: object, idx_mapping: object) -> None:
        pass

    def propose(self, *, dummy_run: bool = False, **kwargs: Any) -> None:
        assert dummy_run, "The scheduler owns MiniCPM-o Ngram proposals."


def scheduler_ngram_factory(factory: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(factory)
    def init_speculator(vllm_config: VllmConfig, *args: Any, **kwargs: Any) -> Any:
        if use_scheduler_ngram(vllm_config):
            return _SchedulerNgramSpeculator()
        return factory(vllm_config, *args, **kwargs)

    return init_speculator


def allow_ascend_scheduler_ngram() -> None:
    try:
        from vllm_ascend import mrv2_utils
    except ImportError:
        return

    original = mrv2_utils._v2_blacklist

    @wraps(original)
    def blacklist(config: VllmConfig) -> list[str]:
        reasons = original(config)
        if use_scheduler_ngram(config):
            return [reason for reason in reasons if reason != "ngram speculative decoding"]
        return reasons

    mrv2_utils._v2_blacklist = blacklist
