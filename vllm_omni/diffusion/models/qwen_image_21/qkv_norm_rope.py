# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Opt-in MindIE-SD preprocessing; prefix-cache ownership stays in Attention."""

from vllm.logger import init_logger

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.platforms import current_omni_platform

logger = init_logger(__name__)


def use_mindiesd_qkv(od_config: OmniDiffusionConfig, *, native_cache: bool, unquantized: bool) -> bool:
    enabled = (od_config.additional_config or {}).get("qwen21_mindiesd_qkv_fusion", False)
    if not isinstance(enabled, bool):
        raise ValueError("qwen21_mindiesd_qkv_fusion must be a bool")
    if not enabled:
        return False
    parallel = od_config.parallel_config
    reason = None
    if not current_omni_platform.is_npu():
        reason = "requires A5 NPU"
    elif not od_config.enforce_eager:
        reason = "requires enforce_eager"
    elif parallel.tensor_parallel_size != 1 or parallel.sequence_parallel_size != 1:
        reason = "requires TP1/SP1"
    elif not native_cache or not unquantized or od_config.diffusion_kv_cache_dtype not in (None, "auto"):
        reason = "requires unquantized weights, native prefix cache and native attention"
    else:
        from vllm_omni.platforms.npu import is_a5

        if not is_a5():
            reason = "requires A5 NPU"
        else:
            try:
                import mindiesd
            except ImportError:
                reason = "MindIE-SD is unavailable"
            if reason is None and not callable(getattr(mindiesd, "norm_rope_concat", None)):
                reason = "MindIE-SD norm_rope_concat Python interface is unavailable"
    if reason is not None:
        logger.warning("Qwen2.1 MindIE-SD QKV fusion disabled: %s", reason)
        return False
    return True
