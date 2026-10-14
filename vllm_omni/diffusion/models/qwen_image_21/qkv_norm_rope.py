# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Opt-in MindIE-SD preprocessing; prefix-cache ownership stays in Attention."""

import torch
from vllm.logger import init_logger

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.platforms import current_omni_platform

logger = init_logger(__name__)


def concat_prefix_bnsd(prefix: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
    """Keep head-major K/V storage through prefix assembly, returning a BSND view.

    MindIE-SD norm_rope_concat already produces head-major storage. Concatenating
    its BSND views would transpose both inputs only for Attention to transpose
    them back. Each call owns its output, as the native torch.cat path does.
    """
    return torch.cat((prefix.transpose(1, 2), current.transpose(1, 2)), dim=2).transpose(1, 2)


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
