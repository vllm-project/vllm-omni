# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2024, Jiarui Fang.
# Adapted from https://github.com/feifeibear/long-context-attention

import logging

# test if flash_attn (FA2) is available
try:
    import flash_attn  # noqa: F401
    from flash_attn.flash_attn_interface import _flash_attn_forward  # noqa: F401

    HAS_FLASH_ATTN = True
except (ImportError, ModuleNotFoundError):
    HAS_FLASH_ATTN = False

# FA3 detection: try multiple sources (forward only, no backward needed for inference)
# Source 1: flash_attn_interface (from flash-attention source build)
# Source 2: fa3_fwd_interface (from fa3-fwd PyPI package, supports Ampere/Ada/Hopper)
# Note: FA3 high-level API may or may not return softmax_lse depending on version.
#       For Ring Attention which requires LSE, we fall back to low-level API if needed.
HAS_FA3 = False
fa3_fwd_func = None  # Low-level forward function (_flash_attn_forward)
fa3_attn_func = None  # High-level attention function (flash_attn_func)

# Try flash_attn_interface first (from flash-attention source build)
try:
    from flash_attn_interface import _flash_attn_forward as fa3_fwd_func  # noqa: F401
    from flash_attn_interface import flash_attn_func as fa3_attn_func  # noqa: F401

    HAS_FA3 = True
except (ImportError, ModuleNotFoundError):
    pass

# Fallback: try fa3_fwd_interface (PyPI package, supports Ampere/Ada/Hopper)
if not HAS_FA3:
    try:
        from fa3_fwd_interface import _flash_attn_forward as fa3_fwd_func  # noqa: F401
        from fa3_fwd_interface import flash_attn_func as fa3_attn_func  # noqa: F401

        HAS_FA3 = True
    except (ImportError, ModuleNotFoundError):
        pass

# Legacy aliases for backward compatibility
HAS_FLASH_ATTN_HOPPER = HAS_FA3
flash_attn_forward_hopper = fa3_fwd_func
flash3_attn_func = fa3_attn_func

logger = logging.getLogger(__name__)

try:
    from flashinfer.prefill import single_prefill_with_kv_cache  # noqa: F401

    HAS_FLASHINFER = True
except Exception as e:
    # flashinfer may raise RuntimeError at import-time for version/binary mismatches.
    # Treat it as unavailable so the runtime can gracefully fall back to other backends.
    HAS_FLASHINFER = False
    logger.warning("FlashInfer is unavailable; falling back to other attention backends. Reason: %s", e)

try:
    import aiter  # noqa: F401
    from aiter import flash_attn_func as flash_attn_func_aiter  # noqa: F401

    HAS_AITER = True
except (ImportError, ModuleNotFoundError):
    HAS_AITER = False

try:
    import sageattention  # noqa: F401

    HAS_SAGE_ATTENTION = True
except (ImportError, ModuleNotFoundError):
    HAS_SAGE_ATTENTION = False

try:
    import spas_sage_attn  # noqa: F401

    HAS_SPARSE_SAGE_ATTENTION = True
except (ImportError, ModuleNotFoundError):
    HAS_SPARSE_SAGE_ATTENTION = False

try:
    import torch_npu  # noqa: F401

    HAS_NPU = True
except (ImportError, ModuleNotFoundError):
    HAS_NPU = False
