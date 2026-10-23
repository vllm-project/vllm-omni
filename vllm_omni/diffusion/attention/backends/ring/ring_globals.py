# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
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

# vLLM's maintained FA2/FA3/FA4 dispatcher.
try:
    from vllm.vllm_flash_attn import flash_attn_varlen_func as vllm_flash_attn_varlen_func  # noqa: F401

    HAS_VLLM_FLASH_ATTN = True
except (ImportError, ModuleNotFoundError):
    HAS_VLLM_FLASH_ATTN = False

# Optional source-built FA3 fallback (forward only, no backward needed for inference).
# Note: FA3 high-level API may or may not return softmax_lse depending on version.
#       For Ring Attention which requires LSE, we fall back to low-level API if needed.
HAS_FA3 = False
fa3_fwd_func = None  # Low-level forward function (_flash_attn_forward)
fa3_attn_func = None  # High-level attention function (flash_attn_func)
# ``None`` means that the extension does not publish an architecture contract.
FA3_SUPPORTED_CUDA_MAJORS: frozenset[int] | None = None

# FA4 detection. The CuTe API returns LSE directly, so it can participate in
# Ring Attention's numerically stable block-wise output accumulation.
HAS_FA4 = False
fa4_attn_func = None
try:
    from flash_attn.cute import flash_attn_func as _fa4_attn_func

    fa4_attn_func = _fa4_attn_func
    HAS_FA4 = True
except Exception:
    # Optional CuTe/CUTLASS/Quack components can be importable but
    # ABI-incompatible. Treat the whole optional backend as unavailable.
    pass

# Try flash_attn_interface first (from flash-attention source build)
try:
    from flash_attn_interface import _flash_attn_forward as _fa3_fwd_func
    from flash_attn_interface import flash_attn_func as _fa3_attn_func

    fa3_fwd_func = _fa3_fwd_func
    fa3_attn_func = _fa3_attn_func
    HAS_FA3 = True
    # The source-build FA3 interface is the Hopper implementation. Importing
    # its Python module on Blackwell does not imply that an SM10x kernel exists.
    FA3_SUPPORTED_CUDA_MAJORS = frozenset({9})
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
    HAS_FLASHINFER = False
    logger.warning("FlashInfer ring kernels are unavailable. Reason: %s", e)

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
