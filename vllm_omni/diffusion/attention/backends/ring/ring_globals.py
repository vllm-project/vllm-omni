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
# ``None`` means that the extension does not publish an architecture contract,
# so retain the historical import-only behavior.  The pinned fa3-fwd wheel is
# handled below because importing its Python module succeeds on Blackwell even
# though its CUDA binary only contains SM8x/SM90 kernels.
FA3_SUPPORTED_CUDA_MAJORS: frozenset[int] | None = None

# FA4 detection. The CuTe API returns LSE directly, so it can participate in
# Ring Attention's numerically stable block-wise output accumulation.
HAS_FA4 = False
fa4_attn_func = None
try:
    from flash_attn.cute import flash_attn_func as fa4_attn_func  # noqa: F401

    HAS_FA4 = True
except Exception:
    # Optional CuTe/CUTLASS/Quack components can be importable but
    # ABI-incompatible. Treat the whole optional backend as unavailable.
    pass

# Try flash_attn_interface first (from flash-attention source build)
try:
    from flash_attn_interface import _flash_attn_forward as fa3_fwd_func  # noqa: F401
    from flash_attn_interface import flash_attn_func as fa3_attn_func  # noqa: F401

    HAS_FA3 = True
    # The source-build FA3 interface is the Hopper implementation. Importing
    # its Python module on Blackwell does not imply that an SM10x kernel exists.
    FA3_SUPPORTED_CUDA_MAJORS = frozenset({9})
except (ImportError, ModuleNotFoundError):
    pass

# Secondary FA3 import path: fa3_fwd_interface PyPI package.
if not HAS_FA3:
    try:
        from importlib.metadata import PackageNotFoundError, version

        import fa3_fwd_interface
        from fa3_fwd_interface import _flash_attn_forward as fa3_fwd_func  # noqa: F401
        from fa3_fwd_interface import flash_attn_func as fa3_attn_func  # noqa: F401

        HAS_FA3 = True
        published_majors = getattr(fa3_fwd_interface, "SUPPORTED_CUDA_MAJORS", None)
        if published_majors is not None:
            FA3_SUPPORTED_CUDA_MAJORS = frozenset(int(major) for major in published_majors)
        else:
            try:
                fa3_fwd_version = version("fa3_fwd")
            except PackageNotFoundError:
                fa3_fwd_version = None
            # requirements/cuda.txt pins this wheel.  Its fat binary contains
            # SM80 and SM90 cubins, but no Blackwell kernel.  Do not apply this
            # restriction to an unknown/future build unless it publishes its
            # own supported-major metadata above.
            if fa3_fwd_version == "0.0.3":
                FA3_SUPPORTED_CUDA_MAJORS = frozenset({8, 9})
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
