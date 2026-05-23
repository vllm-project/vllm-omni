# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Backwards-compatible re-export of vLLM's nunchaku NVFP4 layout adapters.

The implementation moved to
`vllm/model_executor/layers/quantization/utils/svdquant_nvfp4_layout.py`
when SVDQuant landed in vLLM. The vllm-omni converter previously
imported from this module; keep the import surface stable.
"""

from vllm.model_executor.layers.quantization.utils.svdquant_nvfp4_layout import (  # noqa: F401
    _pack_nibbles,
    _unpack_nibbles,
    pack_nunchaku_qweight_fp4,
    pack_nunchaku_wscales_fp4,
    unpack_nunchaku_qweight_fp4,
    unpack_nunchaku_wscales_fp4,
)

__all__ = [
    "pack_nunchaku_qweight_fp4",
    "unpack_nunchaku_qweight_fp4",
    "pack_nunchaku_wscales_fp4",
    "unpack_nunchaku_wscales_fp4",
]
