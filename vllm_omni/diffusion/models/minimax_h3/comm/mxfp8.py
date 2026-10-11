# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Shim for Sol-H3's `mxfp8` module inside the ported `h3comm` package.

`comm_quant.merge_output_fp8_as_mxfp8` returns Sol-H3's `MXActivation` so the transported E4M3 bytes
can be reused directly as the next MXFP8 linear's activation (skipping a BF16 round trip and a
re-quantisation). This lane builds MXFP8 activations through its own module, so this shim exists to
keep the ported code importable now and to be pointed at the lane's real type when that path is wired
(see the lane's comm-quant port plan).

Until then the reuse path raises a clear error instead of silently producing a wrong activation.
"""

from __future__ import annotations

import os

_MX_ACTIVATION = None
try:  # this lane's MXFP8 activation type, if it exposes one under a known name
    from vllm_omni.diffusion.models.minimax_h3.b12x_mxfp8 import (
        MXActivation as _MX_ACTIVATION,  # type: ignore  # noqa: E501
    )
except Exception:  # pragma: no cover - the mapping is not wired yet
    try:
        from vllm_omni.diffusion.models.minimax_h3.mxfp8 import (
            MXActivation as _MX_ACTIVATION,  # type: ignore  # noqa: E501
        )
    except Exception:
        _MX_ACTIVATION = None


if _MX_ACTIVATION is not None:  # pragma: no cover - depends on the lane build
    MXActivation = _MX_ACTIVATION
else:

    class MXActivation:  # noqa: D101 - stand-in, see module docstring
        def __init__(self, *args, **kwargs):
            raise NotImplementedError(
                "fp8->MXFP8 reuse is not wired to this lane's MXActivation type yet; "
                "use dequantize_merge_output_fp8 for now ("
                + os.environ.get("H3COMM_MXACT_OWNER", "vllm_omni.diffusion.models.minimax_h3")
                + ")"
            )
