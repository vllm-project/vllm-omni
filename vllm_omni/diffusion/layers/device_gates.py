# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Device allowlists shared by the diffusion Triton fast paths.

``("cuda", "musa")`` is what these kernels can run on, with MUSA reached through
torchada, which provides the CUDA entry points they call. ROCm also reports
``cuda`` as its device type, so callers that must exclude it check
``torch.version.hip`` separately.
"""

TRITON_FASTPATH_DEVICE_TYPES = ("cuda", "musa")
