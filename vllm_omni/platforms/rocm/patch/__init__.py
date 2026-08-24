# SPDX-License-Identifier: Apache-2.0


def apply_patches():
    """Apply all ROCm-specific patches."""
    from vllm_omni.platforms.rocm.patch import worker  # noqa: F401
