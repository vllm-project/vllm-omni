"""
Compatibility shim.

The MammothModa2 DiT implementation lives under `vllm_omni.diffusion` to align
with other ARDiT structured models. We keep this module path so existing
OmniModelRegistry entries (and downstream code) keep working.
"""

from vllm_omni.diffusion.models.mammoth_moda2.pipeline_mammothmoda2_dit import (  # noqa: F401
    MammothModa2DiTPipeline,
)

__all__ = [
    "MammothModa2DiTPipeline",
]
