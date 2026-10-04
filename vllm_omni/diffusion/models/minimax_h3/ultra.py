# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""One opt-in for the qualified H3 schedule; geometry remains model-owned."""

import os
from collections.abc import MutableMapping

# Explicitly keep nonselected precision/scheduling routes disabled. Enabling
# the preset must not silently accept a conflicting process environment.
ULTRA_ENV = {
    "VLLM_OMNI_FASTVIDEO_VSA_DIRECT_Q2K": "1",
    "VLLM_OMNI_H3_VSA_KERNEL": "flashinfer",
    "VLLM_OMNI_H3_ATTENTION_OVERLAP": "1",
    "VLLM_OMNI_H3_DIT_MXFP8": "1",
    "VLLM_OMNI_H3_SWIGLU_MXFP8_FUSION": "1",
    "VLLM_OMNI_H3_VSA_SAGE": "sage",
    "VLLM_OMNI_H3_VSPLIT": "1",
    "VLLM_OMNI_H3_VSPLIT_MODE": "split",
    "VLLM_OMNI_H3_MXFP8_VSPLIT": "1",
    "VLLM_OMNI_H3_LOSSLESS_GATE": "1",
    "VLLM_OMNI_H3_LOSSLESS_EARLY_Q": "1",
    "VLLM_OMNI_H3_LOSSLESS_LAYOUT_VIEW": "1",
    "VLLM_OMNI_H3_LOSSLESS_COARSE_OVERLAP": "1",
    "VLLM_OMNI_H3_O_PRODUCER_LOOKAHEAD": "1",
    "VLLM_OMNI_ULYSSES_A2A_BACKEND": "flashinfer-pcie",
    "VLLM_OMNI_FLASHINFER_ULYSSES_REQUIRE_RDMA": "1",
    "VLLM_OMNI_FLASHINFER_ULYSSES_MAX_BYTES": "175788032",
    "VLLM_OMNI_FLASHINFER_ULYSSES_QK_PRODUCER_DIRECT": "1",
    "VLLM_OMNI_FLASHINFER_ULYSSES_O_PRODUCER_DIRECT": "1",
    "VLLM_OMNI_FLASHINFER_ULYSSES_RELEASE_AFTER_DENOISE": "0",
    "VLLM_OMNI_FASTVIDEO_VSA_FUSED_TILE_PACK": "1",
    "VLLM_OMNI_FASTVIDEO_VSA_FUSED_UNTILE": "1",
    "VLLM_OMNI_FASTVIDEO_VSA_O_BUNDLE": "1",
    "VLLM_OMNI_FASTVIDEO_VSA_SKIP_SOFTMAX_THRESHOLD_SCALE_FACTOR": "0",
    "VLLM_OMNI_MINIMAX_H3_CHUNKED_CPU_MP4_OUTPUT": "1",
    "VLLM_OMNI_H3_VAE_PAIR_PIPELINE": "1",
    "VLLM_OMNI_H3_VAE_MIXED_BATCH": "1",
    "VLLM_OMNI_H3_VAE_BATCH4": "1",
    "VLLM_OMNI_H3_VAE_GATHER_OVERLAP": "1",
    "VLLM_OMNI_H3_VAE_MXFP8": "1",
    "VLLM_OMNI_H3_FULL_VAE_AUDIO_OVERLAP": "1",
    "VLLM_OMNI_MINIMAX_H3_VAE_EXACT_OPS_SM120": "1",
}


def configure_ultra(env: MutableMapping[str, str] | None = None) -> bool:
    env = os.environ if env is None else env
    value = env.get("VLLM_OMNI_H3_ULTRA", "0")
    if value not in ("0", "1"):
        raise ValueError("VLLM_OMNI_H3_ULTRA must be 0 or 1")
    if value == "0":
        return False
    conflicts = [name for name, expected in ULTRA_ENV.items() if name in env and env[name] != expected]
    if conflicts:
        raise ValueError("H3 Ultra conflicts with explicit settings: " + ", ".join(sorted(conflicts)))
    env.update(ULTRA_ENV)
    return True
