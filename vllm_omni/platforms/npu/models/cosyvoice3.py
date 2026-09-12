# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CosyVoice3 spectral transforms on Ascend 950 (A5)."""

from vllm_omni.platforms.npu import is_a5


def apply_cosyvoice3_patches() -> None:
    """Use CPU STFT/ISTFT only on the A5 platform, which lacks native STFT."""
    if not is_a5():
        return

    from vllm_omni.model_executor.models.cosyvoice3.code2wav_core.hifigan import HiFTGenerator

    # CausalHiFTGenerator inherits both spectral transforms and this flag.
    HiFTGenerator._stft_on_cpu = True
