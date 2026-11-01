# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_code2wav import Qwen3TTSCode2Wav


class BreezeCode2Wav(Qwen3TTSCode2Wav):
    """Breeze bundles the same Qwen3 codec under audio_tokenizer/."""

    tokenizer_subfolder = "audio_tokenizer"
    # Reference conditioning is consumed by the AR stage. The codec receives
    # generated codes only, so it never decodes an ICL reference prefix.
    decoder_cudagraph_modes = ("xvec",)
