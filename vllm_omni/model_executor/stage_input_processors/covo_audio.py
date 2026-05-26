# Copyright 2026 Tencent.
from typing import Any

from vllm_omni.inputs.data import OmniTokensPrompt
from vllm_omni.model_executor.models.covo_audio.config_covo_audio import COVO_AUDIO_TOKEN_INDEX


def llm2code2wav(
    source_outputs: list[Any],
    prompt: Any = None,
    requires_multimodal_data: bool = False,
) -> list[OmniTokensPrompt]:
    talker_outputs = source_outputs
    code2wav_inputs = []

    for i, talker_output in enumerate(talker_outputs):
        output = talker_output.outputs[0]
        token_ids = output.token_ids

        audio_codes = [t - COVO_AUDIO_TOKEN_INDEX for t in token_ids if t >= COVO_AUDIO_TOKEN_INDEX]

        if not audio_codes:
            audio_codes = [-1]

        code2wav_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=audio_codes,
                multi_modal_data=None,
                mm_processor_kwargs=None,
            )
        )

    return code2wav_inputs
