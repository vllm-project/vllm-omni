# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Reuse native speech execution with an explicit different-vocabulary edge."""

from vllm_omni.model_executor.models.drvexo_speech_8b.lexical import RetokenizedTextRouter
from vllm_omni.model_executor.models.drvexo_speech_8b.talker import DRVEXOSpeech8BTalker


class DRVEXOUnifiedTalker(DRVEXOSpeech8BTalker):
    def __init__(self, *, vllm_config, prefix=""):
        if getattr(vllm_config.model_config.hf_config, "drvexo_output_conditioning", None) != "generated_tokens":
            raise ValueError("4B speech requires generated text; the legacy hidden bridge is not compatible")
        super().__init__(vllm_config=vllm_config, prefix=prefix)

    def _make_lexical_router(self, source_tokenizer, target_tokenizer):
        return RetokenizedTextRouter(source_tokenizer, target_tokenizer)
