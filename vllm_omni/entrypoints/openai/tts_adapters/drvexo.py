# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Caller-text entry for the shared DRVEXO Talker, including reference voices."""

from vllm_omni.entrypoints.openai.tts_adapters import register_tts_adapter
from vllm_omni.entrypoints.openai.tts_adapters.qwen3_tts import Qwen3TTSAdapter


@register_tts_adapter
class DRVEXOTTSAdapter(Qwen3TTSAdapter):
    name = "drvexo"
    stage_keys = frozenset()
    model_archs = frozenset({"DRVEXOSpeech8BTalker", "DRVEXOUnifiedTalker"})
    detect_priority = 90
    entry_stage_id = 1

    def apply_sampling_overrides(self, sampling_params_list, request, prompt=None, request_id=None):
        # Serving passes all stages indexed from zero, not an entry-stage slice.
        # Both complete DRVEXO pipelines have Talker followed by Codec.
        if len(sampling_params_list) == self.entry_stage_id:
            raise ValueError(
                "Speech output is unavailable in this DRVEXO Thinker-only deployment; "
                "use a deployment with Talker and Codec"
            )
        if len(sampling_params_list) != self.entry_stage_id + 2:
            raise ValueError("DRVEXO speech requires the full stage-0-indexed sampling list through Talker and Codec")
        # Qwen3-TTS owns the Talker budget. Keep Thinker's params independent.
        return sampling_params_list[: self.entry_stage_id] + super().apply_sampling_overrides(
            sampling_params_list[self.entry_stage_id :], request, prompt, request_id
        )

    def _build_tts_params(self, request):
        params = super()._build_tts_params(request)
        # This entry starts at Talker with explicit caller text. It cannot
        # claim the Thinker-generated token provenance required by chat edges.
        params["drvexo_fixed_reply"] = True
        return params
