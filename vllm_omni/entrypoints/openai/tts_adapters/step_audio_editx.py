# SPDX-License-Identifier: Apache-2.0
"""StepAudio-Editx serving adapter."""

from typing import TYPE_CHECKING

from vllm.logger import init_logger

from vllm_omni.entrypoints.openai.tts_adapters import register_tts_adapter
from vllm_omni.entrypoints.openai.tts_adapters.base import ARTTSAdapter, PreparedRequest

if TYPE_CHECKING:
    from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest

logger = init_logger(__name__)


@register_tts_adapter
class StepAudioEditxTTSAdapter(ARTTSAdapter):
    """Adapter for StepAudio-EditX (AR ``engine_client`` backend)."""

    stage_keys = frozenset({"step_audio_editx_ar"})
    name = "step_audio_editx"

    def validate(self, request: "OpenAICreateSpeechRequest") -> str | None:
        return self.ctx.server._validate_tts_request(request)

    async def build(
        self, request: "OpenAICreateSpeechRequest", sampling_params_list: list, has_inline_ref_audio: bool
    ) -> PreparedRequest:
        prompt = self.ctx.server._build_step_audio_editx_prompt(request)
        return PreparedRequest(
            prompt=prompt,
            tts_params={},
            model_type="step_audio_editx",
        )
