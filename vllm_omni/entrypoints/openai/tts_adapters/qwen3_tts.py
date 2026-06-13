# SPDX-License-Identifier: Apache-2.0
"""Qwen3-TTS serving adapter.

First model migrated onto the adapter framework (RFC #4327). It is the dispatch
surface for Qwen3-TTS: the serving orchestrator routes prompt/param building
through ``build()`` instead of the inline ``else`` branch in
``_prepare_speech_generation``.

To stay behaviour-identical and avoid divergence from the batch path (which
shares ``_validate_qwen_tts_request`` and ``_build_tts_params``), the adapter
reuses the single-source helper implementations on the serving instance through
``ctx.server`` rather than copying them. Subsequent PRs relocate those bodies as
more models migrate and the batch path also routes through adapters.
"""

from typing import TYPE_CHECKING

from vllm.logger import init_logger

from vllm_omni.entrypoints.openai.tts_adapters import register_tts_adapter
from vllm_omni.entrypoints.openai.tts_adapters.base import ARTTSAdapter, PreparedRequest

if TYPE_CHECKING:
    from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest

logger = init_logger(__name__)


@register_tts_adapter
class Qwen3TTSAdapter(ARTTSAdapter):
    """Adapter for Qwen3-TTS (AR ``engine_client`` backend)."""

    stage_keys = frozenset({"qwen3_tts"})
    name = "qwen3_tts"

    def normalize(self, request: "OpenAICreateSpeechRequest") -> None:
        """Qwen3-TTS normalization (Base-task inference, voice lowercasing) is
        performed inside ``validate`` today; kept fused for a strict behaviour
        match. See ``_validate_qwen_tts_request``."""

    def validate(self, request: "OpenAICreateSpeechRequest") -> str | None:
        # Go through the shared dispatcher (which routes to
        # _validate_qwen_tts_request for this model-type) rather than the leaf
        # validator directly, matching the pre-adapter qwen3 path.
        return self.ctx.server._validate_tts_request(request)

    async def build(
        self, request: "OpenAICreateSpeechRequest", sampling_params_list: list, has_inline_ref_audio: bool
    ) -> PreparedRequest:
        prompt, tts_params, warmup_key = await self.ctx.server._build_qwen3_tts_request(request)
        return PreparedRequest(
            prompt=prompt,
            tts_params=tts_params,
            model_type=tts_params.get("task_type", ["unknown"])[0],
            warmup_artifact_key=warmup_key,
        )
