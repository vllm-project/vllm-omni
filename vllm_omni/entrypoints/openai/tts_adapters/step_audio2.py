# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Step-Audio2 serving adapter."""

from typing import TYPE_CHECKING, Any

from vllm_omni.entrypoints.openai.tts_adapters import register_tts_adapter
from vllm_omni.entrypoints.openai.tts_adapters.base import ARTTSAdapter, PreparedRequest, apply_max_new_tokens

if TYPE_CHECKING:
    from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest


@register_tts_adapter
class StepAudio2Adapter(ARTTSAdapter):
    """Adapter for Step-Audio2 (AR ``engine_client`` backend).

    Step-Audio2 runs a single thinker stage that emits audio tokens after a
    ``<tts_start>`` marker, with no uploaded-voice / ref-audio handling.
    Mini uses a native chat prompt string; Base encodes its two segments into
    ``prompt_token_ids`` to match upstream tokenization boundaries.
    """

    stage_keys = frozenset({"step_audio2_thinker"})
    name = "step_audio2"

    def validate(self, request: "OpenAICreateSpeechRequest") -> str | None:
        if not request.input or not request.input.strip():
            return "Input text cannot be empty"
        return None

    async def build(
        self, request: "OpenAICreateSpeechRequest", sampling_params_list: list, has_inline_ref_audio: bool
    ) -> PreparedRequest:
        """Build the speech prompt ending at ``<tts_start>`` for the loaded variant.

        Mini leaves the assistant turn open (omitting ``<|EOT|>``) so the
        thinker continues by generating audio tokens; Base encodes instructions
        and input text separately into ``prompt_token_ids``.
        """
        system_prompt = request.instructions or "以自然的语速读出下面的文字。"
        # Deploy configs mark non-Mini checkpoints through stage-0 hf_overrides.
        engine_client = self.ctx.engine_client
        hf_config = getattr(getattr(engine_client, "model_config", None), "hf_config", None)
        variant = getattr(hf_config, "step_audio2_variant", "mini")
        if variant == "base":
            # Upstream examples-base.py encodes the two segments separately.
            tokenizer = await engine_client.get_tokenizer()
            ids = tokenizer.encode(system_prompt + "\n", add_special_tokens=False)
            ids += tokenizer.encode(request.input + "<tts_start>", add_special_tokens=False)
            return PreparedRequest(prompt={"prompt_token_ids": ids}, tts_params={}, model_type="step_audio2")
        if variant != "mini":
            raise ValueError(f"/v1/audio/speech does not support Step-Audio2 variant {variant!r}")
        prompt = {
            "prompt": (
                f"<|BOT|>system\n{system_prompt}<|EOT|>"
                f"<|BOT|>human\n{request.input}<|EOT|>"
                "<|BOT|>assistant\n<tts_start>"
            )
        }
        return PreparedRequest(prompt=prompt, tts_params={}, model_type="step_audio2")

    def apply_sampling_overrides(
        self,
        sampling_params_list: list,
        request: "OpenAICreateSpeechRequest",
        prompt: dict[str, Any] | None = None,
        request_id: str | None = None,
    ) -> list:
        return apply_max_new_tokens(sampling_params_list, request)
