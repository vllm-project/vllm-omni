# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from vllm_omni.entrypoints.openai.tts_adapters import register_tts_adapter
from vllm_omni.entrypoints.openai.tts_adapters.base import ARTTSAdapter, OutputPolicy, PreparedRequest


@register_tts_adapter
class PhononAdapter(ARTTSAdapter):
    name = "phonon"
    stage_keys = frozenset({"phonon"})
    model_archs = frozenset({"PhononTalkerForConditionalGeneration"})

    def _load_supported_speakers(self):
        return ["default"]

    def _load_supported_languages(self):
        return frozenset({"English", "en", "Auto"})

    def validate(self, request):
        if not request.input or not request.input.strip():
            return "Input text cannot be empty"
        if request.voice not in (None, "default"):
            return "Phonon initial release supports the default pretrained voice"
        if request.ref_audio is not None or request.speaker_embedding is not None:
            return "Phonon initial release supports precomputed speaker conditioning"
        return None

    async def build(self, request, sampling_params_list, has_inline_ref_audio):
        import asyncio

        return await asyncio.to_thread(self._build, request)

    def _build(self, request):
        import sentencepiece
        from transformers.utils.hub import cached_file

        from vllm_omni.model_executor.models.phonon.prompt import load_voice, prepare_text

        config = self.ctx.engine_client.model_config
        revision = getattr(config, "revision", None)
        tokenizer = sentencepiece.SentencePieceProcessor(
            model_file=cached_file(config.model, "tokenizer.model", revision=revision)
        )
        voice, project = load_voice(config.model, config.hf_config.to_dict(), revision)
        text, tail = prepare_text(request.input)
        ids = tokenizer.encode(text)
        frames = voice.shape[-1] if project else voice.shape[-2]
        prompt = {
            "prompt_token_ids": [0] * (frames + len(ids) + 1),
            "additional_information": {
                "phonon_text_ids": ids,
                "phonon_voice": voice.tolist(),
                "phonon_project_voice": project,
                "phonon_tail_frames": tail,
            },
        }
        return PreparedRequest(
            prompt=prompt, model_type="phonon", output_policy=OutputPolicy(accumulate_nonstreaming=True)
        )
