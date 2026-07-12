# SPDX-License-Identifier: Apache-2.0
"""MOSS-TTS serving adapters (Nano + full family).

Both variants share the same build/validate flow (``_build_moss_tts_params``
handles each); they are registered under distinct model-type names.
"""

from typing import TYPE_CHECKING

from vllm.inputs import tokens_input

from vllm_omni.entrypoints.openai.tts_adapters import register_tts_adapter
from vllm_omni.entrypoints.openai.tts_adapters.base import ARTTSAdapter, PreparedRequest, conditioning_cache_salt

if TYPE_CHECKING:
    from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest


class _MossTTSAdapterBase(ARTTSAdapter):
    def validate(self, request: "OpenAICreateSpeechRequest") -> str | None:
        """Validate any MOSS-TTS-family request (nano + 5 full variants).

        Dispatches by ``self._moss_variant``:
          - ``tts``/``realtime``: require ``ref_audio`` (voice cloning).
          - ``ttsd``: require ``ref_audio`` (speaker 1); ``ref_audio_2``
            optional (defaults to the same ref for both speakers).
          - ``sound_effect``: require ``ambient_sound`` (no ref_audio).
          - ``voice_generator``: require ``instructions`` (no ref_audio).
          - For the legacy moss_tts_nano model_type the variant is None and
            we fall through to the original nano contract (ref_audio only).
        """
        server = self.ctx.server
        err = server._apply_uploaded_speaker(request)
        if err:
            return err

        if not request.input or not request.input.strip():
            # SoundEffect can legitimately have empty input (just ambient_sound).
            if server._moss_variant != "sound_effect":
                return "Input text cannot be empty"

        v = server._moss_variant
        if v in (None, "tts", "realtime", "local"):
            if request.ref_audio is None:
                label = (
                    "MOSS-TTS-Nano"
                    if v is None
                    else (
                        "MOSS-TTS-Realtime"
                        if v == "realtime"
                        else ("MOSS-TTS-Local-Transformer" if v == "local" else "MOSS-TTS")
                    )
                )
                return f"{label} requires 'ref_audio' (reference audio for voice cloning)."
            return server._validate_ref_audio_format(request.ref_audio)

        if v == "ttsd":
            if request.ref_audio is None:
                return "MOSS-TTSD requires 'ref_audio' (speaker 1 reference)."
            fmt_err = server._validate_ref_audio_format(request.ref_audio)
            if fmt_err:
                return fmt_err
            if request.ref_audio_2 is not None:
                return server._validate_ref_audio_format(request.ref_audio_2)
            return None

        if v == "sound_effect":
            if not request.ambient_sound or not request.ambient_sound.strip():
                return (
                    "MOSS-SoundEffect requires 'ambient_sound' (natural language "
                    "description of the sound effect to synthesise)."
                )
            return None

        if v == "voice_generator":
            if not request.instructions or not request.instructions.strip():
                return (
                    "MOSS-VoiceGenerator requires 'instructions' (natural language "
                    "voice description, e.g. 'a warm female voice with an American accent')."
                )
            return None

        return None  # unreachable

    async def build(
        self, request: "OpenAICreateSpeechRequest", sampling_params_list: list, has_inline_ref_audio: bool
    ) -> PreparedRequest:
        server = self.ctx.server
        tts_params = await server._build_moss_tts_params(request)
        if request.voice:
            voice_lower = request.voice.lower()
            if voice_lower in server.uploaded_speakers and not has_inline_ref_audio:
                tts_params["voice_name"] = [voice_lower]
                tts_params["voice_created_at"] = [server._voice_created_at(voice_lower)]
        # MOSS reads the resolved seed at build time (it samples internally).
        if sampling_params_list and getattr(sampling_params_list[0], "seed", None) is not None:
            tts_params["seed"] = [sampling_params_list[0].seed]
        if isinstance(tts_params.get("prompt_token_ids"), list):
            prompt_token_ids = tts_params.pop("prompt_token_ids")
            prompt = tokens_input(prompt_token_ids=prompt_token_ids)
        else:
            prompt = tokens_input(prompt_token_ids=[1])
        prompt["additional_information"] = tts_params
        prompt["cache_salt"] = conditioning_cache_salt(request, tts_params)
        return PreparedRequest(prompt=prompt, tts_params=tts_params, model_type=self.name)


@register_tts_adapter
class MossTTSNanoAdapter(_MossTTSAdapterBase):
    stage_keys = frozenset({"moss_tts_nano"})
    name = "moss_tts_nano"


@register_tts_adapter
class MossTTSAdapter(_MossTTSAdapterBase):
    stage_keys = frozenset({"moss_tts", "moss_tts_codec", "moss_tts_local", "moss_tts_local_codec"})
    name = "moss_tts"
