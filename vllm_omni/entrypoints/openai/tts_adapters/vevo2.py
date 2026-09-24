# SPDX-License-Identifier: Apache-2.0
"""Vevo2 serving adapter.

Vevo2 (Amphion's unified AR + flow-matching TTS) runs as a single AR stage and
emits the full waveform as delta chunks, following the MOSS-TTS-Nano pattern.
Voice cloning requires a reference clip (``ref_audio``); ``ref_text`` (the
transcript of the reference) is recommended for prosody but not required.
"""

from typing import TYPE_CHECKING, Any

from vllm.inputs import tokens_input

from vllm_omni.entrypoints.openai.tts_adapters import register_tts_adapter
from vllm_omni.entrypoints.openai.tts_adapters.base import (
    ARTTSAdapter,
    OutputPolicy,
    PreparedRequest,
    conditioning_cache_salt,
)

if TYPE_CHECKING:
    from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest


@register_tts_adapter
class Vevo2Adapter(ARTTSAdapter):
    stage_keys = frozenset({"vevo2"})
    name = "vevo2"

    def validate(self, request: "OpenAICreateSpeechRequest") -> str | None:
        """Every request must include ``ref_audio``.

        Upstream's ``inference_ar_and_fm`` asserts ``timbre_ref_wav_path is not
        None``. ``ref_text`` is the transcription of the style-reference audio
        and is recommended (it conditions the AR LM's prosody) but not strictly
        required by upstream. The ``voice`` OpenAI field is accepted for the
        uploaded-speakers flow and ignored otherwise -- Vevo2 has no built-in
        speaker presets.
        """
        server = self.ctx.server
        err = server._apply_uploaded_speaker(request)
        if err:
            return err
        if not request.input or not request.input.strip():
            return "Input text cannot be empty"
        if request.ref_audio is None:
            return (
                "Vevo2 requires 'ref_audio' (reference audio for timbre "
                "and style); upstream's inference_ar_and_fm asserts "
                "timbre_ref_wav_path is not None."
            )
        return server._validate_ref_audio_format(request.ref_audio)

    async def build(
        self, request: "OpenAICreateSpeechRequest", sampling_params_list: list, has_inline_ref_audio: bool
    ) -> PreparedRequest:
        """Build ``additional_information`` for Vevo2.

        ``ref_audio`` is resolved server-side and passed as a
        ``(wav_list, sample_rate)`` pair under ``prompt_audio_array`` so the
        model owns temp-file lifecycle (matches MOSS-TTS-Nano). ``ref_text`` is
        forwarded as the style reference text. SVS / conversion / editing
        modes are deferred to follow-up PRs.
        """
        server = self.ctx.server
        tts_params: dict[str, Any] = {"text": [request.input]}
        if request.ref_text is not None:
            tts_params["ref_text"] = [request.ref_text]
        wav_list, sr, _ = await server._resolve_ref_audio(request.ref_audio)
        tts_params["prompt_audio_array"] = [[wav_list, sr]]
        if request.voice:
            voice_lower = request.voice.lower()
            if voice_lower in server.uploaded_speakers and not has_inline_ref_audio:
                tts_params["voice_name"] = [voice_lower]
                tts_params["voice_created_at"] = [server._voice_created_at(voice_lower)]
        self._apply_request_sampling(request, sampling_params_list, tts_params)
        prompt = tokens_input(prompt_token_ids=[1])
        prompt["additional_information"] = tts_params
        prompt["cache_salt"] = conditioning_cache_salt(request, tts_params)
        # Vevo2 yields the waveform as delta chunks with async_chunk=false, so
        # the non-streaming path must accumulate across engine outputs.
        return PreparedRequest(
            prompt=prompt,
            tts_params=tts_params,
            model_type=self.name,
            output_policy=OutputPolicy(accumulate_nonstreaming=True),
        )

    @staticmethod
    def _apply_request_sampling(
        request: "OpenAICreateSpeechRequest", sampling_params_list: list, tts_params: dict
    ) -> None:
        """Copy the effective seed and sampling knobs into ``additional_information``.

        Vevo2 samples inside Amphion's ``inference_ar_and_fm``, which reads
        every knob from ``additional_information``. The ``SamplingParams`` the
        dummy AR scheduler carries only drive the forced-EOS/safe-token
        sampler and never reach the model, so anything not copied here cannot
        influence generation at all -- a request's ``seed`` included.

        Seed precedence mirrors the shared path: an explicit ``request.seed``
        wins, otherwise the deploy YAML's ``default_sampling_params.seed``.
        Note that ``build()`` runs *before* the shared path writes
        ``request.seed`` onto ``SamplingParams`` (``serving_speech.py``), so
        ``sampling_params_list`` still holds only the deploy defaults here and
        the request has to be read directly.
        """
        seed = request.seed
        if seed is None and sampling_params_list:
            seed = getattr(sampling_params_list[0], "seed", None)
        if seed is not None:
            tts_params["seed"] = [int(seed)]

        # Per-request overrides for the knobs Vevo2 reads; unset keys fall back
        # to the ``_DEFAULT_*`` values in ``modeling_vevo2``.
        extra = request.extra_params or {}
        for name, cast in (("top_k", int), ("top_p", float), ("temperature", float), ("flow_matching_steps", int)):
            value = extra.get(name)
            if value is not None:
                tts_params[name] = [cast(value)]
