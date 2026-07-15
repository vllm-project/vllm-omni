# SPDX-License-Identifier: Apache-2.0
"""Kimi Audio serving adapter for /v1/audio/speech endpoint.

Kimi Audio is a 2-stage pipeline:
- Stage 0 (LLM): Shared backbone with bifurcation → text + audio logits
- Stage 1 (Detokenizer): Flow-matching DiT → vocoder → 24kHz waveform

The adapter handles request validation, prompt building, and audio output extraction.
"""

from typing import TYPE_CHECKING, Any

from vllm.inputs import tokens_input
from vllm.logger import init_logger

from vllm_omni.entrypoints.openai.tts_adapters import register_tts_adapter
from vllm_omni.entrypoints.openai.tts_adapters.base import ARTTSAdapter, PreparedRequest
from vllm_omni.model_executor.models.kimi_audio.constants import (
    CODEC_CHUNK_FRAMES,
    KIMI_AUDIO_ASSISTANT_MSG_START_TOKEN_ID,
    KIMI_AUDIO_BLANK_TOKEN_ID,
    KIMI_AUDIO_BOS_TOKEN_ID,
    KIMI_AUDIO_EOS_TOKEN_ID,
    KIMI_AUDIO_MSG_END_TOKEN_ID,
    KIMI_AUDIO_OUTPUT_SAMPLE_RATE,
    KIMI_AUDIO_USER_MSG_START_TOKEN_ID,
)

if TYPE_CHECKING:
    from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest

logger = init_logger(__name__)


@register_tts_adapter
class KimiAudioTTSAdapter(ARTTSAdapter):
    """Adapter for Kimi Audio (2-stage AR + diffusion pipeline)."""

    stage_keys = frozenset({"kimi_audio"})
    name = "kimi_audio"

    def normalize(self, request: "OpenAICreateSpeechRequest") -> None:
        """Normalize request parameters."""
        # Lowercase voice if provided
        if request.voice:
            request.voice = request.voice.lower()

    def validate(self, request: "OpenAICreateSpeechRequest") -> str | None:
        """Validate the request."""
        # Basic validation
        if not request.input:
            return "Input text is required"

        # Kimi Audio doesn't support reference audio for TTS yet
        if request.ref_audio:
            return "Kimi Audio does not support reference audio for TTS"

        return None

    async def build(
        self, request: "OpenAICreateSpeechRequest", sampling_params_list: list, has_inline_ref_audio: bool
    ) -> PreparedRequest:
        """Build the prompt and TTS parameters for Kimi Audio."""
        # Build TTS parameters
        tts_params = self._build_kimi_audio_params(request)

        # Build prompt (Kimi Audio uses special tokens for TTS).
        renderer = self.ctx.server.renderer
        tokenizer = renderer.get_tokenizer()

        # Kimi Audio is trained on dual parallel token streams.  The request
        # text is framed as a *user* text message and the assistant turn is
        # opened so the model can generate an audio turn:
        #
        #   user text msg : role=user,      message_type=text
        #   assistant stub: role=assistant, message_type=None (generation start)
        #
        # Text stream:  [BLANK, *text_tokens, BLANK(msg_end), BLANK(asst_start)]
        # Audio stream: [user_start, BLANK*N,   msg_end,        assistant_start]
        #
        # During decoding sample() teacher-forces the text stream with
        # target_text_token_ids so the audio head is conditioned on the exact
        # transcript.  Both streams MUST stay the same length so
        # embed_input_ids can fuse them position-wise.
        #
        # NOTE: Kimi-Audio-7B-Instruct is audio-conditioned for speech output;
        # text-only prompts are not a supported generation mode for this
        # checkpoint (the audio head is not conditioned to speak without audio
        # in the context).  This prompt matches the reference text-message
        # format and is correct for a TTS-capable Kimi variant.
        prompt_text = request.input
        text_token_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        if hasattr(text_token_ids, "tolist"):
            text_token_ids = text_token_ids.tolist()
        text_token_ids = [int(t) for t in text_token_ids]
        # The vLLM Kimi tokenizer prepends [BOS] even with
        # add_special_tokens=False, but the reference prompt_manager encodes
        # text with bos=False/eos=False.  Strip any framing special tokens so
        # the prompt and the teacher-forced target match the reference exactly.
        while text_token_ids and text_token_ids[0] == KIMI_AUDIO_BOS_TOKEN_ID:
            text_token_ids = text_token_ids[1:]
        while text_token_ids and text_token_ids[-1] in (
            KIMI_AUDIO_EOS_TOKEN_ID,
            KIMI_AUDIO_MSG_END_TOKEN_ID,
        ):
            text_token_ids = text_token_ids[:-1]

        prompt_token_ids = [
            KIMI_AUDIO_BLANK_TOKEN_ID,
            *text_token_ids,
            KIMI_AUDIO_BLANK_TOKEN_ID,  # text side of the user msg_end
            KIMI_AUDIO_BLANK_TOKEN_ID,  # text side of the assistant role start
        ]
        audio_stream = [
            KIMI_AUDIO_USER_MSG_START_TOKEN_ID,
            *[KIMI_AUDIO_BLANK_TOKEN_ID] * len(text_token_ids),
            KIMI_AUDIO_MSG_END_TOKEN_ID,  # close the user message (audio side)
            KIMI_AUDIO_ASSISTANT_MSG_START_TOKEN_ID,  # open the assistant turn
        ]
        tts_params["audio_stream"] = [audio_stream]
        # Teacher-forced transcript: sample() feeds these tokens back as the
        # text stream each decode step so the audio head is conditioned on the
        # exact words to speak (the reference ``audio-text`` training format).
        # NOTE: this is correct for a TTS-trained Kimi variant; the released
        # 7B-Instruct checkpoint is audio-conditioned and still will not speak
        # from a text-only prompt (its audio stream here is all BLANK).
        tts_params["target_text_token_ids"] = [text_token_ids]

        # Estimate how many audio semantic tokens we need from the raw text
        # length.  Each semantic frame represents ~20 ms of audio; spoken
        # English is roughly 4-5 chars per token and ~13 chars/sec, so one text
        # token maps to ~15-20 audio frames.  We use a conservative multiplier
        # and round up to the codec chunk size.
        raw_text = request.input or ""
        if hasattr(tokenizer, "encode"):
            try:
                text_token_ids_len = tokenizer.encode(raw_text, add_special_tokens=False)
                if hasattr(text_token_ids_len, "tolist"):
                    text_token_ids_len = text_token_ids_len.tolist()
                text_token_ids_len = [int(t) for t in text_token_ids_len]
                while text_token_ids_len and text_token_ids_len[0] == KIMI_AUDIO_BOS_TOKEN_ID:
                    text_token_ids_len = text_token_ids_len[1:]
                text_token_len = len(text_token_ids_len)
            except Exception as e:
                logger.warning("Failed to encode raw text length: %s", e)
                text_token_len = max(1, len(raw_text) // 5)
        else:
            text_token_len = max(1, len(raw_text) // 5)
        # ~20 audio frames per text token for Chinese (slower syllable rate),
        # plus a small tail buffer.
        max_audio_tokens = max(CODEC_CHUNK_FRAMES, text_token_len * 20 + 40)
        tts_params["text_token_len"] = [text_token_len]
        tts_params["max_audio_tokens"] = [max_audio_tokens]

        prompt = tokens_input(prompt_token_ids=prompt_token_ids)
        prompt["additional_information"] = tts_params

        return PreparedRequest(
            prompt=prompt,
            tts_params=tts_params,
            model_type=self.name,
        )

    def _build_kimi_audio_params(self, request: "OpenAICreateSpeechRequest") -> dict[str, Any]:
        """Build Kimi Audio specific parameters."""
        tts_params = {
            "task_type": ["tts"],
            "text": [request.input],
        }

        # Add voice if specified
        if request.voice:
            tts_params["voice"] = [request.voice]

        # Add response format for audio output
        tts_params["response_format"] = [request.response_format or "wav"]

        # Add sample rate
        tts_params["sample_rate"] = [KIMI_AUDIO_OUTPUT_SAMPLE_RATE]  # Kimi Audio outputs at 24kHz

        # Add speed if specified
        if request.speed:
            tts_params["speed"] = [request.speed]

        return tts_params
