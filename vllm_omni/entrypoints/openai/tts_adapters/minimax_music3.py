# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax Music 3 serving adapter for ``/v1/audio/speech``.

This is a text-to-music model on a speech endpoint, so most of the speech
contract does not apply. ``input`` carries the lyrics and ``instructions`` the
caption that describes genre, instrumentation, tempo and production. There is
no speaker to select, no reference audio, and no temperature: sampling is
fixed by the checkpoint. Unsupported parameters are rejected rather than
silently ignored, so a mistaken request fails immediately instead of quietly
returning something the caller did not ask for.
"""

from typing import TYPE_CHECKING, Any

from vllm_omni.entrypoints.openai.tts_adapters import register_tts_adapter
from vllm_omni.entrypoints.openai.tts_adapters.base import ARTTSAdapter, PreparedRequest
from vllm_omni.model_executor.models.minimax_music3.constants import (
    AUDIO_FRAME_RATE,
    MAX_AUDIO_FRAMES,
    MAX_PROMPT_TOKENS,
)
from vllm_omni.model_executor.models.minimax_music3.prompt import (
    build_prompt,
    validate_tokenizer_ids,
)

if TYPE_CHECKING:
    from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest

# Captions are long by design: the checkpoint was trained on structured
# descriptions covering global attributes, emotional progression, vocal detail
# and arrangement. The reference captions run to roughly 5,000 characters, so
# the generic 500-character instructions cap does not apply here. The real
# limit is the tokenized prompt cap, enforced downstream.
_MAX_CAPTION_CHARS = 20_000

# Sampling is fixed by the checkpoint: guidance at 1.5, then a seeded top-k 50
# draw. Accepting these would imply a control the model does not have.
_UNSUPPORTED_SAMPLING_PARAMS = ("temperature", "top_p", "top_k", "repetition_penalty")

# CFG pair plumbing is derived from the request id by the scheduler. A caller
# supplying these would desync the guided pair.
_INTERNAL_CFG_KEYS = ("cfg_role", "cfg_pair_id", "cfg_null_prompt")


@register_tts_adapter
class MiniMaxMusic3Adapter(ARTTSAdapter):
    """Lyrics plus caption in, one 32 kHz stereo song out."""

    name = "minimax_music3"
    stage_keys = frozenset({"minimax_music3_ar"})
    # 9,000 frames at 25 fps is six minutes, the checkpoint's own ceiling.
    max_new_tokens_min = 1
    max_new_tokens_max = MAX_AUDIO_FRAMES

    def __init__(self, ctx: Any) -> None:
        super().__init__(ctx)
        self._cached_tokenizer: Any = None

    def validate(self, request: "OpenAICreateSpeechRequest") -> str | None:
        if not request.input or not request.input.strip():
            return "MiniMax Music 3 requires non-empty 'input' carrying the lyrics"

        instructions = request.instructions
        if not isinstance(instructions, str) or not instructions.strip():
            return (
                "MiniMax Music 3 requires 'instructions' describing the music "
                "(genre, instrumentation, tempo, mood); it is what decides the arrangement"
            )
        if len(instructions) > _MAX_CAPTION_CHARS:
            return f"'instructions' exceeds the maximum length of {_MAX_CAPTION_CHARS} characters"

        if request.max_new_tokens is not None:
            if request.max_new_tokens < self.max_new_tokens_min:
                return f"'max_new_tokens' must be at least {self.max_new_tokens_min}"
            if request.max_new_tokens > self.max_new_tokens_max:
                return (
                    f"'max_new_tokens' counts audio frames at {AUDIO_FRAME_RATE} per second "
                    f"and cannot exceed {self.max_new_tokens_max}"
                )

        voice = (request.voice or "").strip().lower()
        if voice not in ("", "default"):
            return (
                "MiniMax Music 3 has no speaker to select; the vocal comes from "
                f"'instructions'. Omit 'voice' or pass 'default'; got {request.voice!r}"
            )
        if request.ref_audio is not None or request.ref_text is not None:
            return "MiniMax Music 3 does not support reference-audio conditioning"
        if request.language is not None:
            return "MiniMax Music 3 has no language tag; the language follows the lyrics"
        if request.task_type is not None:
            return "MiniMax Music 3 does not support 'task_type'"
        if request.speed is not None and float(request.speed) != 1.0:
            return "MiniMax Music 3 only supports speed=1.0; put the tempo in 'instructions', e.g. 'at 92 BPM'"

        extra: dict[str, Any] = request.extra_params or {}
        for key in _INTERNAL_CFG_KEYS:
            if key in extra:
                return f"extra_params.{key} is managed internally by the server and cannot be set"
        rejected = sorted(key for key in _UNSUPPORTED_SAMPLING_PARAMS if extra.get(key) is not None)
        if rejected:
            return "MiniMax Music 3 has fixed sampling and does not accept: " + ", ".join(rejected)
        return None

    async def build(
        self,
        request: "OpenAICreateSpeechRequest",
        sampling_params_list: list,
        has_inline_ref_audio: bool,
    ) -> PreparedRequest:
        del sampling_params_list, has_inline_ref_audio
        prompt = build_prompt(request.instructions, request.input)
        # Submit token ids, not text. Guidance needs a companion whose prompt is
        # the same length with the caption-and-lyrics span replaced, and that
        # length match can only be guaranteed on ids. Handing over text would
        # leave the expansion with nothing to build the twin from, and a request
        # that decodes unguided produces a different song.
        token_ids = self._tokenizer().encode(prompt)
        if len(token_ids) > MAX_PROMPT_TOKENS:
            raise ValueError(
                f"MiniMax Music 3 prompt is {len(token_ids)} tokens; "
                f"the maximum is {MAX_PROMPT_TOKENS}. Shorten the caption or the lyrics."
            )
        # The frame budget is enforced by the model, not by the engine's
        # max_tokens. The model needs one decode step beyond the last frame to
        # deliver the song's final window, so capping the engine at exactly
        # this many tokens would cut the tail off.
        max_frames = int(request.max_new_tokens or MAX_AUDIO_FRAMES)
        # Carried on the prompt rather than through tts_params: the adapter
        # path hands PreparedRequest.prompt to the engine untouched, so only
        # what is attached here reaches the model's per-request info.
        return PreparedRequest(
            prompt={
                "prompt_token_ids": token_ids,
                "additional_information": {"max_audio_frames": [max_frames]},
            },
            tts_params={"max_audio_frames": [max_frames]},
            model_type=self.name,
        )

    def _tokenizer(self) -> Any:
        """The music tokenizer, loaded once per process.

        Its special-token ids are pinned: a checkpoint whose tokenizer
        disagrees is not this model, and quietly producing noise is worse than
        failing at the first request.
        """
        if self._cached_tokenizer is None:
            from transformers import AutoTokenizer

            stage = getattr(self.ctx.server, "_tts_stage", None)
            model_path = getattr(getattr(stage, "engine_args", None), "tokenizer", None)
            if not model_path:
                model_path = self.ctx.server.engine_client.model_config.tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            validate_tokenizer_ids(tokenizer)
            self._cached_tokenizer = tokenizer
        return self._cached_tokenizer


__all__ = ["MiniMaxMusic3Adapter"]
