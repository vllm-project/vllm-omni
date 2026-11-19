# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""PersonaPlex ``LiveSessionProcessor``.

One 80 ms frame (1920 samples at 24 kHz) per streaming update. The first
update of a request also reserves the voice-prompt and persona prefill rows;
every later one carries the text token stage 0 sampled for the previous frame
(``live/stage0.py`` runs that frame's depformer step with it).

``instructions`` and developer ``input`` messages form the persona; user and
assistant ``input`` messages become a "Conversation so far" line in it, at
most ``MAX_PERSONA_TOKENS`` tokens. The context is never trimmed: like the
reference model, stage 0 attends to a sliding window of the last 3,000 frames,
and the Live deploy sizes ``max_model_len`` for a whole session.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

from vllm_omni.entrypoints.openai.live.processor import LiveSessionProcessor, ProcessorContext
from vllm_omni.entrypoints.openai.live.protocol import LiveProtocolError
from vllm_omni.entrypoints.openai.live.session import (
    AssistantItem,
    LiveSessionState,
    NativeUnitItem,
    SystemItem,
    UserTextItem,
)
from vllm_omni.model_executor.models.personaplex.live.stage0 import LIVE_KEY
from vllm_omni.model_executor.models.personaplex.prompting import (
    AUDIO_SILENCE_FRAME_CNT,
    CHUNK_PERIOD_MS,
    DEFAULT_PERSONA,
    DEFAULT_VOICE,
    SAMPLE_RATE,
    cached_personaplex_tokenizer,
    personaplex_max_voice_rows,
    personaplex_model_dir,
    personaplex_persona_tokens,
    personaplex_prefill_slots,
)

# Persona prefill frames a session may use; also bounds the prefill in the
# startup check that stage 0 holds a whole session.
MAX_PERSONA_TOKENS = 1024


class PersonaPlexLiveSessionProcessor(LiveSessionProcessor):
    input_sample_rate: ClassVar[int] = SAMPLE_RATE
    output_sample_rate: ClassVar[int] = SAMPLE_RATE
    unit_ms: ClassVar[int] = CHUNK_PERIOD_MS

    @classmethod
    def preload(cls, model_path: str) -> None:
        cached_personaplex_tokenizer(personaplex_model_dir(model_path))

    @classmethod
    def max_prefix_tokens(cls, model_path: str) -> int:
        return (
            personaplex_max_voice_rows(personaplex_model_dir(model_path))
            + 2 * AUDIO_SILENCE_FRAME_CNT
            + MAX_PERSONA_TOKENS
        )

    def __init__(self, context: ProcessorContext) -> None:
        super().__init__(context)
        self.model_dir = personaplex_model_dir(context.model_path)
        self._tokenize = cached_personaplex_tokenizer(self.model_dir)
        self.voice = DEFAULT_VOICE
        self.persona = DEFAULT_PERSONA
        self.prefill_slots = 0
        self._seq = 0

    # ---- configuration -----------------------------------------------------------

    def count_tokens(self, text: str) -> int:
        return len(self._tokenize(text))

    def decode(self, token_ids: list[int]) -> str:
        return ""

    def resolve_voice(self, voice: Any) -> str | None:
        voice_id = super().resolve_voice(voice)
        if voice_id is None or voice_id == "default":
            self.voice = DEFAULT_VOICE
            return voice_id
        name = voice_id if voice_id.endswith(".pt") else f"{voice_id}.pt"
        if Path(name).name != name:
            raise LiveProtocolError("invalid_value", "Voice must be a bundled voice name.", param="audio.output.voice")
        self.voice = name
        return voice_id

    def configure(self, state: LiveSessionState) -> None:
        developer: list[str] = []
        history: list[str] = []
        for item in state.history:
            if isinstance(item, SystemItem):
                developer.append(item.text)
            elif isinstance(item, UserTextItem):
                history.append(f"User: {item.text}")
            elif isinstance(item, AssistantItem):
                history.append(f"You: {item.text}")
        persona = "\n\n".join(developer) or DEFAULT_PERSONA
        if history:
            persona = f"{persona}\n\nConversation so far: {' '.join(history)}"
        persona_tokens = len(personaplex_persona_tokens(self.model_dir, persona))
        if persona_tokens > MAX_PERSONA_TOKENS:
            raise LiveProtocolError(
                "invalid_value",
                f"The persona ('instructions' and 'input') is {persona_tokens} tokens; the limit is "
                f"{MAX_PERSONA_TOKENS}.",
                param="instructions",
            )
        self.persona = persona
        try:
            self.prefill_slots = personaplex_prefill_slots(self.model_dir, self.voice, self.persona)
        except FileNotFoundError as exc:
            raise LiveProtocolError("invalid_value", str(exc), param="audio.output.voice") from exc

    def stage0_sampling_params(self, state: LiveSessionState, base: Any) -> Any:
        params = super().stage0_sampling_params(state, base)
        # Greedy, one temporal text token per frame.
        params.temperature = 0.0
        params.top_k = 1
        params.max_tokens = 1
        return params

    # ---- rendering -------------------------------------------------------------

    def _frame(self, unit: NativeUnitItem, *, first: bool, last_sampled_token: int | None) -> dict[str, Any]:
        self._seq = 1 if first else self._seq + 1
        live: dict[str, Any] = {"seq": self._seq, "pcm": unit.audio.astype("float32").tobytes()}
        if first:
            live["voice"] = self.voice
            live["persona"] = self.persona
        else:
            live["prev_text"] = last_sampled_token
        return {
            "prompt_token_ids": [0] * (1 + (self.prefill_slots if first else 0)),
            "model_intermediate_buffer": {LIVE_KEY: live},
        }

    def render_unit(
        self,
        state: LiveSessionState,
        unit: NativeUnitItem,
        *,
        first: bool,
        last_sampled_token: int | None,
    ) -> dict[str, Any]:
        return self._frame(unit, first=first, last_sampled_token=last_sampled_token)

    # ---- context -----------------------------------------------------------------

    def prefix_tokens(self, state: LiveSessionState) -> int:
        return self.prefill_slots

    def unit_tokens(self, unit: NativeUnitItem) -> int:
        return 1

    def spoken_text(self, token_ids: list[int], text: str) -> str:
        return ""
