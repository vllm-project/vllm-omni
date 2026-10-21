# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""PersonaPlex ``LiveSessionProcessor``.

One 80 ms frame (1920 samples at 24 kHz) per streaming update. The first
update of a request also reserves the voice-prompt and persona prefill rows;
every later one carries the text token stage 0 sampled for the previous frame
(``live/stage0.py`` runs that frame's depformer step with it).

``instructions`` and developer ``input`` messages form the persona; user and
assistant ``input`` messages become a "Conversation so far" line in it. A
resubmission after the context fills (stage-0 ``max_model_len`` frames) starts
a fresh request with the persona and the newest frame only: agent codes of
earlier frames are not kept, so the conversation memory resets.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

from vllm.logger import init_logger

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
    CHUNK_PERIOD_MS,
    DEFAULT_PERSONA,
    DEFAULT_VOICE,
    SAMPLE_RATE,
    cached_personaplex_tokenizer,
    personaplex_model_dir,
    personaplex_prefill_slots,
)

logger = init_logger(__name__)


class PersonaPlexLiveSessionProcessor(LiveSessionProcessor):
    input_sample_rate: ClassVar[int] = SAMPLE_RATE
    output_sample_rate: ClassVar[int] = SAMPLE_RATE
    unit_ms: ClassVar[int] = CHUNK_PERIOD_MS

    @classmethod
    def preload(cls, model_path: str) -> None:
        cached_personaplex_tokenizer(personaplex_model_dir(model_path))

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

    def render_resume(self, state: LiveSessionState) -> dict[str, Any]:
        return self._frame(state.units[-1], first=True, last_sampled_token=None)

    # ---- context -----------------------------------------------------------------

    def prefix_tokens(self, state: LiveSessionState) -> int:
        return self.prefill_slots

    def unit_tokens(self, unit: NativeUnitItem) -> int:
        return 1

    def trim_context(self, state: LiveSessionState) -> int:
        units = state.units
        drop = {id(unit) for unit in units[:-1]}
        logger.debug(
            "Live session %s: context trim: max_model_len=%d prefill=%d; dropping %d frames, "
            "restarting from the frame at %d ms",
            state.id,
            self.context.max_model_len,
            self.prefill_slots,
            len(drop),
            units[-1].start_ms if units else 0,
        )
        state.history = [item for item in state.history if id(item) not in drop]
        return len(drop)

    def spoken_text(self, token_ids: list[int], text: str) -> str:
        return ""
