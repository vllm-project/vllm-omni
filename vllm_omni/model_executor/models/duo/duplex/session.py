# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Per-session computed-row ledger, without owning an engine or model."""

import asyncio
from collections import deque
from dataclasses import dataclass, field

from vllm_omni.engine.duplex.plugin import DefaultDuplexModelSessionState
from vllm_omni.model_executor.models.joyai_vl_interaction.duplex.input import JoyAIFrameBuffer


class DuoInputBuffer(JoyAIFrameBuffer):
    """Official transactional PCM buffer; retain one complete codec observation."""

    def __init__(self):
        super().__init__(accepts_audio=True)
        self.max_frames = 17

    def prepare_append(self, payload, **kwargs):
        old_frames = list(self.frames)
        reservation = super().prepare_append(payload, **kwargs)
        frames = payload.get("video_frames")
        if isinstance(frames, list) and frames:
            self.frames = (old_frames + frames)[-self.max_frames :]
        return reservation


@dataclass
class DuoSessionState(DefaultDuplexModelSessionState):
    request_id: str | None = None
    fence: int = -1
    connector: object = None
    serializer: object = None
    next_block: int = 0
    asr_next_block: int = 0
    asr_previous_audio: tuple[str, object] | None = None
    asr_previous_result: tuple[str, str] | None = None
    ledger: list[int] = field(default_factory=list)
    expected: list[int] = field(default_factory=list)
    reference: object = None
    description: dict | str = field(default_factory=dict)
    body_repair: list[int] | None = None
    fragment: object = None
    busy: bool = False
    intent_done: bool = False
    generation_seed: int | None = None
    history_cursor: int = 0
    sampled_rows: int = 0
    pending_tail_token: int | None = None
    current_text: str = ""
    recent_assistant_tokens: deque[tuple[int, ...]] = field(default_factory=lambda: deque(maxlen=24))
    first_utterance_seen: bool = False
    first_utterance_complete: bool = False
    waiting_for_user: bool = False
    prefilled_wait: bool = False
    media_reset_generation: int = 0
    reset_media_cache: bool = False
    next_body_hidden: object = None
    next_body_ids: list[int] = field(default_factory=list)
    awaiting_clean: bool = False
    last_frames: object = None
    native_intent: bool = False
    native_turns: list[int] = field(default_factory=list)
    native_rope_shift: int = 0
    native_evicted: int = 0
    realization_conditioning: str = "training_prefix"
    session_id: str | None = None
    tenant: str = "official-duplex"
    consumed_input_items: set = field(default_factory=set)
    next_publish: int = 0
    inflight: dict = field(default_factory=dict)
    dispatch_tasks: set = field(default_factory=set)
    intent_ready: object = None
    realization_credit: object = None
    capacity: object = None
    schedule: object = None

    def close(self):
        if self.schedule is not None:
            self.schedule.close()
        try:
            current = asyncio.current_task()
        except RuntimeError:
            current = None
        for task in list(self.dispatch_tasks):
            if task is not current:
                task.cancel()
        self.dispatch_tasks.clear()
        if self.intent_ready is not None:
            self.intent_ready.set()
        self.inflight.clear()
        self.consumed_input_items.clear()
        if self.connector is not None:
            self.connector.close()
        self.awaiting_clean = False
        self.pending_tail_token = None
        self.asr_previous_audio = None
        self.asr_previous_result = None
        self.recent_assistant_tokens.clear()
        self.ledger.clear()
        self.expected.clear()
        self.busy = False
        self.body_repair = None
        self.last_frames = None
        self.next_body_hidden = None
        self.next_body_ids.clear()


@dataclass
class DuoChunkState:
    request_id: str
    parent_request_id: str
    owner: DuoSessionState
    session_id: str
    fence: int
    next_block: int
    tenant: str
    current_text: str
    busy: bool = True
    awaiting_clean: bool = False
    clean_released: bool = False
    reset_media_cache: bool = False
    final: bool = False
