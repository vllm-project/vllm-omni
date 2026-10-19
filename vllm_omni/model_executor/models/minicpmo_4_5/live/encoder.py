# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Stage-0 streaming audio encoding for MiniCPM-o 4.5 Live sessions.

Each streaming update of a Live request carries the mel features of its
unit(s) in ``model_intermediate_buffer["minicpmo_live_audio"]``. The Thinker's
``preprocess`` hands them here: units are encoded once per update, in order,
with ``get_audio_embedding_streaming`` and the session's own encoder cache, as
the official ``streaming_prefill`` does, and the embeddings replace the
update's unit audio rows (placeholder tokens) wherever chunked prefill
schedules them. The cache lives in the request's buffer and goes away with the
request.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from vllm.logger import init_logger

logger = init_logger(__name__)

LIVE_AUDIO_KEY = "minicpmo_live_audio"
LIVE_ENCODER_KEY = "minicpmo_live_encoder"


@dataclass
class LiveEncoderState:
    """Per-request streaming encoder state (stored by reference in the buffer)."""

    past_key_values: Any = None
    seq: int | None = None
    embeds: torch.Tensor | None = None
    cursor: int = 0


def _ensure_dynamic_cache_compat() -> None:
    """The checkpoint's Whisper attention calls ``DynamicCache.get_usable_length``,
    which newer Transformers releases removed."""
    from transformers.cache_utils import DynamicCache

    if hasattr(DynamicCache, "get_usable_length"):
        return

    def get_usable_length(self, new_seq_length: int | None = None, layer_idx: int = 0) -> int:
        return int(self.get_seq_length(layer_idx))

    DynamicCache.get_usable_length = get_usable_length


def _encode_units(thinker: Any, state: LiveEncoderState, units: list[dict[str, Any]]) -> torch.Tensor:
    device = thinker.apm.conv1.weight.device
    dtype = thinker.apm.conv1.weight.dtype
    _ensure_dynamic_cache_compat()
    previous = thinker.audio_past_key_values
    thinker.audio_past_key_values = state.past_key_values
    embeds: list[torch.Tensor] = []
    try:
        for unit in units:
            mel = np.frombuffer(unit["features"], dtype=np.float32).reshape(int(unit["mel_bins"]), -1)
            features = torch.from_numpy(mel.copy()).to(device=device, dtype=dtype).unsqueeze(0)
            batch = {
                "audio_features": features,
                "audio_feature_lens": [torch.tensor([int(unit["frames"])], device=device)],
            }
            result = thinker.get_audio_embedding_streaming(
                batch,
                use_extra_context=True,
                prefix_extra_frames=int(unit["prefix_extra_frames"]),
                suffix_extra_frames=int(unit["suffix_extra_frames"]),
            )
            embeds.extend(tensor for group in result for tensor in group)
        state.past_key_values = thinker.audio_past_key_values
    finally:
        thinker.audio_past_key_values = previous
    return torch.cat(embeds, dim=0)


def apply_live_audio(
    thinker: Any,
    input_ids: torch.Tensor,
    input_embeds: torch.Tensor,
    info: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, Any]] | None:
    """Write the update's unit audio embeddings into this step's span.

    Returns ``(embeds, buffer_update)``, or None when the span has no unit
    audio rows (decode steps, the system prompt, earlier rows of the update).
    """
    live = info.get(LIVE_AUDIO_KEY)
    if not isinstance(live, dict):
        return None
    rows = (input_ids == int(live["placeholder_id"])).nonzero(as_tuple=True)[0]
    if rows.numel() == 0:
        return None
    state = info.get(LIVE_ENCODER_KEY)
    if not isinstance(state, LiveEncoderState):
        state = LiveEncoderState()
    seq = int(live["seq"])
    if state.seq != seq:
        state.embeds = _encode_units(thinker, state, list(live["units"]))
        state.seq = seq
        state.cursor = 0
    available = int(state.embeds.shape[0]) - state.cursor
    count = int(rows.numel())
    if count > available:
        # A recompute after preemption re-runs earlier units' rows, whose
        # embeddings are gone; they keep the placeholder embedding.
        logger.warning(
            "MiniCPM-o Live: %d unit audio rows scheduled but %d embeddings left (request %s)",
            count,
            available,
            info.get("request_id"),
        )
        rows = rows[-available:] if available > 0 else rows[:0]
        count = int(rows.numel())
    embeds = input_embeds.clone()
    if count:
        chunk = state.embeds[state.cursor : state.cursor + count]
        embeds[rows] = chunk.to(device=embeds.device, dtype=embeds.dtype)
        state.cursor += count
    return embeds, {LIVE_ENCODER_KEY: state}
