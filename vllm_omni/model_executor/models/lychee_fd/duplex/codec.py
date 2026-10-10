# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Request-owned codec stream boundaries for the native waveform stage."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch

from vllm_omni.engine.duplex.contracts import duplex_session_id_from_request_id
from vllm_omni.model_executor.models.lychee_fd.output_ring import LYCHEE_OUTPUT_COLUMNS


def output_payload(output: object) -> tuple[object, object | None, Mapping[str, object]]:
    inner = getattr(output, "request_output", None) or output
    completions = getattr(inner, "outputs", None)
    completion = completions[0] if isinstance(completions, list) and completions else None
    payload = getattr(completion, "multimodal_output", None)
    if not payload:
        payload = getattr(inner, "multimodal_output", None)
    if not isinstance(payload, Mapping):
        payload = getattr(payload, "tensors", None)
    if not isinstance(payload, Mapping):
        return inner, completion, {}
    # Native client DELTA columns drain together under chunk.*. Preserve
    # waveform ownership and legacy top-level decisions, then canonicalize
    # only this model's aligned decision columns for history/control/codec.
    canonical = dict(payload)
    chunk = payload.get("chunk")
    nested = dict(chunk) if isinstance(chunk, Mapping) else {}
    for key in LYCHEE_OUTPUT_COLUMNS:
        flat = f"chunk.{key}"
        if flat in payload and key in nested:
            raise ValueError(f"Ambiguous Lychee chunk decision column: {key}")
        value = payload[flat] if flat in payload else nested.get(key)
        if value is None:
            continue
        if key in payload:
            raise ValueError(f"Ambiguous Lychee decision column: {key}")
        canonical[key] = value
        canonical.pop(flat, None)
        nested.pop(key, None)
    if isinstance(chunk, Mapping):
        if nested:
            canonical["chunk"] = nested
        else:
            canonical.pop("chunk", None)
    return inner, completion, canonical


def aligned_values(payload: Mapping[str, object], key: str, *, length: int | None = None) -> list[int]:
    value = payload.get(key)
    if isinstance(value, list) and value and all(isinstance(item, torch.Tensor) for item in value):
        value = torch.cat([item.reshape(-1) for item in value])
    if not isinstance(value, torch.Tensor):
        if length is not None:
            raise ValueError(f"Missing Lychee output field: {key}")
        return []
    values = [int(item) for item in value.reshape(-1).tolist()]
    if length is not None and len(values) != length:
        raise ValueError(f"Lychee {key} output length disagrees with tick output")
    return values


@dataclass(slots=True)
class CodecStreamState:
    last_tick: int = -1
    response_number: int = 0
    response_id: str | None = None
    execution_epoch: int | None = None
    chunk_seq: int = -1
    awaiting_final: bool = False
    voice_mode: str | None = None


class LycheeCodecStreams:
    """Split codec deltas by model response without terminating resident KV."""

    def __init__(self) -> None:
        self.states: dict[str, CodecStreamState] = {}

    def close_request(self, request_id: str) -> None:
        self.states.pop(request_id, None)

    def close_session(self, session_id: str) -> None:
        for request_id in list(self.states):
            if duplex_session_id_from_request_id(request_id) == session_id:
                self.close_request(request_id)

    def consume(
        self, request_id: str, payload: Mapping[str, object], *, session_epoch: int
    ) -> dict[str, object] | None:
        ticks = aligned_values(payload, "lychee_tick")
        if not ticks:
            return None
        speeches = aligned_values(payload, "lychee_speech_token_ids", length=len(ticks))
        controls = aligned_values(payload, "lychee_control_token_ids", length=len(ticks))
        epochs = aligned_values(payload, "lychee_execution_epoch", length=len(ticks))
        state = self.states.setdefault(request_id, CodecStreamState())
        codecs: list[int] = []
        response_id = state.response_id
        final = False
        last_tick = state.last_tick
        for tick, speech, control, epoch in zip(ticks, speeches, controls, epochs):
            if tick < 0 or tick <= state.last_tick:
                continue
            if state.execution_epoch is not None and epoch != state.execution_epoch:
                raise ValueError("Lychee execution epoch changed within a resident codec stream")
            state.execution_epoch = epoch
            if state.response_id is not None and state.voice_mode == "BC" and control == 158352:
                # BC -> S closes the prior utterance before consuming the new
                # StartSpeak row. The next packet owns that same row.
                response_id = state.response_id
                final = True
                state.awaiting_final = True
                state.response_id = None
                state.voice_mode = None
                break
            if state.response_id is None and (control in {158352, 158362} or speech == 151693):
                state.response_number += 1
                state.response_id = f"{request_id}:response:{state.response_number}"
                state.chunk_seq = -1
                state.voice_mode = "BC" if control == 158362 else "S"
            if state.response_id is not None:
                response_id = state.response_id
                if 158257 <= speech < 158352:
                    raise ValueError("Lychee speech token exceeds the waveform codebook")
                if 151696 <= speech < 158257:
                    codecs.append(speech - 151696)
                if speech == 151694 or control == 158353:
                    final = True
                    state.awaiting_final = True
                    state.response_id = None
                    state.voice_mode = None
            state.last_tick = tick
            last_tick = tick
            if final:
                # Control decisions occur once per window. A packet may not
                # silently combine independent responses into one waveform.
                break
        if not codecs and not final:
            return None
        state.chunk_seq += 1
        return {
            "request_id": request_id,
            "session_id": duplex_session_id_from_request_id(request_id) or request_id,
            "response_id": response_id,
            "response_number": state.response_number,
            "session_epoch": session_epoch,
            "execution_epoch": state.execution_epoch,
            "chunk_seq": state.chunk_seq,
            "tick": last_tick,
            "codec_ids": codecs,
            "empty": not codecs,
            "final": final,
            "cancel": False,
        }

    def consume_all(
        self, request_id: str, payload: Mapping[str, object], *, session_epoch: int
    ) -> list[dict[str, object]]:
        """Drain every independent response in a cumulative model snapshot."""
        packets = []
        while True:
            packet = self.consume(request_id, payload, session_epoch=session_epoch)
            if packet is None:
                return packets
            packets.append(packet)
