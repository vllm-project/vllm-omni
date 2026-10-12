# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Typed output boundary for the incremental Lychee duplex port."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass

import torch

from vllm_omni.engine.duplex.contracts import (
    duplex_resource_request_belongs_to_session,
    duplex_turn_id_from_request_id,
)
from vllm_omni.engine.duplex.plugin import DuplexDataPlane, EncodeAudio

from .codec import LycheeCodecStreams, aligned_values, output_payload
from .history import LycheeSessionHistory


@dataclass(frozen=True, slots=True)
class LycheeDataPlaneContext:
    epoch: int
    turn_id: int
    active_response_turn_id: int | None
    active_response_id: str | None
    auto_responds: bool
    response_format: str
    speed: float | None
    modalities: tuple[str, ...]


class LycheeDataPlaneSession(DuplexDataPlane):
    """Project validated Lychee control decisions while owning request cursors.

    P4 exposes only typed listen/speak/backchannel decisions. Codec-to-PCM
    projection remains a P5 responsibility; raw token tensors never cross the
    public protocol boundary.
    """

    def __init__(
        self,
        encode_audio: EncodeAudio | None = None,
        *,
        codec_streams: LycheeCodecStreams | None = None,
        histories: dict[str, LycheeSessionHistory] | None = None,
        decode_text: Callable[[list[int]], str] | None = None,
    ) -> None:
        self.histories = histories if histories is not None else {}
        self._decode_text = decode_text
        self._text_tokens: dict[str, list[int]] = {}
        self._text_sent: dict[str, str] = {}
        self._text_tick: dict[str, int] = {}
        self._text_active: set[str] = set()
        self._text_mode: dict[str, str] = {}
        self._text_number: dict[str, int] = {}
        self._published_completed: dict[str, int] = {}
        self._pending_events: dict[str, dict[int, list[dict[str, object]]]] = {}
        self._encode_audio = encode_audio
        self.codec_streams = codec_streams or LycheeCodecStreams()
        self._audio_seq: dict[tuple[str, str, int], int] = {}
        self._audio_completed: dict[str, int] = {}
        self._terminal: set[str] = set()
        self._last_projected_tick: dict[str, int] = {}

    def begin_request(self, request_id: str) -> None:
        self._terminal.discard(request_id)

    def is_terminal(self, request_id: str | None) -> bool:
        return request_id in self._terminal if request_id is not None else False

    def mark_terminal(self, request_id: str) -> None:
        self._terminal.add(request_id)

    def close_stream(self, request_id: str) -> None:
        self._terminal.discard(request_id)
        self._last_projected_tick.pop(request_id, None)
        self.codec_streams.close_request(request_id)
        for history in self.histories.values():
            if request_id in history.request_ids:
                history.force_listen_at_frontier = True
                history.request_ids.discard(request_id)
        self._audio_seq = {key: seq for key, seq in self._audio_seq.items() if key[0] != request_id}
        self._audio_completed.pop(request_id, None)
        self._text_tokens.pop(request_id, None)
        self._text_sent.pop(request_id, None)
        self._text_tick.pop(request_id, None)
        self._text_active.discard(request_id)
        self._text_mode.pop(request_id, None)
        self._text_number.pop(request_id, None)
        self._published_completed.pop(request_id, None)
        self._pending_events.pop(request_id, None)

    def close_session(self, session_id: str, *, active_request_id: str | None = None) -> None:
        request_ids = set(self.codec_streams.states)
        for cursors in (
            self._text_tick,
            self._text_number,
            self._text_mode,
            self._last_projected_tick,
            self._audio_completed,
            self._published_completed,
            self._pending_events,
        ):
            request_ids.update(cursors)
        request_ids.update(self._terminal)
        request_ids.update(key[0] for key in self._audio_seq)
        if active_request_id is not None:
            request_ids.add(active_request_id)
        for request_id in request_ids:
            if request_id == active_request_id or duplex_resource_request_belongs_to_session(request_id, session_id):
                self.close_stream(request_id)
        self.histories.pop(session_id, None)

    @staticmethod
    def _tensor_values(tensors: Mapping[str, object], key: str) -> list[int]:
        return aligned_values(tensors, key)

    def project(self, result: object, *, context: object | None = None) -> Iterator[dict[str, object]]:
        if not isinstance(result, dict):
            return
        outputs = result.get("data_plane_outputs")
        if not isinstance(outputs, list):
            return
        for output in outputs:
            request_id = getattr(output, "request_id", None)
            if not isinstance(request_id, str) or not request_id or request_id in self._terminal:
                continue
            inner, completion, tensors = output_payload(output)
            if getattr(completion, "finish_reason", None) == "error":
                yield {
                    "data_plane_request_id": request_id,
                    "error_code": "lychee_request_aborted",
                    "error": str(getattr(completion, "stop_reason", "Native Lychee transaction failed")),
                    "retryable": True,
                    "recover_binding": True,
                }
                continue
            audio_owner = tensors.get("lychee_t2w")
            if not isinstance(audio_owner, Mapping):
                audio_owner = {
                    key.removeprefix("chunk.").removeprefix("lychee_t2w."): value
                    for key, value in tensors.items()
                    if key.startswith(("chunk.lychee_t2w.", "lychee_t2w."))
                }
            if not audio_owner and isinstance(tensors.get("chunk"), Mapping):
                chunk = tensors["chunk"]
                audio_owner = chunk.get("lychee_t2w")
                if not isinstance(audio_owner, Mapping):
                    audio_owner = {
                        key.removeprefix("lychee_t2w."): value
                        for key, value in chunk.items()
                        if key.startswith("lychee_t2w.")
                    }
            if isinstance(audio_owner, Mapping) and audio_owner:
                yield from self._project_audio(request_id, tensors, audio_owner, context)
                continue
            if not tensors:
                continue
            from vllm_omni.engine.duplex.contracts import duplex_session_id_from_request_id

            history = self.histories.get(duplex_session_id_from_request_id(request_id))
            if history is not None:
                history.record_outputs(tensors)
            ticks = self._tensor_values(tensors, "lychee_tick")
            controls = self._tensor_values(tensors, "lychee_control_token_ids")
            if not ticks or not controls:
                continue
            if len(ticks) != len(controls):
                raise ValueError("Lychee tick/control output lengths disagree")
            window_seqs = self._tensor_values(tensors, "lychee_audio_window_seq")
            if window_seqs and len(window_seqs) != len(ticks):
                raise ValueError("Lychee tick/audio-window output lengths disagree")
            epochs = self._tensor_values(tensors, "lychee_execution_epoch")
            if epochs and len(epochs) != len(ticks):
                raise ValueError("Lychee tick/execution-epoch output lengths disagree")
            model_turn_id = duplex_turn_id_from_request_id(request_id)
            if model_turn_id is None and isinstance(context, LycheeDataPlaneContext):
                model_turn_id = context.turn_id
            texts = aligned_values(tensors, "lychee_text_token_ids")
            speeches = aligned_values(tensors, "lychee_speech_token_ids")
            if texts and len(texts) != len(ticks):
                raise ValueError("Lychee text/tick output lengths disagree")
            if speeches and len(speeches) != len(ticks):
                raise ValueError("Lychee speech/tick output lengths disagree")
            for index, (tick, control) in enumerate(zip(ticks, controls)):
                text_events = []
                if tick >= 0 and tick > self._text_tick.get(request_id, -1):
                    text_events = list(
                        self._project_text(
                            request_id,
                            tick,
                            control,
                            texts[index] if texts else 158358,
                            speeches[index] if speeches else -1,
                            model_turn_id,
                        )
                    )
                number = self._text_number.get(request_id, 0)
                # Waveform EOF owns response.done after its final cache flush.
                if control == 158353 and request_id in self.codec_streams.states:
                    # Cumulative snapshots replay old EOS rows. Their waveform
                    # boundary must never rewind published control decisions.
                    self._last_projected_tick[request_id] = max(tick, self._last_projected_tick.get(request_id, -1))
                elif tick % 10 == 9 and tick > self._last_projected_tick.get(request_id, -1):
                    event: dict[str, object] = {
                        "stage_role": "lychee_fd",
                        "data_plane_request_id": request_id,
                        # The resident request spans several protocol turns.
                        # Session ownership resolves the current turn on delivery.
                        "model_turn_id": None if number else model_turn_id,
                        "model_response_number": number,
                        "lychee_tick": tick,
                        "lychee_control_token_id": control,
                    }
                    metadata = {"tick": tick, "control_token_id": control, "model_response_number": number}
                    if window_seqs:
                        event["lychee_audio_window_seq"] = window_seqs[index]
                        metadata["audio_window_seq"] = window_seqs[index]
                    if epochs:
                        metadata["execution_epoch"] = epochs[index]
                    event["model_metadata"] = metadata
                    if control in {158353, 158354, 158356, 158357}:
                        event.update(
                            is_listen=True,
                            model_listen=True,
                            preserve_request=True,
                            end_of_turn=False,
                            reason="model_listen",
                            await_waveform_final=(number > self._published_completed.get(request_id, 0)),
                        )
                    elif control in {158352, 158355, 158362}:
                        event.update(
                            model_speak=True,
                            model_backchannel=control == 158362,
                            preserve_request=True,
                            reason="model_backchannel" if control == 158362 else "model_speak",
                        )
                    else:
                        raise ValueError(f"Invalid Lychee control decision: {control}")
                    self._last_projected_tick[request_id] = tick
                    yield from self._ordered_event(request_id, number, event)
                for event in text_events:
                    yield from self._ordered_event(request_id, number, event)

    def _ordered_event(self, request_id, number, event):
        """Finish prior PCM before publishing the next model utterance."""
        completed = self._published_completed.get(request_id, 0)
        if number == 0:
            yield event
            return
        if number <= completed:
            # A later listening decision has no response of its own.
            if event.get("is_listen") is True:
                event["await_waveform_final"] = False
                event["model_turn_id"] = None
                yield event
            return
        if number > completed + 1:
            pending = self._pending_events.setdefault(request_id, {})
            pending.setdefault(number, []).append(event)
            return
        yield event
        if event.get("stage_role") != "tts" or event.get("end_of_turn") is not True:
            return
        self._published_completed[request_id] = number
        self._audio_completed[request_id] = number
        metadata = event.get("model_metadata", {})
        self._audio_seq.pop((request_id, metadata.get("model_response_id"), metadata.get("execution_epoch")), None)
        pending = self._pending_events.get(request_id, {})
        following = pending.pop(number + 1, [])
        if not pending:
            self._pending_events.pop(request_id, None)
        for queued in following:
            yield from self._ordered_event(request_id, number + 1, queued)

    def _project_text(self, request_id, tick, control, text, speech, model_turn_id):
        starts = control in {158352, 158362} or speech == 151693
        bc_to_s = self._text_mode.get(request_id) == "BC" and control == 158352
        if starts and (request_id not in self._text_active or bc_to_s):
            self._text_number[request_id] = self._text_number.get(request_id, 0) + 1
            self._text_mode[request_id] = "BC" if control == 158362 else "S"
            self._text_active.add(request_id)
            self._text_tokens[request_id] = []
            self._text_sent[request_id] = ""
        self._text_tick[request_id] = tick
        if request_id in self._text_active and text not in {151665, 151693, 151694, 151695, 158358}:
            tokens = self._text_tokens.setdefault(request_id, [])
            tokens.append(text)
            if self._decode_text is None:
                raise RuntimeError("Lychee text projection has no tokenizer")
            decoded = self._decode_text(tokens)
            # Retain partial UTF-8 BPE sequences until the tokenizer can render
            # them without a replacement character at the chunk boundary.
            if not decoded.endswith("\ufffd"):
                sent = self._text_sent.get(request_id, "")
                if not decoded.startswith(sent):
                    raise ValueError("Lychee cumulative text changed after publication")
                delta = decoded[len(sent) :]
                self._text_sent[request_id] = decoded
                if delta:
                    yield {
                        "stage_role": "lychee_fd",
                        "data_plane_request_id": request_id,
                        "model_turn_id": None,
                        "model_response_number": self._text_number[request_id],
                        "text": delta,
                        "end_of_turn": False,
                        "preserve_request": True,
                    }
        if speech == 151694 or control == 158353:
            self._text_active.discard(request_id)
            self._text_mode.pop(request_id, None)

    def _project_audio(self, request_id, payload, owner, context):
        # The generation materializer may concatenate several chunk tensors
        # in one delivery. Numeric ownership columns and sample lengths retain
        # each chunk boundary through that transport.
        columns = {}
        count = 1
        names = (
            "session_epoch",
            "execution_epoch",
            "response_number",
            "chunk_seq",
            "tick",
            "final",
            "discarded",
            "num_samples",
        )
        for name in names:
            value = owner.get(name)
            if isinstance(value, torch.Tensor):
                values = value.reshape(-1).tolist()
                if not values:
                    raise ValueError(f"Lychee waveform ownership {name} is empty")
                columns[name] = values
                count = max(count, len(values))
        if count > 1:
            for name in names:
                if len(columns.get(name, [])) != count:
                    raise ValueError(f"Lychee waveform ownership {name} does not align to chunk rows")
        audio = payload.get("audio", payload.get("model_outputs"))
        if isinstance(audio, list):
            if not all(isinstance(chunk, torch.Tensor) for chunk in audio):
                raise ValueError("Native Lychee waveform chunks must be PCM tensors")
            audio = torch.cat([chunk.reshape(-1) for chunk in audio]) if audio else torch.empty(0)
        if audio is not None and not isinstance(audio, torch.Tensor):
            raise ValueError("Native Lychee waveform output must contain a PCM tensor")
        if isinstance(audio, torch.Tensor):
            audio = audio.reshape(-1)
        sizes = columns.get("num_samples")
        if sizes is None:
            size = owner.get("num_samples")
            sizes = [audio.numel() if isinstance(audio, torch.Tensor) else 0] if size is None else [size]
        if len(sizes) != count or any(type(size) is not int or size < 0 for size in sizes):
            raise ValueError("Native Lychee waveform sample lengths do not align to chunk rows")
        available = audio.numel() if isinstance(audio, torch.Tensor) else 0
        if sum(sizes) != available:
            raise ValueError("Native Lychee waveform sample lengths disagree with PCM")
        rates = payload.get("sr", 24000)
        if isinstance(rates, list):
            rates = torch.cat(
                [rate.reshape(-1) if isinstance(rate, torch.Tensor) else torch.tensor([rate]) for rate in rates]
            )
        if isinstance(rates, torch.Tensor):
            rates = rates.reshape(-1).tolist()
        if not isinstance(rates, list):
            rates = [rates]
        if len(rates) not in {1, count} or any(rate != 24000 for rate in rates):
            raise ValueError("Native Lychee waveform output must be 24000 Hz")
        offset = 0
        for index, size in enumerate(sizes):
            row = dict(owner)
            for name, values in columns.items():
                row[name] = values[index]
            waveform = audio[offset : offset + size] if isinstance(audio, torch.Tensor) else None
            offset += size
            yield from self._project_one_audio(request_id, waveform, row, context)

    def _project_one_audio(self, request_id, audio, owner, context):
        owner = dict(owner)
        for name in ("session_epoch", "execution_epoch", "response_number", "chunk_seq", "tick", "final", "discarded"):
            value = owner.get(name)
            if isinstance(value, torch.Tensor):
                if value.numel() != 1:
                    raise ValueError(f"Lychee waveform ownership {name} must be scalar")
                owner[name] = value.item()
        number = owner.get("response_number")
        if "response_id" not in owner and isinstance(number, int):
            owner["response_id"] = f"{request_id}:response:{number}"
        if owner.get("discarded") is True:
            return
        if isinstance(context, LycheeDataPlaneContext) and owner.get("session_epoch") != context.epoch:
            return
        response_id, epoch, seq = owner.get("response_id"), owner.get("execution_epoch"), owner.get("chunk_seq")
        if not isinstance(response_id, str) or not isinstance(epoch, int) or not isinstance(seq, int):
            raise ValueError("Native Lychee waveform output lacks response ownership")
        key = (request_id, response_id, epoch)
        number = owner.get("response_number")
        if not isinstance(number, int) or number < 1:
            raise ValueError("Native Lychee waveform output lacks a response sequence")
        if number <= self._audio_completed.get(request_id, 0) or seq <= self._audio_seq.get(key, -1):
            return
        sample_rate = 24000
        encoded = None
        samples = 0
        if isinstance(audio, torch.Tensor) and audio.numel():
            if self._encode_audio is None:
                raise RuntimeError("Lychee waveform projection has no audio encoder")
            samples = audio.numel()
            encoded = self._encode_audio(
                audio.reshape(-1),
                sample_rate,
                getattr(context, "response_format", "pcm16"),
                getattr(context, "speed", None),
            )
            if not encoded:
                raise ValueError("Lychee waveform output could not be encoded")
        final = owner.get("final") is True
        self._audio_seq[key] = seq
        if final:
            stream = self.codec_streams.states.get(request_id)
            if stream is not None and stream.response_number == number:
                stream.awaiting_final = False
        if encoded or final:
            event = {
                "stage_role": "tts",
                "data_plane_request_id": request_id,
                "model_turn_id": None,
                "model_response_number": number,
                "audio": encoded or "",
                "audio_format": getattr(context, "response_format", "pcm16"),
                "sample_rate_hz": sample_rate,
                "audio_duration_ms": round(samples * 1000 / sample_rate),
                "audio_complete": final,
                "audio_text_mark": False,
                "end_of_turn": final,
                "preserve_request": True,
                "model_metadata": {
                    "tick": owner.get("tick"),
                    "execution_epoch": epoch,
                    "codec_chunk_seq": seq,
                    "model_response_id": response_id,
                    "model_response_number": number,
                },
            }

            yield from self._ordered_event(request_id, number, event)


__all__ = ["LycheeDataPlaneContext", "LycheeDataPlaneSession"]
