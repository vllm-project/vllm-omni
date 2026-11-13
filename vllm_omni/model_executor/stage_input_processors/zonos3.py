# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Producer-side Zonos3 delay reconstruction, after native output D2H.

The reference is ``zonos3.tts.delay`` / ``zonos3.tts.stream.PrefixDecoder``.
This module never decodes audio or decides whether the AR runner should stop.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol, cast

import torch

from vllm_omni.data_entry_keys import CodesStruct, MetaStruct, OmniPayloadStruct, unflatten_payload
from vllm_omni.engine.serialization import deserialize_additional_information


@dataclass(frozen=True)
class Zonos3ChunkConfig:
    codebook_sizes: tuple[int, ...]
    eoa_id: int
    pad_id: int
    chunk_size: int = 15
    chunk_ramp: tuple[int, ...] = (1, 2, 4, 8, 15)
    first_chunk_codebooks: int = 10

    def __post_init__(self) -> None:
        if len(self.codebook_sizes) != 10 or any(size <= 0 for size in self.codebook_sizes):
            raise ValueError("Zonos3 requires ten positive codebook sizes")
        if self.eoa_id < max(self.codebook_sizes) or self.pad_id != self.eoa_id + 1:
            raise ValueError("EOA must follow the code vocabulary and padding must follow EOA")
        if self.chunk_size < 1 or any(size < 1 for size in self.chunk_ramp):
            raise ValueError("Chunk sizes must be positive")
        if not 1 <= self.first_chunk_codebooks <= 10:
            raise ValueError("first_chunk_codebooks must be in 1..10")


@dataclass
class Zonos3RVQStageInputProcessor:
    """One external request's bounded delayed-row buffer.

    A single publication emits all ready frames once its ramp threshold is
    reached, as in the reference PrefixDecoder. This fits the native hook's
    one-payload-per-request contract even when several rows arrive together.
    ``finished`` means AR has stopped (including max-token truncation), whereas
    ``eoa_frame`` bounds audio coverage and does not itself trigger a flush.
    """

    config: Zonos3ChunkConfig
    rows: list[list[int]] = field(default_factory=list)
    row_base: int = 0
    received: int = 0
    emitted: int = 0
    chunk_seq: int = 0
    eoa_frame: int | None = None
    done: bool = False
    prefill_frames: int = 0

    def prime(self, rows: torch.Tensor) -> None:
        if self.received or self.chunk_seq or self.done:
            raise ValueError("Zonos3 Codec priming must precede generated RVQ rows")
        if (
            rows.device.type != "cpu"
            or rows.ndim != 2
            or rows.shape[1] != 10
            or rows.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError("Zonos3 silence priming requires CPU delayed rows [frames, 10]")
        if bool(((rows < 0) | (rows > self.config.pad_id) | (rows == self.config.eoa_id)).any()):
            raise ValueError("Invalid Zonos3 silence priming codes")
        self.rows = rows.tolist()
        self.received = self.prefill_frames = len(self.rows)

    def push(
        self,
        request_id: str,
        codes: torch.Tensor | None,
        *,
        finished: bool = False,
        cancelled: bool = False,
    ) -> OmniPayloadStruct | None:
        if self.done:
            raise RuntimeError("Zonos3 RVQ stream already closed")
        if cancelled:
            self.rows.clear()
            self.done = True
            return None
        cfg = self.config
        if codes is not None and codes.numel() > 0:
            # This is a CPU materialization boundary, never a new D2H copy.
            if codes.device.type != "cpu":
                raise ValueError("Zonos3 producer hook requires CPU-materialized RVQ codes")
            if codes.ndim != 2 or codes.shape[1] != 10 or codes.dtype not in (torch.int32, torch.int64):
                raise ValueError("Expected integer delayed RVQ rows [T, 10]")
            for row in codes.tolist():
                eoa_columns = [j for j, code in enumerate(row) if code == cfg.eoa_id]
                if eoa_columns:
                    observed = max(0, self.received - max(eoa_columns))
                    self.eoa_frame = observed if self.eoa_frame is None else min(self.eoa_frame, observed)
                self.rows.append(row)
                self.received += 1

        complete = max(0, self.received - 9)
        if self.eoa_frame is not None:
            complete = min(complete, self.eoa_frame)
        # At final/max-token truncation only fully arrived frames are flushed.
        partial = self.chunk_seq == 0 and not finished and cfg.first_chunk_codebooks < 10
        ready = max(0, self.received - (cfg.first_chunk_codebooks - 1)) if partial else complete
        if self.eoa_frame is not None:
            ready = min(ready, self.eoa_frame)
        threshold = cfg.chunk_ramp[self.chunk_seq] if self.chunk_seq < len(cfg.chunk_ramp) else cfg.chunk_size
        if not finished and ready - max(self.emitted, self.prefill_frames) < threshold:
            return None
        end = max(self.emitted, ready)
        output = torch.full((end - self.emitted, 10), -1, dtype=torch.int64)
        for f in range(self.emitted, end):
            for j, size in enumerate(cfg.codebook_sizes):
                raw_index = f + j
                if raw_index >= self.received:
                    continue
                code = self.rows[raw_index - self.row_base][j]
                if 0 <= code < size:
                    output[f - self.emitted, j] = code
                elif not (partial and j > 0 and code == cfg.pad_id):
                    raise ValueError(f"Invalid RVQ code at frame {f}, codebook {j}: {code}")
        result = OmniPayloadStruct(
            codes=CodesStruct(audio=output),
            meta=MetaStruct(
                request_id=request_id,
                chunk_seq=self.chunk_seq,
                num_processed_tokens=self.emitted,
                ref_context_size=self.prefill_frames if self.chunk_seq == 0 else None,
                next_stage_prompt_len=1,
                next_stage_prompt_ids=[0],
                finished=torch.tensor(finished),
                codec_streaming=torch.tensor(True),
                override_keys=[("meta", "finished")],
            ),
        )
        self.emitted = end
        self.chunk_seq += 1
        # Row f is no longer needed after frame f has been emitted. Later
        # codebooks of an early partial frame are intentionally never decoded.
        discard = min(self.emitted, self.received) - self.row_base
        del self.rows[:discard]
        self.row_base += discard
        if finished:
            self.rows.clear()
            self.done = True
        return result


class _Connector(Protocol):
    config: dict


class _TransferManager(Protocol):
    connector: _Connector
    request_payload: dict[str, dict[str, object]]


class _Request(Protocol):
    external_req_id: str
    aborted: bool

    def is_finished(self) -> bool: ...

    @property
    def additional_information(self) -> object: ...

_STATE_KEY = "_zonos3_rvq_state"


def _zonos3_conditioning(metadata: Mapping[str, object]) -> dict[str, object]:
    return dict(metadata)


def zonos3_rvq_async_chunk(
    transfer_manager: _TransferManager,
    pooling_output: Mapping[str, object] | None,
    request: _Request,
    is_finished: bool = False,
) -> OmniPayloadStruct | None:
    """Native producer hook; state uses the transport's existing cleanup owner."""
    request_id = request.external_req_id
    is_transport_terminal = bool(is_finished or request.is_finished())
    if getattr(request, "aborted", False):
        transfer_manager.request_payload.pop(request_id, None)
        return None
    root = transfer_manager.request_payload.setdefault(request_id, {})
    state = root.get(_STATE_KEY)
    if not isinstance(state, Zonos3RVQStageInputProcessor):
        config = transfer_manager.connector.config
        extra = config.get("extra", config)
        state = Zonos3RVQStageInputProcessor(
            Zonos3ChunkConfig(
                codebook_sizes=tuple(extra["codebook_sizes"]),
                eoa_id=int(extra["eoa_id"]),
                pad_id=int(extra["audio_pad_id"]),
                chunk_size=int(extra.get("codec_chunk_frames", 15)),
                chunk_ramp=tuple(extra.get("codec_chunk_ramp", (1, 2, 4, 8, 15))),
                first_chunk_codebooks=int(extra.get("first_chunk_codebooks", 10)),
            )
        )
        root[_STATE_KEY] = state
        original = getattr(request, "additional_information", None)
        if original is not None:
            metadata = deserialize_additional_information(original)
            if any("." in key for key in metadata):
                metadata = unflatten_payload(metadata)
            conditioning = _zonos3_conditioning(metadata)
            count = conditioning.get("silence_prefill_frames", 0)
            frames = conditioning.get("frame_ids")
            if not isinstance(count, int) or count < 0:
                raise ValueError("Invalid Zonos3 silence prefill metadata")
            if count:
                if not isinstance(frames, torch.Tensor) or frames.shape[0] < count:
                    raise ValueError("Zonos3 silence prefill rows are missing from request metadata")
                state.prime(frames[-count:, :10])
            overrides = conditioning.get("codec_chunking", {})
            if overrides:
                if not isinstance(overrides, Mapping):
                    raise ValueError("Invalid Zonos3 per-request Codec chunk settings")
                chunk_size = overrides.get("chunk_size", state.config.chunk_size)
                chunk_ramp = overrides.get("chunk_ramp", state.config.chunk_ramp)
                first_chunk_codebooks = overrides.get("first_chunk_codebooks", state.config.first_chunk_codebooks)
                if (
                    not isinstance(chunk_size, int)
                    or chunk_size < 1
                    or not isinstance(chunk_ramp, (list, tuple))
                    or any(not isinstance(size, int) or size < 1 for size in chunk_ramp)
                    or not isinstance(first_chunk_codebooks, int)
                    or not 1 <= first_chunk_codebooks <= 10
                ):
                    raise ValueError("Invalid Zonos3 per-request Codec chunk settings")
                state.config = Zonos3ChunkConfig(
                    codebook_sizes=state.config.codebook_sizes,
                    eoa_id=state.config.eoa_id,
                    pad_id=state.config.pad_id,
                    chunk_size=chunk_size,
                    chunk_ramp=tuple(chunk_ramp),
                    first_chunk_codebooks=first_chunk_codebooks,
                )
    else:
        original = getattr(request, "additional_information", None)
        if original is not None:
            metadata = deserialize_additional_information(original)
            if any("." in key for key in metadata):
                metadata = unflatten_payload(metadata)
            conditioning = _zonos3_conditioning(metadata)
            overrides = conditioning.get("codec_chunking", {})
            if overrides and (
                state.config.chunk_size != overrides.get("chunk_size", state.config.chunk_size)
                or state.config.chunk_ramp != tuple(overrides.get("chunk_ramp", state.config.chunk_ramp))
                or state.config.first_chunk_codebooks
                != overrides.get("first_chunk_codebooks", state.config.first_chunk_codebooks)
            ):
                raise ValueError("Zonos3 per-request Codec chunk settings changed during the request")
    codes = None
    if pooling_output is not None:
        namespace = pooling_output.get("codes", {})
        if not isinstance(namespace, Mapping):
            raise ValueError("Expected a codes namespace")
        value = namespace.get("audio")
        if value is not None and not isinstance(value, torch.Tensor):
            raise ValueError("Expected a materialized audio-code tensor")
        codes = cast(torch.Tensor | None, value)
    # The sampler owns EOA detection and tail drain on device. Its logical EOS
    # stops further sampling; the ordered transport terminal arrives only
    # after all in-flight raw-code outputs have been delivered, so it is the
    # processor's sole signal to flush the remaining delayed rows.
    result = state.push(
        request_id,
        codes,
        finished=is_transport_terminal,
    )
    if is_transport_terminal:
        transfer_manager.request_payload.pop(request_id, None)
    return result
