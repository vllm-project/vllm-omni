# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Talker -> Code2Wav input processors for PersonaPlex.

The talker (stage 0) emits, per frame, the ``dep_q`` depformer audio codes under
``("codes","audio")``. Only the leading ``num_active_codebooks`` (the agent's
``cb 0..7``) are decoded to PCM by Mimi; the trailing codebooks are the user
stream and are not vocoded. These processors take the accumulated per-frame agent
codes ``[F, dep_q]``, keep ``cb 0..7``, and flatten them codebook-major
(``[8 * F]``) — the exact layout :class:`PersonaPlexCode2Wav` consumes.

Mirrors the Qwen3-TTS processors (sync ``full_payload`` + ``token_only``; an
async-chunk variant for the streaming path), but with PersonaPlex's agent-codebook
slice instead of Qwen3-TTS's residual layout.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from vllm_omni.data_entry_keys import (
    SKIP_TRANSFER,
    CodesStruct,
    MetaStruct,
    OmniPayloadStruct,
    _SkipTransfer,
)

_NUM_ACTIVE_CODEBOOKS = 8  # agent cb 0..7 (the PCM-bearing rows)


def _empty_finished_payload() -> OmniPayloadStruct:
    return OmniPayloadStruct(
        codes=CodesStruct(audio=torch.empty(0, dtype=torch.long)),
        meta=MetaStruct(finished=torch.tensor(True, dtype=torch.bool)),
    )


def _agent_codes_to_codebook_major(audio: torch.Tensor) -> torch.Tensor:
    """``[F, dep_q]`` raw depformer agent codes -> de-delayed flat codebook-major.

    The talker emits the raw per-frame depformer codes ``gen[t]`` (cb 0..7). Mimi
    needs them ACOUSTICALLY DE-DELAYED to a common time step (Moshi agent delays
    ``[0, 1, 1, 1, 1, 1, 1, 1]``): acoustic frame t's cb_k is predicted at step
    ``t + delay[k]``, so output frame t = ``[gen[t][0], gen[t+1][1:8]]`` (cb0 from
    the current step, cb1..7 from the NEXT step, since the delayed codebooks lag).
    Without this the codebooks are misaligned by one frame and Mimi decodes garble.
    The last frame has no successor for cb1..7, so it is dropped (the delay warmup).
    """
    if audio.ndim != 2 or audio.shape[0] < 2:
        return torch.empty(0, dtype=torch.long)
    audio = audio.to(torch.long)
    k = min(_NUM_ACTIVE_CODEBOOKS, int(audio.shape[1]))
    agent = audio[:, :k]
    valid = (agent >= 0).all(dim=1)
    agent = agent[valid]
    if agent.shape[0] < 2:
        return torch.empty(0, dtype=torch.long)
    # De-delay: cb0 from frame t, cb1..7 from frame t+1 (drop the last frame).
    cb0 = agent[:-1, 0:1]  # [F-1, 1]
    cb_rest = agent[1:, 1:k]  # [F-1, k-1]
    dd = torch.cat([cb0, cb_rest], dim=1)  # [F-1, k] de-delayed
    # [F-1, k] -> [k, F-1] -> flat [k * (F-1)] (codebook-major), as Code2Wav expects.
    return dd.transpose(0, 1).contiguous().reshape(-1)


def talker2code2wav_token_only(
    source_outputs: list,
    prompt: Any = None,
    _requires_multimodal_data: bool = False,
) -> list:
    """Sync ``process_engine_inputs``: build the code2wav placeholder inputs.

    Returns one :class:`OmniTokensPrompt` per finished talker request, with
    ``prompt_token_ids`` sized to the flat codebook-major codec length
    (``num_active_codebooks * num_agent_frames``). The actual codec ids are
    delivered via the worker connector payload from ``talker2code2wav_full_payload``.
    """
    from vllm_omni.inputs.data import OmniTokensPrompt

    del prompt, _requires_multimodal_data
    inputs: list = []
    for talker_output in source_outputs:
        if not getattr(talker_output, "finished", False):
            continue
        output = talker_output.outputs[0]
        # The per-request output is the "latent" (engine_output_type="latent"); the
        # codes ship via the connector full_payload. Size the placeholder from the
        # generated token count (one AR token == one Mimi frame). prompt was 1 frame.
        token_ids = getattr(output, "cumulative_token_ids", None) or getattr(output, "token_ids", None) or []
        n_frames = max(len(token_ids) - 1, 0)
        prompt_len = _NUM_ACTIVE_CODEBOOKS * n_frames
        inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=[0] * prompt_len,
                additional_information=None,
                multi_modal_data=None,
                mm_processor_kwargs=None,
            )
        )
    return inputs


def talker2code2wav_full_payload(
    transfer_manager: Any = None,
    pooling_output: Any = None,
    request: Any = None,
    is_finished: bool = False,
    **kwargs: Any,
) -> OmniPayloadStruct:
    """Producer: collect the talker's accumulated agent codes -> Code2Wav input.

    Called by the connector with (transfer_manager, pooling_output, request,
    is_finished). Prefer the request's additional_information payload under
    ("codes","audio") (talker_mtp_output_key), while accepting pooling_output
    and the legacy multimodal_output keyword as compatibility fallbacks.
    """
    del transfer_manager, is_finished

    audio = None
    for source in (
        getattr(request, "additional_information", None),
        getattr(request, "additional_information_cpu", None),
        pooling_output,
        # Temporary compatibility shim for the two producer call contracts.
        # Remove after https://github.com/vllm-project/vllm-omni/issues/4872
        # provides validated full-payload and async-chunk processor protocols.
        kwargs.get("multimodal_output"),
    ):
        audio = _codes_from(source)
        if audio is not None:
            break
    if not isinstance(audio, torch.Tensor) or audio.numel() == 0:
        return _empty_finished_payload()
    flat = _agent_codes_to_codebook_major(audio)
    if flat.numel() == 0:
        return _empty_finished_payload()
    return OmniPayloadStruct(
        codes=CodesStruct(audio=flat),
        meta=MetaStruct(finished=torch.tensor(True, dtype=torch.bool)),
    )


def talker2code2wav_async_chunk(
    transfer_manager: Any,
    multimodal_output: Any,
    request: Any,
    is_finished: bool = False,
) -> OmniPayloadStruct | _SkipTransfer:
    """Streaming: accumulate per-frame agent codes, emit a codebook-major chunk.

    Minimal fixed-chunk variant (no left-context / ref-code complexity, which
    PersonaPlex does not use). Frames are buffered on the transfer manager until a
    chunk's worth is ready (or the request finishes), then flushed; a frame that
    only buffers sends nothing.
    """
    (result,) = talker2code2wav_async_chunk_batch(
        transfer_manager,
        [{"multimodal_output": multimodal_output, "request": request, "is_finished": is_finished}],
    )
    return result


def talker2code2wav_async_chunk_batch(
    transfer_manager: Any,
    items: Sequence[Mapping[str, Any]],
) -> list[OmniPayloadStruct | _SkipTransfer]:
    """:func:`talker2code2wav_async_chunk` for a run of queued rows at once.

    ``items`` holds each row's ``multimodal_output``, ``request`` and
    ``is_finished``, in queue order, and the results are what calling the
    one-row processor on each row in that order returns. The tensor work does
    not grow with the rows: their latest frames are gathered to the host in
    one concatenation, and every chunk that is due is de-delayed in one go.
    """
    request_payload = getattr(transfer_manager, "request_payload", None)
    if not isinstance(request_payload, dict):
        request_payload = {}
        transfer_manager.request_payload = request_payload
    connector = getattr(transfer_manager, "connector", None)
    raw_cfg = getattr(connector, "config", {}) or {}
    cfg = raw_cfg.get("extra", raw_cfg) if isinstance(raw_cfg, dict) else {}
    chunk = int(cfg.get("codec_chunk_frames", 25))
    initial_chunk = int(cfg.get("initial_codec_chunk_frames") or 0)
    if chunk <= 0 or initial_chunk < 0:
        raise ValueError(
            "PersonaPlex codec chunk sizes must be positive/non-negative: "
            f"codec_chunk_frames={chunk}, initial_codec_chunk_frames={initial_chunk}"
        )

    latest = _latest_frames([_codes_of(item) for item in items])
    results: list[OmniPayloadStruct | _SkipTransfer | None] = [None] * len(items)
    due: list[tuple[int, list[torch.Tensor], bool]] = []
    for index, item in enumerate(items):
        request = item["request"]
        request_id = getattr(request, "external_req_id", getattr(request, "request_id", "?"))
        # The adapter passes ``is_finished=True`` for both a resumable segment
        # boundary and the terminal request boundary. PersonaPlex must preserve
        # the delayed cb1..7 tail across the former; only a non-resumable stop
        # flushes the stream.
        finished = bool(item.get("is_finished") and not getattr(request, "resumable", False))
        state = request_payload.setdefault(request_id, {})
        frames = state.setdefault("personaplex_frames", [])
        frame = latest[index]
        if frame is not None:
            frames.append(frame)  # latest frame's codes
        target_frames = initial_chunk if not state.get("personaplex_emitted") and initial_chunk > 0 else chunk

        # De-delay needs one successor raw frame: N output acoustic frames require
        # N + 1 raw depformer rows.
        available_frames = max(0, len(frames) - 1)
        if available_frames < target_frames and not finished:
            # Each full-duplex input frame is a resumable stage-0 segment. None
            # would make the generic chunk adapter synthesize a segment-finished
            # marker, wake Code2Wav with its one-token placeholder, and discard the
            # buffered de-delay tail; send nothing until enough frames exist.
            results[index] = SKIP_TRANSFER
            continue
        emit_frames = available_frames if finished else target_frames
        if emit_frames <= 0:
            if finished:
                request_payload.pop(request_id, None)
                results[index] = _empty_finished_payload()
            else:
                results[index] = SKIP_TRANSFER
            continue

        due.append((index, frames[: emit_frames + 1], finished))
        if finished:
            request_payload.pop(request_id, None)
        else:
            # Row ``emit_frames`` is the successor used by the last emitted frame
            # and the cb0 source for the next frame.
            state["personaplex_frames"] = frames[emit_frames:]
            state["personaplex_emitted"] = True

    # The chunks never modify ``meta.finished``, so they can share its flags.
    finished_flags: dict[bool, torch.Tensor] = {}
    flats = _codebook_major_chunks([frames for _, frames, _ in due])
    for (index, _, finished), flat in zip(due, flats, strict=True):
        flag = finished_flags.get(finished)
        if flag is None:
            flag = finished_flags[finished] = torch.tensor(finished, dtype=torch.bool)
        results[index] = OmniPayloadStruct(codes=CodesStruct(audio=flat), meta=MetaStruct(finished=flag))
    return results  # type: ignore[return-value]


def _codes_from(src: Any) -> torch.Tensor | None:
    """``src["codes"]["audio"]``, else its flat ``"codes.audio"`` key; None off a dict."""
    if not isinstance(src, dict):
        return None
    nested = src.get("codes")
    audio = nested.get("audio") if isinstance(nested, dict) else None
    return audio if audio is not None else src.get("codes.audio")


def _codes_of(item: Mapping[str, Any]) -> Any:
    """A row's ``("codes","audio")``: the request's own first, else the stage output."""
    # Codes live in the server-side request's additional_information under
    # ("codes","audio") (talker_mtp_output_key), not in multimodal_output (latent).
    # Explicit None fallback: `a or b` would evaluate bool(a) on a multi-element
    # Tensor and raise "Boolean value of Tensor ... is ambiguous".
    audio = _codes_from(getattr(item["request"], "additional_information", None))
    if audio is None:
        audio = _codes_from(item.get("multimodal_output"))
    return audio


def _latest_frames(audios: Sequence[Any]) -> list[torch.Tensor | None]:
    """Each row's latest ``[dep_q]`` frame as host int64, gathered in one copy.

    ``None`` for a row without codes. The frames are views of one new tensor,
    so they do not alias the producer's buffers.
    """
    frames: list[torch.Tensor | None] = [None] * len(audios)
    rows = [index for index, audio in enumerate(audios) if isinstance(audio, torch.Tensor) and audio.numel() > 0]
    if not rows:
        return frames
    views = []
    for index in rows:
        audio = audios[index]
        a = audio if audio.ndim == 2 else audio.reshape(1, -1)
        # The talker emits one [1, dep_q] frame per row: that is its own view.
        views.append(a if a.shape[0] == 1 else a[-1:])
    try:
        latest = torch.cat(views).to(torch.long).cpu().unbind(0)
    except RuntimeError:
        # Rows that cannot share a tensor (codebook counts or devices differ).
        latest = tuple(view[0].to(torch.long).cpu() for view in views)
    for index, frame in zip(rows, latest, strict=True):
        frames[index] = frame
    return frames


def _codebook_major_chunks(chunks: Sequence[Sequence[torch.Tensor]]) -> list[torch.Tensor]:
    """:func:`_agent_codes_to_codebook_major` of each chunk's stacked frames.

    Chunks with the same frame count and width are stacked and, when none of
    their frames is dropped for a negative code, de-delayed together.
    """
    out: list[torch.Tensor | None] = [None] * len(chunks)
    groups: dict[tuple[int, int], list[int]] = {}
    for index, frames in enumerate(chunks):
        groups.setdefault((len(frames), int(frames[0].shape[-1])), []).append(index)
    for (num_frames, width), indices in groups.items():
        try:
            stacked = torch.stack([frame for index in indices for frame in chunks[index]])
            stacked = stacked.view(len(indices), num_frames, width).to(torch.long)
        except RuntimeError:
            for index in indices:
                out[index] = _agent_codes_to_codebook_major(torch.stack(list(chunks[index]), dim=0))
            continue
        k = min(_NUM_ACTIVE_CODEBOOKS, width)
        agent = stacked[:, :, :k]
        if num_frames >= 2 and bool((agent >= 0).all()):
            # De-delay: cb0 from frame t, cb1..7 from frame t+1 (drop the last
            # frame), then codebook-major per chunk -- exactly the per-chunk
            # path, since no chunk drops a frame.
            dd = torch.cat([agent[:, :-1, 0:1], agent[:, 1:, 1:k]], dim=2)  # [chunks, F-1, k]
            flats = dd.transpose(1, 2).reshape(len(indices), -1).unbind(0)
        else:
            flats = tuple(_agent_codes_to_codebook_major(chunk_codes) for chunk_codes in stacked.unbind(0))
        for index, flat in zip(indices, flats, strict=True):
            out[index] = flat
    return out  # type: ignore[return-value]


# ``process_batch`` lets the chunk sender hand a run of queued rows to one call.
talker2code2wav_async_chunk.process_batch = talker2code2wav_async_chunk_batch  # type: ignore[attr-defined]
# The request fields read above, so the chunk sender snapshots only these
# instead of the whole request and its token lists on every frame. On the
# duplex path the codes come only from the stage output (``multimodal_output``):
# the scheduler-side ``additional_information`` never carries them, which is
# what lets a retired segment's save read it after it already holds the next
# frame's input.
talker2code2wav_async_chunk.request_fields = ("resumable", "additional_information")  # type: ignore[attr-defined]


__all__ = [
    "talker2code2wav_token_only",
    "talker2code2wav_full_payload",
    "talker2code2wav_async_chunk",
    "talker2code2wav_async_chunk_batch",
]
