# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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

from typing import Any

import torch

from vllm_omni.data_entry_keys import (
    CodesStruct,
    MetaStruct,
    OmniPayloadStruct,
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

    def _codes_from(src: Any) -> torch.Tensor | None:
        if not isinstance(src, dict):
            return None
        nested = src.get("codes")
        audio = nested.get("audio") if isinstance(nested, dict) else None
        return audio if audio is not None else src.get("codes.audio")

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
) -> OmniPayloadStruct | None:
    """Streaming: accumulate per-frame agent codes, emit a codebook-major chunk.

    Minimal fixed-chunk variant (no left-context / ref-code complexity, which
    PersonaPlex does not use). Frames are buffered on the transfer manager until a
    chunk's worth is ready (or the request finishes), then flushed.
    """
    request_id = getattr(request, "external_req_id", getattr(request, "request_id", "?"))
    # The adapter passes ``is_finished=True`` for both a resumable segment
    # boundary and the terminal request boundary. PersonaPlex must preserve the
    # delayed cb1..7 tail across the former; only a non-resumable stop flushes
    # the stream.
    finished = bool(is_finished and not getattr(request, "resumable", False))
    request_payload = getattr(transfer_manager, "request_payload", None)
    if not isinstance(request_payload, dict):
        request_payload = {}
        transfer_manager.request_payload = request_payload
    state = request_payload.setdefault(request_id, {})
    frames = state.setdefault("personaplex_frames", [])

    # Codes live in the server-side request's additional_information under
    # ("codes","audio") (talker_mtp_output_key), not in multimodal_output (latent).
    def _codes_from(src: Any) -> torch.Tensor | None:
        if isinstance(src, dict):
            nested = src.get("codes")
            a = nested.get("audio") if isinstance(nested, dict) else None
            return a if a is not None else src.get("codes.audio")
        return None

    # Explicit None fallback: `a or b` would evaluate bool(a) on a multi-element
    # Tensor and raise "Boolean value of Tensor ... is ambiguous".
    audio = _codes_from(getattr(request, "additional_information", None))
    if audio is None:
        audio = _codes_from(multimodal_output)
    if isinstance(audio, torch.Tensor) and audio.numel() > 0:
        a = audio if audio.ndim == 2 else audio.reshape(1, -1)
        frames.append(a[-1].to(torch.long).cpu())  # latest frame's codes

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
    target_frames = initial_chunk if not state.get("personaplex_emitted") and initial_chunk > 0 else chunk

    # De-delay needs one successor raw frame: N output acoustic frames require
    # N + 1 raw depformer rows.
    available_frames = max(0, len(frames) - 1)
    if available_frames < target_frames and not finished:
        # Each full-duplex input frame is a resumable stage-0 segment. Returning
        # None would make the generic chunk adapter synthesize a
        # segment-finished marker, wake Code2Wav with its one-token placeholder,
        # and discard the buffered de-delay tail. Explicitly keep this transport
        # chunk non-terminal until enough successor frames exist.
        pending = torch.tensor(False, dtype=torch.bool)
        return OmniPayloadStruct(
            meta=MetaStruct(
                finished=pending,
                is_segment_finished=pending,
            )
        )
    emit_frames = available_frames if finished else target_frames
    if emit_frames <= 0:
        if finished:
            request_payload.pop(request_id, None)
            return _empty_finished_payload()
        return None

    stacked = torch.stack(frames[: emit_frames + 1], dim=0)  # [F+1, dep_q]
    flat = _agent_codes_to_codebook_major(stacked)
    if finished:
        request_payload.pop(request_id, None)
    else:
        # Row ``emit_frames`` is the successor used by the last emitted frame
        # and the cb0 source for the next frame.
        state["personaplex_frames"] = frames[emit_frames:]
        state["personaplex_emitted"] = True
    return OmniPayloadStruct(
        codes=CodesStruct(audio=flat),
        meta=MetaStruct(finished=torch.tensor(bool(finished), dtype=torch.bool)),
    )


__all__ = [
    "talker2code2wav_token_only",
    "talker2code2wav_full_payload",
    "talker2code2wav_async_chunk",
]
