# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage transfer processors for the NemotronVoiceChat 3-stage pipeline.

thinker (0) -> talker (1): token path only. The thinker's frame-locked text
timeline becomes the talker's ``prompt_token_ids``:
``[text_pad_id] * logical_prompt_token_len + generated frame tokens``. The
talker must process the PAD prompt region too (its KV state at the first
acoustic frame depends on it — NeMo trims prompt-region codes only after the
loop), so the prompt slots are reconstructed here, and the timeline metadata
rides ``additional_information``.

talker (1) -> code2wav (2): PersonaPlex-style full payload. The talker
accumulates one 31-quantizer code stack per frame under ``("codes","audio")``.
The synchronous producer ships the prompt-trimmed ``[frames, 31]`` stack once
the request finishes.  Native duplex wakes expose that same cumulative stack,
so the streaming producer must delta it before feeding code2wav's causal cache.
"""

from __future__ import annotations

from typing import Any

import torch

from vllm_omni.data_entry_keys import (
    CodesStruct,
    IdsStruct,
    MetaStruct,
    OmniPayloadStruct,
)
from vllm_omni.model_executor.models.nemotron_voicechat.runtime_info import scalar_bool

# Full-sequence tensors are re-shipped whole; the full-payload accumulator must
# replace (not concatenate) them if the producer fires more than once. The AR
# runner flattens nested payloads before accumulation, so the effective key is
# the flattened "codes.audio" (the unflattened root is kept for robustness).
_FULL_PAYLOAD_REPLACE_KEYS: frozenset[str] = frozenset({"codes", "codes.audio", "meta", "meta.nvc_logical_prompt_len"})

# <SPECIAL_12> in the Nemotron-Nano-9B-v2 vocab; the checkpoint config's
# stt pad_token. The thinker also reports it in its latent metadata, which
# takes precedence when present.
_DEFAULT_TEXT_PAD_ID = 12
_STEP_TOKEN_SNAPSHOT_MISSING = object()


def _info_get(info: Any, key: str) -> Any:
    if isinstance(info, dict):
        if key in info:
            return info[key]
        meta = info.get("meta")
        if isinstance(meta, dict):
            return meta.get(key)
    return None


def _request_info(request: Any) -> dict[str, Any]:
    """Return the live runner payload, accepting both transport generations."""
    for name in (
        "model_intermediate_buffer",
        "additional_information",
        "additional_information_cpu",
    ):
        info = getattr(request, name, None)
        if isinstance(info, dict):
            return info
        if info is not None and hasattr(info, "entries"):
            from vllm_omni.engine.serialization import (
                deserialize_additional_information,
            )

            decoded = deserialize_additional_information(info)
            if isinstance(decoded, dict):
                return decoded
    return {}


def thinker2talker_token_only(
    source_outputs: list,
    prompt: Any = None,
    _requires_multimodal_data: bool = False,
    next_stage_hf_config: Any = None,
) -> list:
    """Build the talker's engine inputs from the thinker's finished outputs.

    ``prompt_token_ids`` = the full frame-aligned text timeline including the
    PAD prompt region. The thinker's vLLM prompt covers
    ``logical_prompt_token_len + 1`` positions (the +1 is acoustic frame 0), so
    ``logical_prompt_token_len = len(thinker prompt) - 1`` and the generated
    tokens are exactly the acoustic-frame text channel.
    """
    from vllm_omni.inputs.data import OmniTokensPrompt

    del prompt, _requires_multimodal_data
    # Default: one placeholder token (timeline position 0; the NeMo TTS loop
    # starts at t=1) — the vendored talker paths ignore the vLLM positions.
    # The NATIVE talker (hf_overrides.use_native_talker) instead maps vLLM
    # positions onto real PagedAttention KV: its prompt must be exactly the
    # speaker-prompt length (hf_overrides.talker_init_len; 37 for the shipped
    # "Aria" speaker) so the prefill covers the speaker-init embeddings — the
    # talker validates the value and reports the right one on mismatch.
    talker_prompt_len = 1
    if next_stage_hf_config is not None and bool(getattr(next_stage_hf_config, "use_native_talker", False)):
        talker_prompt_len = max(int(getattr(next_stage_hf_config, "talker_init_len", 37)), 1)
    inputs: list = []
    for thinker_output in source_outputs:
        if not getattr(thinker_output, "finished", False):
            continue
        output = thinker_output.outputs[0]
        generated = list(getattr(output, "cumulative_token_ids", None) or getattr(output, "token_ids", None) or [])
        prompt_ids = list(getattr(thinker_output, "prompt_token_ids", None) or [])
        logical_prompt_len = max(len(prompt_ids) - 1, 0)
        pad_id = _DEFAULT_TEXT_PAD_ID
        info = getattr(thinker_output, "additional_information", None)
        reported_pad = _info_get(info, "nvc_text_pad_id")
        if reported_pad is not None:
            pad_id = int(reported_pad)
        timeline = [pad_id] * logical_prompt_len + [int(t) for t in generated]
        inputs.append(
            OmniTokensPrompt(
                # Each decode step consumes one timeline position from
                # additional_information, so the talker needs
                # len(timeline) - 1 decode steps.
                prompt_token_ids=[0] * talker_prompt_len,
                additional_information={
                    "nvc_text_timeline": timeline,
                    "nvc_logical_prompt_len": logical_prompt_len,
                    "nvc_frame_count": len(generated),
                    "nvc_text_pad_id": pad_id,
                },
                multi_modal_data=None,
                mm_processor_kwargs=None,
            )
        )
    return inputs


def thinker2talker_async_chunk(
    transfer_manager: Any,
    multimodal_output: Any = None,
    request: Any = None,
    is_finished: bool = False,
    **kwargs: Any,
) -> OmniPayloadStruct | None:
    """Streaming producer: ship the CUMULATIVE text timeline each thinker step.

    The AR receive path REPLACES the downstream request's
    additional_information with each chunk payload, so every chunk must carry
    the whole timeline so far: ``[pad] * logical_prompt_len + sampled tokens``
    under ``ids.all`` (the talker adopts any longer timeline it sees). The
    timeline is a few hundred ints, so cumulative re-ship is cheap and keeps
    the payload self-contained. ``meta.num_processed_tokens`` carries the
    logical prompt length for the code2wav producer's prompt-region trim.
    """
    step_snapshot = kwargs.pop("new_token_ids", _STEP_TOKEN_SNAPSHOT_MISSING)
    has_step_snapshot = step_snapshot is not _STEP_TOKEN_SNAPSHOT_MISSING
    generated_delta = [int(token_id) for token_id in ((step_snapshot or ()) if has_step_snapshot else ())]
    request_id = request.external_req_id
    request_info = _request_info(request)
    duplex_info = request_info.get("duplex") if isinstance(request_info, dict) else None
    is_duplex = isinstance(duplex_info, dict) and duplex_info.get("data_plane") is True

    state = transfer_manager.request_payload.get(request_id)
    generated_now = [int(token_id) for token_id in (getattr(request, "output_token_ids", None) or [])]
    prompt_ids = list(getattr(request, "prompt_token_ids", None) or [])
    if is_duplex:
        # NeMo's realtime pipeline never feeds the system prompt to the TTS
        # model — the prompt conditions the thinker's KV only, and the talker
        # sees just the speaker prefill plus the per-frame text tokens. A
        # single leading PAD anchors timeline position 0 (the NeMo per-frame
        # loop reads timeline[t-1] as the previous subword at t=1). The
        # OFFLINE pipelines keep the full PAD prompt region for bit parity
        # with NeMo's offline recipe.
        logical_prompt_len = 1
        previous = list(state.get("nvc_duplex_text_tokens", ())) if isinstance(state, dict) else []
        if has_step_snapshot:
            generated = previous + generated_delta
        elif generated_now[: len(previous)] == previous and len(generated_now) > len(previous):
            generated = generated_now
        elif generated_now:
            generated = previous + [generated_now[-1]]
        else:
            generated = previous
    else:
        # vLLM prompt = logical prompt + 1 placeholder (acoustic frame 0).
        logical_prompt_len = max(len(prompt_ids) - 1, 0)
        generated = generated_now
    pad_id = _DEFAULT_TEXT_PAD_ID
    reported_pad = _info_get(request_info, "nvc_text_pad_id")
    if reported_pad is not None:
        pad_id = int(reported_pad)
    timeline = [pad_id] * logical_prompt_len + generated

    # The NATIVE talker (connector extra ``native_talker: true``, matching the
    # stage's hf_overrides.use_native_talker) maps engine positions 1:1 onto
    # timeline steps and consumes them strictly sequentially (per-frame code
    # feedback), so a chunk must never make more than ONE new timeline step
    # schedulable at once when the stage batches prompt tokens
    # (max_num_batched_tokens > 1 lets the speaker prefill run as one chunked
    # prefill). The vendored talker drains chunks model-side instead.
    connector = getattr(transfer_manager, "connector", None)
    raw_cfg = getattr(connector, "config", {}) or {}
    cfg = raw_cfg.get("extra", raw_cfg) if isinstance(raw_cfg, dict) else {}
    native_talker = bool(cfg.get("native_talker", False))

    # Skip no-progress wakeups (same length as the last emitted chunk).
    last_len = int(state.get("nvc_timeline_len", -1)) if isinstance(state, dict) else -1
    if is_duplex and native_talker:
        # Drip: ship at most one NEW timeline token per producer invocation.
        # The producer fires on every thinker step (one per 80 ms frame, and
        # the duplex thinker never stops emitting), so a save-thread burst
        # that coalesced K thinker steps drains within K subsequent wakes —
        # audio trails text by at most one frame per coalesced token.
        ship_len = min(len(timeline), max(last_len, logical_prompt_len) + 1)
        if ship_len <= last_len and not is_finished:
            return None
        timeline = timeline[:ship_len]
    elif len(timeline) <= last_len and not is_finished:
        return None
    transfer_manager.request_payload[request_id] = {
        "nvc_timeline_len": len(timeline),
        "nvc_duplex_text_tokens": generated if is_duplex else (),
    }

    # Offline timelines are frame-locked: the thinker emits exactly max_tokens
    # tokens (ignore_eos), so the FINAL timeline length is known up front.
    # Shipping it lets the talker's CUDA-graph fast path pick the smallest
    # fitting StaticCache bucket (per-step attention cost scales with the
    # bucket size) instead of assuming an unbounded session. Duplex sessions
    # really are unbounded — no expectation is shipped for them.
    expected_total = None
    if not is_duplex:
        max_tokens = getattr(getattr(request, "sampling_params", None), "max_tokens", None)
        if max_tokens is not None:
            expected_total = logical_prompt_len + int(max_tokens)

    new_positions = 1
    if native_talker:
        # A timeline of length L needs L-1 engine positions (the NeMo loop
        # starts at t=1), so grow by new_len - previous_len against a baseline
        # of logical_prompt_len for the first chunk. Under the duplex drip
        # above this is exactly 1 per chunk.
        new_positions = max(len(timeline) - max(last_len, logical_prompt_len), 1)

    return OmniPayloadStruct(
        ids=IdsStruct(all=timeline, prompt=[0]),
        meta=MetaStruct(
            finished=torch.tensor(bool(is_finished and not is_duplex), dtype=torch.bool),
            num_processed_tokens=logical_prompt_len,
            # Vendored talker: one placeholder position per chunk (it drains
            # all new timeline tokens model-side). Native talker: one position
            # PER NEW TOKEN (see native_talker above).
            next_stage_prompt_len=new_positions,
            codec_streaming=is_duplex,
            request_id=request_id if is_duplex else None,
            expected_total_tokens=expected_total,
        ),
    )


def talker2code2wav_async_chunk(
    transfer_manager: Any,
    multimodal_output: Any = None,
    request: Any = None,
    is_finished: bool = False,
    **kwargs: Any,
) -> OmniPayloadStruct | None:
    """Streaming producer: ship prompt-trimmed code stacks in chunks.

    The regular path follows NeMo's prefix-decode recipe: each chunk carries
    the full trimmed history and ``meta.left_context_size`` identifies frames
    already emitted. Native duplex talker wakes also carry cumulative history,
    but code2wav keeps causal codec state for that path and therefore must see
    only rows not observed on an earlier wake. Chunk cadence comes from
    ``codec_chunk_frames`` (default 13 frames ~= 1 s of audio).
    """
    request_id = request.external_req_id
    del kwargs
    request_info = _request_info(request)
    codec_streaming = scalar_bool(_info_get(request_info, "codec_streaming")) or scalar_bool(
        _info_get(multimodal_output, "codec_streaming")
    )
    state = transfer_manager.request_payload.get(request_id)
    if isinstance(state, dict):
        codec_streaming = codec_streaming or bool(state.get("nvc_codec_stream_started", False))
    # ``is_finished`` describes the current resumable Stage-1 scheduler
    # segment, not the lifetime of a native-duplex codec stream. Marking each
    # segment terminal makes the connector clean sender state after its first
    # chunk; every later one-row wake is then mistaken for a fresh stream and
    # trimmed as prompt context. Duplex lifetime is closed by session abort.
    stream_finished = bool(is_finished and not codec_streaming)

    codes = None
    if isinstance(multimodal_output, dict):
        nested = multimodal_output.get("codes")
        codes = nested.get("audio") if isinstance(nested, dict) else None
        if codes is None:
            codes = multimodal_output.get("codes.audio")
    if not isinstance(codes, torch.Tensor) or codes.numel() == 0:
        if not stream_finished:
            # Returning ``None`` at a resumable segment boundary makes the
            # generic chunk adapter synthesize ``is_segment_finished=True``.
            # Stage 2 then stops polling after the first 80 ms wake.  An
            # explicit non-terminal empty payload consumes this connector
            # sequence number without closing the long-lived codec stream.
            if codec_streaming:
                return _empty_streaming_payload(request_id=request_id)
            return None
        return _empty_finished_payload(
            codec_streaming=codec_streaming,
            request_id=request_id,
        )
    if codes.ndim == 1:
        codes = codes.reshape(1, -1)

    # Prompt-region trim (rows are NeMo timeline steps t=1..; keep t >= P).
    # Read the prompt length from the PER-STEP model payload first: the
    # request-level additional_information is overwritten both by arriving
    # thinker chunks and by talker info updates, so it races. Cache the first
    # sighting in the transfer-manager state as a fallback.
    prompt_len = _info_get(multimodal_output if isinstance(multimodal_output, dict) else None, "nvc_logical_prompt_len")
    if prompt_len is None and isinstance(multimodal_output, dict):
        prompt_len = multimodal_output.get("meta.nvc_logical_prompt_len")
    if prompt_len is None:
        prompt_len = _info_get(request_info, "nvc_logical_prompt_len")
        if prompt_len is None:
            prompt_len = _info_get(request_info, "num_processed_tokens")
    if prompt_len is None and isinstance(state, dict):
        prompt_len = state.get("nvc_prompt_len")
    if prompt_len is None:
        raise ValueError(
            "NemotronVoiceChat talker request is missing its logical prompt length "
            "('nvc_logical_prompt_len' or async meta.num_processed_tokens); cannot "
            "trim the prompt-region codes for streaming code2wav."
        )
    frames_sent = int(state.get("nvc_frames_sent", 0)) if isinstance(state, dict) else 0
    if codec_streaming:
        # The AR runner exposes its cumulative model_intermediate_buffer on
        # every wake. Feeding that whole prefix into a stateful causal codec
        # replays history into its cache (live traces showed lengths
        # 19,20,20,21,21,...) and corrupts the waveform. Track rows OBSERVED,
        # separately from rows EMITTED at the chunk cadence, and ship only the
        # suffix first seen on this wake.
        full_trimmed = codes[max(int(prompt_len) - 1, 0) :].detach().to(dtype=torch.long, device="cpu")
        frames_seen = int(state.get("nvc_frames_seen", 0)) if isinstance(state, dict) else 0
        if int(full_trimmed.shape[0]) < frames_seen:
            raise ValueError(
                "NemotronVoiceChat duplex talker code history shrank from "
                f"{frames_seen} to {int(full_trimmed.shape[0])} frames; cannot "
                "safely continue the stateful code2wav stream."
            )
        trimmed = full_trimmed[frames_seen:]
        frames_seen = int(full_trimmed.shape[0])
        pending = state.get("nvc_pending_codes") if isinstance(state, dict) else None
        if isinstance(pending, torch.Tensor) and pending.numel() > 0:
            trimmed = torch.cat([pending.reshape(-1, codes.shape[-1]), trimmed], dim=0)
        new_frames = int(trimmed.shape[0])
    else:
        trimmed = codes[max(int(prompt_len) - 1, 0) :]
        new_frames = int(trimmed.shape[0]) - frames_sent
    if new_frames <= 0 and not stream_finished:
        if codec_streaming:
            transfer_manager.request_payload[request_id] = {
                "nvc_frames_sent": frames_sent,
                "nvc_frames_seen": frames_seen,
                "nvc_prompt_len": int(prompt_len),
                "nvc_codec_stream_started": True,
                "nvc_pending_codes": trimmed,
            }
            return _empty_streaming_payload(request_id=request_id)
        return None

    connector = getattr(transfer_manager, "connector", None)
    raw_cfg = getattr(connector, "config", {}) or {}
    cfg = raw_cfg.get("extra", raw_cfg) if isinstance(raw_cfg, dict) else {}
    chunk_frames = int(cfg.get("codec_chunk_frames", 13))
    if new_frames < chunk_frames and not stream_finished:
        if codec_streaming:
            transfer_manager.request_payload[request_id] = {
                "nvc_frames_sent": frames_sent,
                "nvc_frames_seen": frames_seen,
                "nvc_prompt_len": int(prompt_len),
                "nvc_codec_stream_started": True,
                "nvc_pending_codes": trimmed,
            }
            return _empty_streaming_payload(request_id=request_id)
        return None
    if new_frames <= 0 and stream_finished and frames_sent > 0:
        # Everything already shipped; emit a terminal empty chunk so the
        # code2wav request observes meta.finished.
        return _empty_finished_payload(
            codec_streaming=codec_streaming,
            request_id=request_id,
        )

    transfer_manager.request_payload[request_id] = {
        "nvc_frames_sent": frames_sent + new_frames if codec_streaming else int(trimmed.shape[0]),
        **({"nvc_frames_seen": frames_seen} if codec_streaming else {}),
        "nvc_prompt_len": int(prompt_len),
        "nvc_codec_stream_started": codec_streaming,
    }
    code_payload = trimmed
    return OmniPayloadStruct(
        codes=CodesStruct(audio=code_payload.detach().to(dtype=torch.long, device="cpu")),
        meta=MetaStruct(
            left_context_size=0 if codec_streaming else frames_sent,
            codec_streaming=codec_streaming,
            request_id=request_id if codec_streaming else None,
            finished=torch.tensor(stream_finished, dtype=torch.bool),
            # Native duplex scheduler segments are transport wakes, not codec
            # stream boundaries.  Preserve polling across successive wakes.
            is_segment_finished=(torch.tensor(False, dtype=torch.bool) if codec_streaming else None),
        ),
    )


def _empty_streaming_payload(*, request_id: str) -> OmniPayloadStruct:
    return OmniPayloadStruct(
        codes=CodesStruct(audio=torch.empty(0, dtype=torch.long)),
        meta=MetaStruct(
            codec_streaming=True,
            request_id=request_id,
            finished=torch.tensor(False, dtype=torch.bool),
            is_segment_finished=torch.tensor(False, dtype=torch.bool),
        ),
    )


def _empty_finished_payload(
    *,
    codec_streaming: bool = False,
    request_id: str | None = None,
) -> OmniPayloadStruct:
    return OmniPayloadStruct(
        codes=CodesStruct(audio=torch.empty(0, dtype=torch.long)),
        meta=MetaStruct(
            finished=torch.tensor(True, dtype=torch.bool),
            codec_streaming=codec_streaming,
            request_id=request_id if codec_streaming else None,
        ),
    )


def talker2code2wav_full_payload(
    transfer_manager: Any = None,
    pooling_output: Any = None,
    request: Any = None,
    is_finished: bool = False,
    **kwargs: Any,
) -> OmniPayloadStruct | None:
    """Producer: ship the talker's prompt-trimmed ``[frames, 31]`` code stack.

    Codes accumulate per frame under ``("codes","audio")`` in the request's
    additional_information; the prompt-region rows were never emitted by the
    talker (it only stores codes for generated positions), so no extra trim is
    needed here. Ships once, at request finish, with replace semantics.
    """
    # NOTE: no is_finished gating — in sync full-payload mode the producer only
    # runs at flush time (request already finished), and the flush-time request
    # object may still report is_finished()=False. Returning a meta-only
    # "pending" struct here would be SENT as the real payload and starve the
    # code2wav stage of its codes.
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
        kwargs.get("multimodal_output"),
    ):
        audio = _codes_from(source)
        if audio is not None:
            break
    if not isinstance(audio, torch.Tensor) or audio.numel() == 0:
        import logging

        logging.getLogger(__name__).error(
            "nemotron_voicechat talker full-payload producer found no codes: "
            "pooling_output keys=%s, request.additional_information keys=%s",
            sorted(pooling_output.keys()) if isinstance(pooling_output, dict) else type(pooling_output).__name__,
            sorted(getattr(request, "additional_information", None) or {})
            if isinstance(getattr(request, "additional_information", None), dict)
            else type(getattr(request, "additional_information", None)).__name__,
        )
        return _empty_finished_payload()
    if audio.ndim == 1:
        audio = audio.reshape(1, -1)
    # Rows correspond to NeMo timeline steps t = 1..T-1. Trim the prompt
    # region: NeMo keeps gen_codes rows t = P..T-1 (P = logical prompt len),
    # i.e. row indices P-1 onward here. P >= 1 always (a system prompt is
    # required), so this never drops acoustic frames.
    prompt_len = _info_get(getattr(request, "additional_information", None), "nvc_logical_prompt_len")
    if prompt_len is None:
        prompt_len = _info_get(getattr(request, "additional_information_cpu", None), "nvc_logical_prompt_len")
    if prompt_len is None:
        # Explicit failure, never a silent trim=0 (untrimmed prompt-region codes
        # would ship to the codec). The connector layer logs producer errors.
        raise ValueError(
            "NemotronVoiceChat talker request is missing 'nvc_logical_prompt_len' in its "
            "additional_information; cannot trim the prompt-region codes for code2wav."
        )
    audio = audio[max(int(prompt_len) - 1, 0) :]
    if audio.shape[0] == 0:
        return _empty_finished_payload()
    # 2-D [frames, Q] payload: the chunk/full-payload transfer adapters route
    # >=2-D tensors through the payload channel (1-D would be coerced into
    # prompt token ids and dropped from the info merge).
    return OmniPayloadStruct(
        codes=CodesStruct(audio=audio.detach().to(dtype=torch.long, device="cpu")),
        meta=MetaStruct(finished=torch.tensor(True, dtype=torch.bool)),
    )


def talker2code2wav_token_only(
    source_outputs: list,
    prompt: Any = None,
    _requires_multimodal_data: bool = False,
) -> list:
    """Sync placeholder builder for stage 2 (codes arrive via the connector)."""
    from vllm_omni.inputs.data import OmniTokensPrompt

    del prompt, _requires_multimodal_data
    inputs: list = []
    for talker_output in source_outputs:
        if not getattr(talker_output, "finished", False):
            continue
        output = talker_output.outputs[0]
        token_ids = getattr(output, "cumulative_token_ids", None) or getattr(output, "token_ids", None) or []
        n_frames = len(token_ids)
        inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=[0] * max(n_frames, 1),
                additional_information=None,
                multi_modal_data=None,
                mm_processor_kwargs=None,
            )
        )
    return inputs


__all__ = [
    "thinker2talker_token_only",
    "thinker2talker_async_chunk",
    "talker2code2wav_full_payload",
    "talker2code2wav_async_chunk",
    "talker2code2wav_token_only",
]
