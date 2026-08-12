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
accumulates one 31-quantizer code stack per frame under ``("codes","audio")``;
the producer ships the prompt-trimmed ``[frames, 31]`` stack once the request
finishes, and the sync placeholder builder sizes stage-2's dummy prompt.
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

# Full-sequence tensors are re-shipped whole; the full-payload accumulator must
# replace (not concatenate) them if the producer fires more than once. The AR
# runner flattens nested payloads before accumulation, so the effective key is
# the flattened "codes.audio" (the unflattened root is kept for robustness).
_FULL_PAYLOAD_REPLACE_KEYS: frozenset[str] = frozenset({"codes", "codes.audio", "meta", "meta.nvc_logical_prompt_len"})

# <SPECIAL_12> in the Nemotron-Nano-9B-v2 vocab; the checkpoint config's
# stt pad_token. The thinker also reports it in its latent metadata, which
# takes precedence when present.
_DEFAULT_TEXT_PAD_ID = 12


def _info_get(info: Any, key: str) -> Any:
    if isinstance(info, dict):
        if key in info:
            return info[key]
        meta = info.get("meta")
        if isinstance(meta, dict):
            return meta.get(key)
    return None


def thinker2talker_token_only(
    source_outputs: list,
    prompt: Any = None,
    _requires_multimodal_data: bool = False,
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
                # The vLLM-side prompt is one placeholder (timeline position 0;
                # the NeMo TTS loop starts at t=1). Each decode step consumes
                # one timeline position from additional_information, so the
                # talker needs len(timeline) - 1 decode steps.
                prompt_token_ids=[0],
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
    del multimodal_output, kwargs
    request_id = request.external_req_id
    prompt_ids = list(getattr(request, "prompt_token_ids", None) or [])
    # vLLM prompt = logical prompt + 1 placeholder (acoustic frame 0).
    logical_prompt_len = max(len(prompt_ids) - 1, 0)
    generated = list(getattr(request, "output_token_ids", None) or [])
    pad_id = _DEFAULT_TEXT_PAD_ID
    reported_pad = _info_get(getattr(request, "additional_information", None), "nvc_text_pad_id")
    if reported_pad is not None:
        pad_id = int(reported_pad)
    timeline = [pad_id] * logical_prompt_len + [int(t) for t in generated]

    # Skip no-progress wakeups (same length as the last emitted chunk).
    state = transfer_manager.request_payload.get(request_id)
    last_len = int(state.get("nvc_timeline_len", -1)) if isinstance(state, dict) else -1
    if len(timeline) <= last_len and not is_finished:
        return None
    transfer_manager.request_payload[request_id] = {"nvc_timeline_len": len(timeline)}

    return OmniPayloadStruct(
        ids=IdsStruct(all=timeline),
        meta=MetaStruct(
            finished=torch.tensor(bool(is_finished), dtype=torch.bool),
            num_processed_tokens=logical_prompt_len,
            # The talker's vLLM prompt stays a single placeholder token.
            next_stage_prompt_len=1,
        ),
    )


def talker2code2wav_async_chunk(
    transfer_manager: Any,
    multimodal_output: Any = None,
    request: Any = None,
    is_finished: bool = False,
    **kwargs: Any,
) -> OmniPayloadStruct | None:
    """Streaming producer: ship prompt-trimmed CUMULATIVE code stacks in chunks.

    Follows the NeMo incremental-decode recipe (``decode_one_audio_step``):
    each chunk carries the FULL trimmed code history and
    ``meta.left_context_size`` = frames the codec already emitted, so the
    code2wav stage re-decodes the whole prefix and slices off only the new
    samples — no seams, no crossfade, and the concatenated stream equals the
    prefix-decode of the final stack. Chunk cadence comes from the connector
    config key ``codec_chunk_frames`` (default 13 frames ~= 1 s of audio).
    """
    del kwargs
    request_id = request.external_req_id

    codes = None
    if isinstance(multimodal_output, dict):
        nested = multimodal_output.get("codes")
        codes = nested.get("audio") if isinstance(nested, dict) else None
        if codes is None:
            codes = multimodal_output.get("codes.audio")
    if not isinstance(codes, torch.Tensor) or codes.numel() == 0:
        if not is_finished:
            return None
        return _empty_finished_payload()
    if codes.ndim == 1:
        codes = codes.reshape(1, -1)

    # Prompt-region trim (rows are NeMo timeline steps t=1..; keep t >= P).
    # Read the prompt length from the PER-STEP model payload first: the
    # request-level additional_information is overwritten both by arriving
    # thinker chunks and by talker info updates, so it races. Cache the first
    # sighting in the transfer-manager state as a fallback.
    state = transfer_manager.request_payload.get(request_id)
    prompt_len = _info_get(multimodal_output if isinstance(multimodal_output, dict) else None, "nvc_logical_prompt_len")
    if prompt_len is None and isinstance(multimodal_output, dict):
        prompt_len = multimodal_output.get("meta.nvc_logical_prompt_len")
    if prompt_len is None:
        info = getattr(request, "additional_information", None)
        prompt_len = _info_get(info, "nvc_logical_prompt_len")
        if prompt_len is None:
            prompt_len = _info_get(info, "num_processed_tokens")
    if prompt_len is None and isinstance(state, dict):
        prompt_len = state.get("nvc_prompt_len")
    if prompt_len is None:
        raise ValueError(
            "NemotronVoiceChat talker request is missing its logical prompt length "
            "('nvc_logical_prompt_len' or async meta.num_processed_tokens); cannot "
            "trim the prompt-region codes for streaming code2wav."
        )
    trimmed = codes[max(int(prompt_len) - 1, 0) :]

    frames_sent = int(state.get("nvc_frames_sent", 0)) if isinstance(state, dict) else 0
    new_frames = int(trimmed.shape[0]) - frames_sent
    if new_frames <= 0 and not is_finished:
        return None

    connector = getattr(transfer_manager, "connector", None)
    raw_cfg = getattr(connector, "config", {}) or {}
    cfg = raw_cfg.get("extra", raw_cfg) if isinstance(raw_cfg, dict) else {}
    chunk_frames = int(cfg.get("codec_chunk_frames", 13))
    if new_frames < chunk_frames and not is_finished:
        return None
    if new_frames <= 0 and is_finished and frames_sent > 0:
        # Everything already shipped; emit a terminal empty chunk so the
        # code2wav request observes meta.finished.
        return _empty_finished_payload()

    transfer_manager.request_payload[request_id] = {
        "nvc_frames_sent": int(trimmed.shape[0]),
        "nvc_prompt_len": int(prompt_len),
    }
    return OmniPayloadStruct(
        codes=CodesStruct(audio=trimmed.detach().to(dtype=torch.long, device="cpu")),
        meta=MetaStruct(
            left_context_size=frames_sent,
            finished=torch.tensor(bool(is_finished), dtype=torch.bool),
        ),
    )


def _empty_finished_payload() -> OmniPayloadStruct:
    return OmniPayloadStruct(
        codes=CodesStruct(audio=torch.empty(0, dtype=torch.long)),
        meta=MetaStruct(finished=torch.tensor(True, dtype=torch.bool)),
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
