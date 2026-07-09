# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Audex thinker → code2wav stage input processors.

The thinker is a plain token-autoregressive LM whose sampled stream contains
``<speechcodec_N>`` tokens in a contiguous vocab block. These processors read
the sampled token ids directly (no tokenizer, no multimodal outputs), map them
to codec frame ids by subtracting the vocab offset, and ship single-codebook
frame payloads to the code2wav stage.

The downstream decoder is stateful-streaming (one session per request), so
chunks carry ONLY new frames — no left-context replay — plus a ``finished``
flag whose final (possibly empty) payload triggers the decoder's lookahead
flush.
"""

from typing import Any

import torch
from vllm.logger import init_logger

from vllm_omni.data_entry_keys import CodesStruct, MetaStruct, OmniPayloadStruct
from vllm_omni.inputs.data import OmniTokensPrompt
from vllm_omni.model_executor.stage_input_processors.bagel import ExpandedPrompt

logger = init_logger(__name__)

CFG_UNCOND_SUFFIX = "__cfg_uncond"

# Vocab layout of nvidia/Nemotron-Labs-Audex-2B checkpoint_folder_audiogen:
# <speechcodec_0> .. <speechcodec_65535> occupy a contiguous id block. The
# deploy yaml can override both values; a unit test pins these defaults
# against the real tokenizer.
_DEFAULT_CODEC_TOKEN_OFFSET = 131077
_DEFAULT_CODEC_VOCAB_SIZE = 65536

# TTA <audiocodec_N> block (see models/audex/tta.py; pinned against the real
# tokenizer there). Interleaved 4-codebook RVQ.
_DEFAULT_AUDIOCODEC_TOKEN_OFFSET = 196613
_DEFAULT_AUDIOCODEC_VOCAB_SIZE = 8192
_TTA_NUM_CODEBOOKS = 4

_STATE_KEY = "_audex_async_state"


def _ensure_list(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, torch.Tensor):
        return value.reshape(-1).tolist()
    return list(value)


def _connector_extra(transfer_manager: Any) -> dict[str, Any]:
    connector = getattr(transfer_manager, "connector", None)
    raw_cfg = getattr(connector, "config", {}) or {}
    if isinstance(raw_cfg, dict):
        extra = raw_cfg.get("extra", raw_cfg)
        return extra if isinstance(extra, dict) else {}
    return {}


def _codec_frames(tokens: list[int], codec_offset: int, codec_size: int) -> list[int]:
    """Map ``<speechcodec_N>`` token ids to codec frame ids, dropping all other tokens."""
    return [int(t) - codec_offset for t in tokens if codec_offset <= int(t) < codec_offset + codec_size]


def expand_cfg_prompts(prompt: dict[str, Any] | str, sampling_params: Any) -> list[ExpandedPrompt]:
    """Emit the unconditional CFG companion for a guided Audex TTS request.

    A request opts into CFG by carrying ``extra_args`` with
    ``cfg_scale > 1.0``, ``cfg_role="cond"``, ``cfg_pair_id`` and the
    pre-built length-matched ``cfg_null_prompt`` (constructed where a
    tokenizer is available: the serving layer or the offline example).
    Unguided requests (no/1.0 scale) expand to nothing, keeping the
    disabled path identical to the non-CFG flow.
    """
    extra_args = getattr(sampling_params, "extra_args", None) or {}
    try:
        cfg_scale = float(extra_args.get("cfg_scale", 1.0))
    except (TypeError, ValueError):
        return []
    if cfg_scale <= 1.0 or extra_args.get("cfg_role") != "cond":
        return []

    pair_id = extra_args.get("cfg_pair_id")
    null_prompt = extra_args.get("cfg_null_prompt")
    if not pair_id or not isinstance(null_prompt, str) or not null_prompt:
        raise ValueError(
            "Audex CFG request is missing cfg_pair_id/cfg_null_prompt in extra_args; "
            "the caller must build the length-matched null prompt before submission"
        )

    if isinstance(prompt, dict):
        companion_prompt: dict[str, Any] | str = {**prompt, "prompt": null_prompt}
        companion_prompt.pop("prompt_token_ids", None)
    else:
        companion_prompt = null_prompt

    return [
        ExpandedPrompt(
            prompt=companion_prompt,
            role="uncond",
            request_id_suffix=CFG_UNCOND_SUFFIX,
            sampling_params_override={
                "extra_args": {
                    "cfg_scale": cfg_scale,
                    "cfg_role": "uncond",
                    "cfg_pair_id": pair_id,
                }
            },
        )
    ]


def _extract_new_codes(
    request: Any,
    state: dict[str, Any],
    codec_offset: int,
    codec_size: int,
) -> list[int]:
    """Return codec frame ids from tokens sampled since the last call."""
    output_token_ids = _ensure_list(getattr(request, "output_token_ids", []))
    new_tokens = output_token_ids[int(state.get("seen_len", 0)) :]
    state["seen_len"] = len(output_token_ids)
    return _codec_frames(new_tokens, codec_offset, codec_size)


def thinker2code2wav_async_chunk(
    transfer_manager: Any,
    multimodal_output: dict[str, Any] | None,
    request: Any,
    is_finished: bool = False,
) -> OmniPayloadStruct | None:
    """Async-chunk producer: sampled token stream → per-chunk codec frames."""
    del multimodal_output  # thinker emits none; codes come from the token stream
    request_id = request.external_req_id
    finished = bool(is_finished or request.is_finished())

    cfg = _connector_extra(transfer_manager)
    chunk_frames = int(cfg.get("codec_chunk_frames", 25))
    initial_chunk_frames = int(cfg.get("initial_codec_chunk_frames", chunk_frames))
    codec_offset = int(cfg.get("codec_token_offset", _DEFAULT_CODEC_TOKEN_OFFSET))
    codec_size = int(cfg.get("codec_vocab_size", _DEFAULT_CODEC_VOCAB_SIZE))
    if chunk_frames <= 0 or initial_chunk_frames <= 0:
        raise ValueError(
            f"Invalid codec chunk config: codec_chunk_frames={chunk_frames}, "
            f"initial_codec_chunk_frames={initial_chunk_frames}"
        )

    request_state = transfer_manager.request_payload.get(request_id)
    if not isinstance(request_state, dict) or _STATE_KEY not in request_state:
        request_state = {
            _STATE_KEY: {
                "seen_len": 0,
                "pending": [],
                "emitted_chunks": 0,
                "total_frames": 0,
                "terminal_sent": False,
            }
        }
        transfer_manager.request_payload[request_id] = request_state
    state = request_state[_STATE_KEY]
    if state.get("terminal_sent", False):
        return None

    pending: list[int] = state["pending"]
    new_codes = _extract_new_codes(request, state, codec_offset, codec_size)
    pending.extend(new_codes)
    state["total_frames"] = int(state.get("total_frames", 0)) + len(new_codes)

    if finished and state["total_frames"] == 0:
        # Fail fast: the transfer adapter logs this and still delivers the
        # terminal marker, and the serving layer turns the audio-less request
        # into a request-level error instead of a silent empty stream.
        state["terminal_sent"] = True
        raise ValueError(f"Audex thinker produced no codec tokens for request {request_id}")

    # Hold back at least one frame on non-terminal emissions so the terminal
    # chunk always carries codes: the generation scheduler never runs a final
    # forward for an empty terminal chunk, and the decoder's lookahead tail is
    # only flushed inside a forward that sees the finished flag.
    target = initial_chunk_frames if state["emitted_chunks"] == 0 else chunk_frames
    if not finished and len(pending) <= target:
        return None

    if finished:
        emit, state["pending"] = pending, []
        state["terminal_sent"] = True
    else:
        emit, state["pending"] = pending[:target], pending[target:]

    state["emitted_chunks"] += 1
    finished_flag = torch.tensor(finished, dtype=torch.bool)
    # Shape [frames, 1]: 2-D codec payloads ride the connector's payload
    # channel into ``runtime_additional_information`` (1-D tensors are
    # rerouted onto the token path and never reach the model's payload view).
    return OmniPayloadStruct(
        codes=CodesStruct(audio=torch.tensor(emit, dtype=torch.long).reshape(-1, 1)),
        meta=MetaStruct(
            finished=finished_flag,
            # The chunk adapter strips ``meta.finished`` before it reaches the
            # stage-1 model; ``stream_finished`` survives and is what
            # AudexCode2Wav uses to trigger the lookahead flush.
            stream_finished=finished_flag,
            req_id=[request_id],
        ),
    )


def thinker2code2wav_full_payload(
    transfer_manager: Any,
    pooling_output: dict[str, Any] | None,
    request: Any,
) -> dict[str, Any] | None:
    """Sync-path producer: ship the full codec sequence once at finish."""
    del pooling_output  # thinker emits none; codes come from the token stream
    cfg = _connector_extra(transfer_manager)
    codec_offset = int(cfg.get("codec_token_offset", _DEFAULT_CODEC_TOKEN_OFFSET))
    codec_size = int(cfg.get("codec_vocab_size", _DEFAULT_CODEC_VOCAB_SIZE))

    output_token_ids = _ensure_list(getattr(request, "output_token_ids", []))
    codes = _codec_frames(output_token_ids, codec_offset, codec_size)
    request_id = getattr(request, "external_req_id", None) or getattr(request, "request_id", None)
    if not codes:
        raise ValueError(f"Audex thinker produced no codec tokens for request {request_id}")
    return {
        "codes": {"audio": codes},
        "meta": {
            "finished": torch.tensor(True, dtype=torch.bool),
            "req_id": [request_id],
        },
    }


def thinker2xcodec_full_payload(
    transfer_manager: Any,
    pooling_output: dict[str, Any] | None,
    request: Any,
) -> dict[str, Any] | None:
    """TTA sync-path producer: ship the de-interleaved RVQ payload at finish.

    Filters ``<audiocodec_N>`` ids from the sampled stream, truncates to whole
    4-token frames, and ships a ``[frames, 4]`` codec-id payload (offsets NOT
    subtracted; the XCodec1 stage removes the per-layer offset).
    """
    del pooling_output  # thinker emits none; codes come from the token stream
    cfg = _connector_extra(transfer_manager)
    codec_offset = int(cfg.get("audiocodec_token_offset", _DEFAULT_AUDIOCODEC_TOKEN_OFFSET))
    codec_size = int(cfg.get("audiocodec_vocab_size", _DEFAULT_AUDIOCODEC_VOCAB_SIZE))

    output_token_ids = _ensure_list(getattr(request, "output_token_ids", []))
    codes = _codec_frames(output_token_ids, codec_offset, codec_size)
    request_id = getattr(request, "external_req_id", None) or getattr(request, "request_id", None)
    if not codes:
        raise ValueError(f"Audex TTA thinker produced no codec tokens for request {request_id}")
    usable = len(codes) - (len(codes) % _TTA_NUM_CODEBOOKS)
    if usable == 0:
        raise ValueError(f"Audex TTA thinker produced fewer than one full RVQ frame for request {request_id}")

    # Validate the ACTUAL generated stream against the 4-codebook phase
    # contract before decode: XCodec's per-layer offset removal clamps
    # invalid ids into range, which would silently mask a phase violation.
    from vllm_omni.model_executor.models.audex.tta import validate_rvq_phase

    validity = validate_rvq_phase(codes[:usable])
    if not validity["phase_valid"]:
        raise ValueError(
            f"Audex TTA thinker produced a phase-invalid RVQ stream for request {request_id}: "
            f"{validity['mismatch_count']} mismatched positions "
            f"(first mismatch [index, codec_id, got_phase, expected_phase] = {validity['first_mismatch']})"
        )

    frames = torch.tensor(codes[:usable], dtype=torch.long).reshape(-1, _TTA_NUM_CODEBOOKS)
    return {
        "codes": {"audio": frames},
        "meta": {
            "finished": torch.tensor(True, dtype=torch.bool),
            "req_id": [request_id],
        },
    }


def thinker2code2wav_token_only(
    source_outputs: list[Any],
    prompt: Any = None,
    _requires_multimodal_data: bool = False,
):
    """Sync-side stage-1 engine-input builder (length placeholder).

    The bulk codec payload ships via the worker connector from
    ``thinker2code2wav_full_payload``; this only sizes the stage-1 request.
    """
    del prompt
    engine_inputs: list[OmniTokensPrompt] = []
    for source_output in source_outputs:
        if not source_output.finished:
            continue
        output = source_output.outputs[0]
        prompt_ids = _ensure_list(source_output.prompt_token_ids)
        raw_output_ids = _ensure_list(output.cumulative_token_ids)
        if raw_output_ids[: len(prompt_ids)] == prompt_ids:
            raw_output_ids = raw_output_ids[len(prompt_ids) :]
        engine_inputs.append(OmniTokensPrompt(prompt_token_ids=raw_output_ids))
    return engine_inputs
