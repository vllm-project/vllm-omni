# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage-input processors for MiniMax Music 3: AR talker -> acoustic decoder.

Three entry points, all referenced from
``vllm_omni/model_executor/models/minimax_music3/pipeline.py``:

* ``expand_cfg_prompts`` (stage 0 ``prompt_expand_func``) builds the
  unconditioned twin of a conditioned request. Classifier-free guidance is
  mandatory for this checkpoint, so in practice every request expands.
* ``ar2acoustic_async_chunk`` (stage 0 ``async_chunk_process_next_stage_input_func``)
  forwards one AR frame span per call to stage 1, and is also the flush point
  for the request's tail.
* ``ar2acoustic`` (stage 1 ``sync_process_input_func``) builds the stage-1
  engine input for the non-streaming path, where the whole song arrives at
  once after stage 0 finishes.

Stage-0 emit contract
---------------------
Per emission the talker publishes a multimodal output shaped like::

    {
        "latent": FloatTensor[1, T, 32768] or FloatTensor[T, 32768],
        "codes": {"audio": LongTensor[T, 8]},
        "meta": {
            "num_processed_tokens": int,   # global AR frame index of frame 0
            "last_chunk": bool,            # True only on the terminal emission
            "audio_seed": int,             # optional
        },
    }

The span is normally one window: ``T == AR_CHUNK_FRAMES``, windows advance by
``AR_CHUNK_HOP_FRAMES``, and window ``k`` starts at AR frame
``k * AR_CHUNK_HOP_FRAMES``. The terminal emission is the exception. A window
cannot be proven non-final until a further frame exists, so the talker holds
the most recently completed window back and, on the step where it knows the
request is ending, publishes the whole remaining frame span at once. That span
covers up to two windows (the held-back complete one plus the truncated final
one), so ``T`` can reach ``AR_CHUNK_FRAMES + AR_CHUNK_HOP_FRAMES``. Stage 1
re-derives the window boundaries inside a span from
``meta.num_processed_tokens`` and the span length through
``models/minimax_music3/chunking.chunk_windows``.

``meta.num_processed_tokens`` is required rather than derived: once a terminal
span carries more than one window, a running ``chunk_seq * hop`` count no
longer identifies the span's first frame. The derived value is kept only as a
fallback for pre-terminal emissions.

``hidden_states.output`` is accepted as an alias for ``latent`` because both
are legal homes for a float payload; ``latent`` is canonical and matches the
stage's ``engine_output_type``.

Stage-0 -> stage-1 payload
--------------------------
Every key below is a declared field of ``OmniPayloadStruct`` /
``MetaStruct``, which are ``forbid_unknown_fields=True`` msgspec structs, so
an undeclared key is a construction error rather than silent data loss::

    latent                     FloatTensor[T, 32768]  span conditioning, bf16
    codes.audio                LongTensor[T, 8]       RVQ codes for the span
    meta.chunk_seq             int    emission index; is_first == (chunk_seq == 0)
    meta.num_processed_tokens  int    absolute AR frame index of the span start
    meta.codec_chunk_frames    int    T; span end (exclusive) = start + T
    meta.last_chunk            bool   terminal span
    meta.stream_finished       BoolTensor  terminal marker that survives transport
    meta.audio_seed            int    seed for the acoustic stage's noise draw
    meta.req_id                list[str]   single-element external request id
    meta.finished              BoolTensor  overwritten by the chunk adapter

``codes.audio`` is rank 2 on purpose: the chunk transfer adapter reroutes rank-1
code tensors onto the token path (``.tolist()``), while rank-2 tensors ride the
payload channel and leave a single ``[0]`` placeholder prompt token per chunk.
``meta.finished`` is stripped before the payload reaches the stage-1 model, so
``meta.stream_finished`` and ``meta.last_chunk`` are what stage 1 reads.

The overlap crop is deliberately NOT in the payload. Stage 1 recovers it from
the span's global frame range and ``last_chunk`` through
``models/minimax_music3/chunking.crop_sample_bounds``, which is the single
owner of that arithmetic; duplicating it here would let the two copies drift.
"""

from __future__ import annotations

from typing import Any

import torch
from vllm.logger import init_logger

from vllm_omni.data_entry_keys import CodesStruct, MetaStruct, OmniPayloadStruct
from vllm_omni.inputs.data import OmniTokensPrompt
from vllm_omni.model_executor.models.minimax_music3.constants import (
    AR_CFG_SCALE,
    AR_CHUNK_HOP_FRAMES,
    AR_HIDDEN_SIZE,
    NUM_CODEBOOKS,
)
from vllm_omni.model_executor.models.minimax_music3.prompt import build_cfg_null_token_ids
from vllm_omni.model_executor.stage_input_processors.bagel import ExpandedPrompt

__all__ = [
    "CFG_UNCOND_SUFFIX",
    "ar2acoustic",
    "ar2acoustic_async_chunk",
    "expand_cfg_prompts",
]

logger = init_logger(__name__)

CFG_UNCOND_SUFFIX = "__cfg_uncond"

_COND = "cond"
_UNCOND = "uncond"

# Per-request forwarder state, parked on the transfer manager's request map.
_STATE_KEY = "_minimax_music3_async_state"

# The window conditioning is emitted in bfloat16. Each chunk is its own
# /dev/shm segment and consecutive windows overlap by half, so a float32
# 200-frame window would move 26 MB of mostly-redundant bytes per hop against
# 13 MB in bfloat16. The backbone runs in bfloat16 anyway, so this discards no
# information the AR stage actually produced.
_LATENT_TRANSPORT_DTYPE = torch.bfloat16


# ---------------------------------------------------------------------------
# Stage 0: classifier-free-guidance prompt expansion
# ---------------------------------------------------------------------------


def _prompt_token_ids(prompt: dict[str, Any] | str | Any) -> list[int] | None:
    """Return the prompt's token ids, or ``None`` when only text is available."""
    token_ids: Any = None
    if isinstance(prompt, dict):
        token_ids = prompt.get("prompt_token_ids")
    elif isinstance(prompt, (list, tuple)):
        token_ids = prompt
    if token_ids is None:
        return None
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.reshape(-1).tolist()
    if not isinstance(token_ids, (list, tuple)) or not token_ids:
        return None
    return [int(t) for t in token_ids]


def _uncond_extra_args(extra_args: dict[str, Any], cfg_scale: float) -> dict[str, Any]:
    """Clone the conditioned row's extra_args for the unconditioned twin.

    Every key is carried through, not only the ``cfg_*`` ones: the stage runs
    with ``has_sampling_extra_args`` on, so extra_args reach the model's
    ``forward`` per row, and dropping a key here would make the two rows
    execute under different settings and decode out of lockstep.
    ``cfg_null_prompt`` is the exception: the companion IS the null prompt.
    """
    companion = dict(extra_args)
    companion.pop("cfg_null_prompt", None)
    companion["cfg_role"] = _UNCOND
    companion["cfg_scale"] = cfg_scale
    return companion


def expand_cfg_prompts(
    prompt: dict[str, Any] | str,
    sampling_params: Any,
) -> list[ExpandedPrompt]:
    """Emit the unconditioned companion for a guided MiniMax Music 3 request.

    Classifier-free guidance is not optional for this checkpoint, and the
    deploy YAMLs set ``cfg_role: cond`` / ``cfg_scale: 1.5`` as stage-0
    defaults, so an ordinary request expands into exactly one companion and
    the pair decodes in lockstep.

    The companion's token ids come from ``build_cfg_null_token_ids``, which
    overwrites the caption-and-lyrics span with ``<|audio_cfg|>`` while keeping
    the framing tokens. Token ids are preferred over text because the two rows
    are only blendable when they have identical length and an identical
    ``<|audio_start|>`` position, and only the token-id form guarantees that
    without a tokenizer round trip.

    Args:
        prompt: The raw user prompt. A dict carrying ``prompt_token_ids`` is
            the preferred form; a text-only prompt must supply a pre-built,
            length-matched ``cfg_null_prompt`` in ``extra_args``.
        sampling_params: Stage-0 sampling params for the conditioned row.

    Returns:
        A one-element list with the unconditioned companion, or an empty list
        when the request is not CFG-guided.

    Raises:
        ValueError: If the request is guided but neither prompt token ids nor
            a pre-built ``cfg_null_prompt`` is available.
    """
    extra_args = getattr(sampling_params, "extra_args", None) or {}
    if extra_args.get("cfg_role") != _COND:
        # Not guided, or this IS the companion. Either way, nothing to expand.
        return []
    try:
        cfg_scale = float(extra_args.get("cfg_scale", AR_CFG_SCALE))
    except (TypeError, ValueError):
        logger.warning(
            "MiniMax Music 3: ignoring non-numeric cfg_scale %r; request decodes unguided",
            extra_args.get("cfg_scale"),
        )
        return []
    if cfg_scale <= 1.0:
        return []

    token_ids = _prompt_token_ids(prompt)
    companion_prompt: dict[str, Any] | str
    if token_ids is not None:
        null_ids = build_cfg_null_token_ids(token_ids)
        if isinstance(prompt, dict):
            companion_prompt = {**prompt, "prompt_token_ids": null_ids}
            # Drop the text form so the input processor does not re-tokenize
            # and hand back a row of a different length.
            companion_prompt.pop("prompt", None)
            companion_prompt.pop("prompt_embeds", None)
        else:
            companion_prompt = {"prompt_token_ids": null_ids}
    else:
        null_prompt = extra_args.get("cfg_null_prompt")
        if not isinstance(null_prompt, str) or not null_prompt:
            raise ValueError(
                "MiniMax Music 3 CFG request carries neither prompt_token_ids nor a "
                "pre-built cfg_null_prompt in extra_args. Submit the prompt as token "
                "ids (preferred; the null twin is then built by "
                "models.minimax_music3.prompt.build_cfg_null_token_ids) or attach a "
                "length-matched cfg_null_prompt string."
            )
        if isinstance(prompt, dict):
            companion_prompt = {**prompt, "prompt": null_prompt}
            companion_prompt.pop("prompt_token_ids", None)
            companion_prompt.pop("prompt_embeds", None)
        else:
            companion_prompt = null_prompt

    return [
        ExpandedPrompt(
            prompt=companion_prompt,
            role=_UNCOND,
            request_id_suffix=CFG_UNCOND_SUFFIX,
            sampling_params_override={"extra_args": _uncond_extra_args(extra_args, cfg_scale)},
        )
    ]


# ---------------------------------------------------------------------------
# Stage 0 -> stage 1: async-chunk window forwarder
# ---------------------------------------------------------------------------


def _payload_section(payload: Any, key: str) -> dict[str, Any]:
    """Return a nested payload section as a dict (empty when absent)."""
    if not isinstance(payload, dict):
        return {}
    section = payload.get(key)
    return section if isinstance(section, dict) else {}


def _window_latent(payload: Any) -> torch.Tensor | None:
    """Return this window's conditioning as ``[T, 32768]``, or ``None``.

    ``latent`` is the canonical key; ``hidden_states.output`` is accepted for
    talkers that publish through the hidden-state channel instead. A leading
    singleton batch dimension is squeezed so stage 1 sees a stable rank.
    """
    if not isinstance(payload, dict):
        return None
    tensor = payload.get("latent")
    if not isinstance(tensor, torch.Tensor):
        tensor = _payload_section(payload, "hidden_states").get("output")
    if not isinstance(tensor, torch.Tensor) or tensor.numel() == 0:
        return None
    if tensor.ndim == 3:
        if tensor.shape[0] != 1:
            raise ValueError(
                f"MiniMax Music 3 window latent must have a singleton batch dim; got {tuple(tensor.shape)}"
            )
        tensor = tensor[0]
    if tensor.ndim != 2:
        raise ValueError(f"MiniMax Music 3 window latent must be rank 2 or 3; got {tuple(tensor.shape)}")
    if tensor.shape[1] != AR_HIDDEN_SIZE:
        raise ValueError(f"MiniMax Music 3 window latent has {tensor.shape[1]} features; expected {AR_HIDDEN_SIZE}")
    return tensor


def _window_codes(payload: Any) -> torch.Tensor | None:
    """Return this window's RVQ codes as ``[T, 8]``, or ``None``."""
    codes = _payload_section(payload, "codes").get("audio")
    if not isinstance(codes, torch.Tensor) or codes.numel() == 0:
        return None
    if codes.ndim == 1:
        if codes.numel() % NUM_CODEBOOKS != 0:
            raise ValueError(
                f"MiniMax Music 3 flat code payload of length {codes.numel()} is not a "
                f"multiple of {NUM_CODEBOOKS} codebooks"
            )
        codes = codes.reshape(-1, NUM_CODEBOOKS)
    if codes.ndim != 2 or codes.shape[1] != NUM_CODEBOOKS:
        raise ValueError(f"MiniMax Music 3 codes must be [T, {NUM_CODEBOOKS}]; got {tuple(codes.shape)}")
    return codes.to(torch.long)


def _request_audio_seed(request: Any, payload: Any) -> int | None:
    """Resolve the seed the acoustic stage draws its noise from.

    A seed published by the talker wins, because a talker that already
    consumed the request seed knows which value it used. Otherwise fall back
    to the stage-0 request's own sampling seed.
    """
    published = _payload_section(payload, "meta").get("audio_seed")
    if isinstance(published, int):
        return published
    seed = getattr(getattr(request, "sampling_params", None), "seed", None)
    return int(seed) if isinstance(seed, int) else None


def _forwarder_state(transfer_manager: Any, request_id: str) -> dict[str, Any]:
    """Return (creating if needed) the per-request forwarder state."""
    request_state = transfer_manager.request_payload.get(request_id)
    if not isinstance(request_state, dict) or _STATE_KEY not in request_state:
        request_state = {_STATE_KEY: {"chunk_seq": 0, "terminal_sent": False}}
        transfer_manager.request_payload[request_id] = request_state
    return request_state[_STATE_KEY]


def ar2acoustic_async_chunk(
    transfer_manager: Any,
    multimodal_output: dict[str, Any] | None,
    request: Any,
    is_finished: bool = False,
) -> OmniPayloadStruct | None:
    """Forward one AR frame span to the acoustic stage.

    The talker publishes whole spans, so this is a relay plus bookkeeping
    rather than an accumulator: it stamps the emission index, the span's global
    start frame, its length and the terminal flags, then hands the tensors to
    the connector.

    This hook is the flush point for the request's tail. The AR scheduler calls
    ``save_async`` on the step that stops the request, including when that
    step produced no inter-stage output, so this function is guaranteed one
    call with ``is_finished=True`` (omni_ar_scheduler.py:551, base
    chunk_transfer_adapter.py:341). Whatever the talker published on that step
    therefore reaches stage 1 in the same hop.

    Returns ``None`` on steps where the talker released no span and the request
    is still running.
    """
    request_id = request.external_req_id
    finished = bool(is_finished or request.is_finished())

    state = _forwarder_state(transfer_manager, request_id)

    latent = _window_latent(multimodal_output)
    codes = _window_codes(multimodal_output)
    published_meta = _payload_section(multimodal_output, "meta")
    published_last = published_meta.get("last_chunk") is True

    if state["terminal_sent"]:
        if latent is not None:
            logger.error(
                "MiniMax Music 3 talker emitted %d more frames for request %s after the "
                "terminal span; the downstream stream is already closed and they are dropped",
                int(latent.shape[0]),
                request_id,
            )
        return None

    if latent is None:
        if not finished:
            return None
        # Terminal step with nothing left to send: still deliver the marker so
        # stage 1 can flush and complete instead of polling until timeout.
        state["terminal_sent"] = True
        # The unconditioned twin never publishes a span: the pair shares one
        # song and the conditioned row emits it. Its silence is the contract,
        # not a failure.
        if state["chunk_seq"] == 0 and not request_id.endswith(CFG_UNCOND_SUFFIX):
            logger.error("MiniMax Music 3 talker produced no audio for request %s", request_id)
        return OmniPayloadStruct(
            meta=MetaStruct(
                chunk_seq=state["chunk_seq"],
                last_chunk=True,
                stream_finished=torch.tensor(True, dtype=torch.bool),
                finished=torch.tensor(True, dtype=torch.bool),
                req_id=[request_id],
            )
        )

    span_frames = int(latent.shape[0])
    if codes is not None and codes.shape[0] != span_frames:
        raise ValueError(
            f"MiniMax Music 3 span mismatch for request {request_id}: latent has "
            f"{span_frames} frames but codes have {codes.shape[0]}"
        )

    chunk_seq = int(state["chunk_seq"])
    state["chunk_seq"] = chunk_seq + 1

    # A span is terminal when the engine stopped the request or the talker said
    # so. Only the engine's verdict closes the stream, so a talker that flags
    # a span last while generation continues is reported rather than obeyed.
    terminal = bool(finished or published_last)
    if published_last and not finished:
        logger.warning(
            "MiniMax Music 3 talker flagged a terminal span for request %s while the "
            "engine still has the request running",
            request_id,
        )
    state["terminal_sent"] = terminal

    span_start = published_meta.get("num_processed_tokens")
    if not isinstance(span_start, int):
        # Fallback for talkers that publish one hop-aligned window per step.
        # It is only sound before the terminal span, which may cover more than
        # one window and therefore break the running count.
        span_start = chunk_seq * AR_CHUNK_HOP_FRAMES
        if terminal and span_frames > AR_CHUNK_HOP_FRAMES:
            logger.warning(
                "MiniMax Music 3 terminal span for request %s carries %d frames but no "
                "meta.num_processed_tokens; the derived start frame %d assumes a single "
                "hop-aligned window and may misplace the tail",
                request_id,
                span_frames,
                span_start,
            )

    finished_flag = torch.tensor(terminal, dtype=torch.bool)
    meta = MetaStruct(
        chunk_seq=chunk_seq,
        num_processed_tokens=int(span_start),
        codec_chunk_frames=span_frames,
        last_chunk=terminal,
        # meta.finished is stripped before the payload reaches the stage-1
        # model; stream_finished is what survives and drives the final flush.
        stream_finished=finished_flag,
        finished=finished_flag,
        req_id=[request_id],
    )
    audio_seed = _request_audio_seed(request, multimodal_output)
    if audio_seed is not None:
        meta.audio_seed = audio_seed

    payload = OmniPayloadStruct(
        latent=latent.detach().to(_LATENT_TRANSPORT_DTYPE).contiguous(),
        meta=meta,
    )
    if codes is not None:
        payload.codes = CodesStruct(audio=codes.detach().contiguous())
    return payload


# ---------------------------------------------------------------------------
# Stage 1: non-streaming engine-input builder
# ---------------------------------------------------------------------------


def _empty_acoustic_prompt() -> OmniTokensPrompt:
    """Zero-length stage-1 request.

    Stage 1 decodes it to empty audio, which the serving layer's no-audio
    guard turns into a request-level error. That is preferable to raising
    here, where the failure would surface as a stalled request.
    """
    return OmniTokensPrompt(
        prompt_token_ids=[],
        multi_modal_data=None,
        mm_processor_kwargs=None,
        additional_information=None,
    )


def ar2acoustic(
    source_outputs: list[Any],
    prompt: Any = None,
    _requires_multimodal_data: bool = False,
) -> list[OmniTokensPrompt]:
    """Build stage-1 engine inputs for the non-streaming path.

    With ``async_chunk: false`` the whole song arrives once stage 0 has
    finished, so the song is handed over as a single logical window and stage
    1 does its own windowing through ``chunking.chunk_windows``. The payload
    layout matches the streaming one key for key, so the acoustic stage reads
    one contract in both modes.

    The prompt is ``[0] * num_windows``: the generation scheduler sizes the
    request from ``prompt_token_ids``, and one placeholder per acoustic window
    keeps that budget proportional to the work without shipping the
    conditioning through the token path.
    """
    del prompt, _requires_multimodal_data

    acoustic_inputs: list[OmniTokensPrompt] = []
    for ar_output in source_outputs:
        if not ar_output.finished:
            continue
        output = ar_output.outputs[0]
        multimodal_output = getattr(output, "multimodal_output", None)

        try:
            latent = _window_latent(multimodal_output)
            codes = _window_codes(multimodal_output)
        except ValueError as exc:
            logger.warning("Skipping malformed MiniMax Music 3 stage-0 output: %s", exc)
            acoustic_inputs.append(_empty_acoustic_prompt())
            continue

        if latent is None:
            logger.error("MiniMax Music 3 talker produced no audio for request %s", ar_output.request_id)
            acoustic_inputs.append(_empty_acoustic_prompt())
            continue

        total_frames = int(latent.shape[0])
        meta: dict[str, Any] = {
            "chunk_seq": 0,
            "num_processed_tokens": 0,
            "codec_chunk_frames": total_frames,
            "last_chunk": True,
            "stream_finished": torch.tensor(True, dtype=torch.bool),
            "req_id": [ar_output.request_id],
        }
        published_seed = _payload_section(multimodal_output, "meta").get("audio_seed")
        if isinstance(published_seed, int):
            meta["audio_seed"] = published_seed

        additional_information: dict[str, Any] = {
            "latent": latent.detach().to(_LATENT_TRANSPORT_DTYPE).contiguous(),
            "meta": meta,
        }
        if codes is not None:
            additional_information["codes"] = {"audio": codes.detach().contiguous()}

        num_windows = max(1, -(-total_frames // AR_CHUNK_HOP_FRAMES))
        acoustic_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=[0] * num_windows,
                multi_modal_data=None,
                mm_processor_kwargs=None,
                additional_information=additional_information,
            )
        )
    return acoustic_inputs
