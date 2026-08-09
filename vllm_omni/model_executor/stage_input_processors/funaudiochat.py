from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch
from vllm.logger import init_logger

from vllm_omni.data_entry_keys import (
    CodesStruct,
    MetaStruct,
    OmniPayloadStruct,
)
from vllm_omni.inputs.data import OmniTokensPrompt

_MAX_COSYVOICE_TOKEN_ID = 6561
logger = init_logger(__name__)

# Streaming chunk hops, expressed in CRQ codec frames (CosyVoice speech tokens
# are 25 frames/s -> 40ms each, so 25 frames == 1s of audio).
#
# NOTE on the "official hop": FunAudioChat's *non-streaming* code2wav decodes
# whole 30-second *segments* (``_OFFICIAL_TOKEN_HOP_LEN = 25 * 30 = 750`` in
# funaudiochat_code2wav.py, used by ``_decode_segment_like_official``). The
# values below are the *streaming* chunk hop — much smaller so the client gets
# audio incrementally during stage-0 decode. They are NOT derived from the 750
# segment hop; 25 was chosen because even a single 25-frame chunk exceeds the
# flow model's ``pre_lookahead`` (3) and yields a valid mel/speech suffix, and
# 50 frames keeps the same CosyVoice3 flow/HiFT semantics while halving the
# steady-state Stage1 call count relative to a 25-frame (1s) hop.
_FAC_TOKEN_HOP_LEN = 50  # steady-state hop (2s audio per emitted chunk)
_FAC_PRE_LOOKAHEAD_LEN = 3  # flow pre-lookahead held (not counted as new audio)
_FAC_FLOW_SEGMENT_LEN = 200  # bounded Flow segment (8s at 25 frames/s)
# First-chunk fast-start hop: emit the initial chunk as soon as this many *new*
# frames are available (plus pre_lookahead) to lower time-to-first-audio, then
# switch to the steady-state hop. The default is smaller than the steady hop
# and can be tuned independently in yaml.
# Mirrors Qwen3-Omni's initial_codec_chunk_frames vs codec_chunk_frames split.
_FAC_INITIAL_TOKEN_HOP_LEN = 10

# yaml-settable keys (top-level connectors.<name>.extra in the deploy yaml;
# read from transfer_manager.connector.config). Same key names as CosyVoice3's
# talker2code2wav_async_chunk for consistency.
_FAC_CFG_KEY_STEADY_HOP = "codec_chunk_frames"
_FAC_CFG_KEY_PRE_LOOKAHEAD = "codec_pre_lookahead_frames"
_FAC_CFG_KEY_INITIAL_HOP = "initial_codec_chunk_frames"
_FAC_CFG_KEY_FLOW_SEGMENT = "codec_flow_segment_frames"

_FAC_ASYNC_STATE_KEY = "_fac_async_state"


def _resolve_chunk_cfg(transfer_manager: Any) -> tuple[int, int, int, int]:
    """Resolve hop/lookahead/bounded-Flow sizes for streaming chunks.

    Reads the connector extra dict (set via the deploy yaml's top-level
    ``connectors:`` section; plumbed through ``stage_connector_config`` ->
    ``ChunkTransferAdapter`` -> ``transfer_manager.connector.config``), falling
    back to the ``_FAC_*`` defaults when unset. Returns ints >= 1 / >= 0.
    """
    connector = getattr(transfer_manager, "connector", None)
    cfg = getattr(connector, "config", None)
    if not isinstance(cfg, dict) or not cfg:
        return (
            _FAC_TOKEN_HOP_LEN,
            _FAC_INITIAL_TOKEN_HOP_LEN,
            _FAC_PRE_LOOKAHEAD_LEN,
            _FAC_FLOW_SEGMENT_LEN,
        )

    raw_extra = cfg.get("extra", cfg) if isinstance(cfg, dict) else cfg
    if not isinstance(raw_extra, dict):
        return (
            _FAC_TOKEN_HOP_LEN,
            _FAC_INITIAL_TOKEN_HOP_LEN,
            _FAC_PRE_LOOKAHEAD_LEN,
            _FAC_FLOW_SEGMENT_LEN,
        )

    def _get_int(key: str, default: int, minimum: int) -> int:
        try:
            value = int(raw_extra.get(key, default))
        except (TypeError, ValueError):
            value = default
        return max(minimum, value)

    steady_hop = _get_int(_FAC_CFG_KEY_STEADY_HOP, _FAC_TOKEN_HOP_LEN, 1)
    initial_hop = _get_int(_FAC_CFG_KEY_INITIAL_HOP, _FAC_INITIAL_TOKEN_HOP_LEN, 1)
    pre_lookahead = _get_int(_FAC_CFG_KEY_PRE_LOOKAHEAD, _FAC_PRE_LOOKAHEAD_LEN, 0)
    flow_segment = _get_int(_FAC_CFG_KEY_FLOW_SEGMENT, _FAC_FLOW_SEGMENT_LEN, 1)
    return steady_hop, initial_hop, pre_lookahead, flow_segment


def _to_flat_audio_token_ids(audio_token_ids: Any) -> torch.Tensor:
    if not isinstance(audio_token_ids, torch.Tensor):
        audio_token_ids = torch.as_tensor(audio_token_ids, dtype=torch.long)
    audio_token_ids = audio_token_ids.to(dtype=torch.long)
    if audio_token_ids.ndim == 2:
        # Token id 0 is valid for code2wav. Only drop rows that are fully negative
        # placeholders, and preserve all-zero codec groups from stage-0.
        valid_rows = (audio_token_ids >= 0).any(dim=-1)
        audio_token_ids = audio_token_ids[valid_rows]
    return audio_token_ids.reshape(-1)


def funaudiochat2code2wav(
    source_outputs: list[Any],
    prompt: Any = None,
    requires_multimodal_data: bool = False,
) -> list[OmniTokensPrompt]:
    """Convert FunAudioChat stage-0 audio codec output into code2wav prompts.

    The orchestrator hands us the upstream stage's per-request outputs as
    ``source_outputs`` (a list of EngineCoreOutput-like wrappers). Each entry
    exposes ``.outputs[0].multimodal_output``, where stage-0 deposited the
    ``audio_token_ids`` (and legacy ``speech_ids``) codec tokens. We flatten,
    filter to the CosyVoice3 codec id range, and emit one ``OmniTokensPrompt``
    per request for the stage-1 code2wav engine.
    """
    del prompt, requires_multimodal_data

    if not source_outputs:
        raise ValueError("source_outputs cannot be empty for funaudiochat2code2wav")

    code2wav_inputs: list[OmniTokensPrompt] = []
    for output_wrapper in source_outputs:
        outputs = getattr(output_wrapper, "outputs", None)
        if not outputs:
            raise RuntimeError("FunAudioChat stage-0 output has no per-request outputs yet")
        output = outputs[0]
        mm_output = getattr(output, "multimodal_output", None) or {}
        audio_token_ids = mm_output.get("audio_token_ids")
        if audio_token_ids is None:
            audio_token_ids = mm_output.get("speech_ids")
        if audio_token_ids is None:
            # No speech codec emitted by stage-0 (e.g. text-only / ASR tasks).
            # Skip this request — returning no inputs lets the orchestrator's
            # ``if not next_inputs`` path finalize the request without invoking
            # the code2wav stage on an empty prompt (which would crash the
            # generation scheduler with a negative token budget).
            logger.debug(
                "FunAudioChat stage0->stage1: no audio_token_ids for request; skipping code2wav (text-only output)."
            )
            continue
        flat_audio_token_ids = _to_flat_audio_token_ids(audio_token_ids)
        filtered = flat_audio_token_ids[
            (flat_audio_token_ids >= 0) & (flat_audio_token_ids < _MAX_COSYVOICE_TOKEN_ID)
        ]
        raw_min = int(flat_audio_token_ids.min().item()) if flat_audio_token_ids.numel() > 0 else None
        raw_max = int(flat_audio_token_ids.max().item()) if flat_audio_token_ids.numel() > 0 else None
        logger.debug(
            "FunAudioChat stage0->stage1 audio tokens: raw_len=%d filtered_len=%d raw_min=%s raw_max=%s tail=%s",
            flat_audio_token_ids.numel(),
            filtered.numel(),
            raw_min,
            raw_max,
            flat_audio_token_ids[-8:].tolist() if flat_audio_token_ids.numel() > 0 else [],
        )
        if filtered.numel() == 0:
            # Audio tower produced only placeholder/padding tokens (no usable
            # codec ids). Skip rather than feeding an empty prompt to code2wav.
            logger.debug(
                "FunAudioChat stage0->stage1: filtered audio tokens empty; skipping code2wav."
            )
            continue
        code2wav_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=filtered.to(dtype=torch.long).reshape(-1).tolist(),
                multi_modal_data=None,
                mm_processor_kwargs=None,
            )
        )
    return code2wav_inputs


def funaudiochat2code2wav_async_chunk(
    transfer_manager: Any,
    multimodal_output: dict[str, Any] | None,
    request: Any,
    is_finished: bool = False,
) -> OmniPayloadStruct | None:
    """FunAudioChat async_chunk producer: stage-0 CRQ codec increment -> code2wav chunks.

    Runs on stage-0's connector save thread once per decode step. Stage-0 deposits the
    per-step CRQ codec increment (shape ``(1, group_size)``, long; ``-1`` placeholder on
    non-speech steps) under ``multimodal_output["audio_token_ids"]`` (routed there by
    ``pooler_output_buffer_keys = ("audio_token_ids",)``). Unlike CosyVoice3 (which reads
    the autoregressive ``request.output_token_ids`` stream), FunAudioChat's codec ids live
    in the sidecar buffer, so we accumulate the per-step *increment* here rather than
    snapshotting a cumulative token stream.

    Accumulates valid codec ids into ``transfer_manager.code_prompt_token_ids[request_id]``
    and emits an :class:`OmniPayloadStruct` chunk whenever ``emitted + current_hop +
    pre_lookahead`` codes are available (or the remaining tail on finish), mirroring
    cosyvoice3's ``talker2code2wav_async_chunk`` state machine but without prompt
    conditioning (FunAudioChat uses a fixed default speaker embedding, so no ``embed``
    struct is shipped).

    Chunk hops are two-tier: the first chunk fires at the small ``initial`` hop for low
    time-to-first-audio, then the ``steady`` hop takes over. The four knobs
    (``codec_chunk_frames`` / ``initial_codec_chunk_frames`` /
    ``codec_pre_lookahead_frames`` / ``codec_flow_segment_frames``, in CRQ
    frames = 25 frames/s) are yaml-tunable via the deploy yaml's connector
    config, falling back to the ``_FAC_*`` defaults when unset.

    Returns ``None`` when no chunk is ready yet and the request is not finished.
    """
    request_id = getattr(request, "external_req_id", None) or getattr(request, "request_id", None)
    if request_id is None:
        return None
    request_id = str(request_id)
    finished = bool(is_finished or (callable(getattr(request, "is_finished", None)) and request.is_finished()))

    # --- per-request accumulator + state machine ------------------------------------
    request_payload = getattr(transfer_manager, "request_payload", None)
    if request_payload is None:
        # OmnicChunkTransferAdapter initializes request_payload lazily; mirror the
        # cosyvoice3 producer's defensive default so a plain dict stub works in tests.
        request_payload = defaultdict(dict)
        transfer_manager.request_payload = request_payload
    if not isinstance(request_payload, dict) and not isinstance(request_payload, defaultdict):
        request_payload = defaultdict(dict)
        transfer_manager.request_payload = request_payload

    # --- resolve chunk hops (yaml-tunable) for this request, once --------------------
    req_state_entry = request_payload.get(request_id)
    if not isinstance(req_state_entry, dict) or _FAC_ASYNC_STATE_KEY not in req_state_entry:
        steady_hop, initial_hop, pre_lookahead_len, flow_segment_len = _resolve_chunk_cfg(
            transfer_manager
        )
        req_state_entry = req_state_entry if isinstance(req_state_entry, dict) else {}
        req_state_entry[_FAC_ASYNC_STATE_KEY] = {
            "emitted_token_len": 0,
            # Absolute index represented by token_frames[0]. Old complete Flow
            # segments are discarded as soon as no later chunk can reference
            # them, keeping the producer-side buffer bounded as well.
            "buffer_start_token": 0,
            # Active hop; starts at the small fast-start hop so the first chunk
            # ships ASAP, then flips to the steady-state hop once emitted.
            "token_hop_len": initial_hop,
            "steady_hop_len": steady_hop,
            "pre_lookahead_len": pre_lookahead_len,
            "flow_segment_len": flow_segment_len,
            "is_first_chunk": True,
            "terminal_sent": False,
        }
        logger.debug(
            "FAC async config: req=%s steady_hop=%d initial_hop=%d pre_lookahead=%d "
            "flow_mode=bounded segment_tokens=%d",
            request_id,
            steady_hop,
            initial_hop,
            pre_lookahead_len,
            flow_segment_len,
        )
    request_payload[request_id] = req_state_entry
    state = req_state_entry[_FAC_ASYNC_STATE_KEY]

    if bool(state.get("terminal_sent", False)):
        return None

    # --- fold this step's codec increment into the per-request token frames ---------
    if not hasattr(transfer_manager, "code_prompt_token_ids"):
        transfer_manager.code_prompt_token_ids = defaultdict(list)
    token_frames = transfer_manager.code_prompt_token_ids[request_id]
    if isinstance(multimodal_output, dict):
        audio_token_ids = multimodal_output.get("audio_token_ids")
        if audio_token_ids is not None:
            flat = _to_flat_audio_token_ids(audio_token_ids)
            valid = flat[(flat >= 0) & (flat < _MAX_COSYVOICE_TOKEN_ID)]
            if valid.numel() > 0:
                token_frames.extend(int(t) for t in valid.reshape(-1).tolist())

    emitted_token_len = int(state.get("emitted_token_len", 0))
    buffer_start_token = int(state.get("buffer_start_token", 0))
    steady_hop_len = int(state.get("steady_hop_len", _FAC_TOKEN_HOP_LEN))
    token_hop_len = int(state.get("token_hop_len", steady_hop_len))
    pre_lookahead_len = int(state.get("pre_lookahead_len", _FAC_PRE_LOOKAHEAD_LEN))
    # All positions from here on are absolute request-token positions. The
    # actual list may start later because completed Flow segments are pruned.
    length = buffer_start_token + len(token_frames)

    # --- terminal sentinel: request finished with nothing left to flush -------------
    if finished and length <= emitted_token_len:
        state["terminal_sent"] = True
        transfer_manager.code_prompt_token_ids.pop(request_id, None)
        return OmniPayloadStruct(
            codes=CodesStruct(audio=torch.empty(0, dtype=torch.long)),
            meta=MetaStruct(
                finished=torch.tensor(True, dtype=torch.bool),
                stream_finished=torch.tensor(True, dtype=torch.bool),
                req_id=[request_id],
                left_context_size=0,
                num_processed_tokens=emitted_token_len,
            ),
        )

    required = token_hop_len + pre_lookahead_len
    if not finished:
        # Not enough new codec ids to fill a hop + lookahead yet.
        if length - emitted_token_len < required:
            return None
        prefix_len = emitted_token_len + required
    else:
        # Final flush: ship every remaining code in one chunk.
        if length <= emitted_token_len:
            return None
        prefix_len = length

    # Use the official-style tumbling segment: the segment start remains fixed
    # while emitted tokens advance inside it, then jumps by one complete segment.
    # Unlike a per-hop sliding window, overlapping Flow conditions do not change
    # on every call. Work is bounded by segment + hop + lookahead.
    flow_segment_len = int(state.get("flow_segment_len", _FAC_FLOW_SEGMENT_LEN))
    segment_start = (emitted_token_len // flow_segment_len) * flow_segment_len
    local_segment_start = segment_start - buffer_start_token
    local_prefix_len = prefix_len - buffer_start_token
    if local_segment_start < 0:
        raise RuntimeError(
            "FunAudioChat codec buffer was pruned past the required Flow segment: "
            f"req={request_id} buffer_start={buffer_start_token} "
            f"segment_start={segment_start}"
        )
    code_predictor_codes = [
        int(frame) for frame in token_frames[local_segment_start:local_prefix_len]
    ]
    segment_offset = emitted_token_len - segment_start
    payload = OmniPayloadStruct(
        codes=CodesStruct(audio=torch.tensor(code_predictor_codes, dtype=torch.long)),
        meta=MetaStruct(
            finished=torch.tensor(finished, dtype=torch.bool),
            stream_finished=torch.tensor(finished, dtype=torch.bool),
            req_id=[request_id],
            left_context_size=segment_offset,
            num_processed_tokens=segment_start,
        ),
    )
    if not finished:
        next_emitted_token_len = emitted_token_len + token_hop_len
        state["emitted_token_len"] = next_emitted_token_len
        # After the fast-start first chunk, switch to the steady-state hop.
        # Unconditional (config-safe): compares the per-request first-chunk flag,
        # not a constant, so an initial_hop != steady_hop survives yaml tuning.
        if bool(state.get("is_first_chunk", True)):
            state["token_hop_len"] = steady_hop_len
            state["is_first_chunk"] = False

        # Once the next emitted position belongs to a later tumbling segment,
        # no subsequent Flow call can reference the completed prefix. Prune it
        # now so a long request does not leave an O(total_tokens) Python list.
        next_segment_start = (
            next_emitted_token_len // flow_segment_len
        ) * flow_segment_len
        prune_count = next_segment_start - buffer_start_token
        if prune_count > 0:
            del token_frames[:prune_count]
            state["buffer_start_token"] = next_segment_start
    else:
        state["terminal_sent"] = True
        state["emitted_token_len"] = prefix_len
        state["buffer_start_token"] = prefix_len
        transfer_manager.code_prompt_token_ids.pop(request_id, None)
    return payload
