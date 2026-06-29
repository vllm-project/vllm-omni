# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2025 The Qwen team.
"""Stage input processor for Qwen3 Omni MoE: Thinker → Talker transition."""

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import torch
from vllm.inputs import TextPrompt
from vllm.platforms import current_platform

from vllm_omni.data_entry_keys import (
    CodesStruct,
    EmbeddingsStruct,
    HiddenStatesStruct,
    IdsStruct,
    MetaStruct,
    OmniPayload,
    OmniPayloadStruct,
    to_dict,
)
from vllm_omni.engine import OmniEngineCoreRequest
from vllm_omni.inputs.data import OmniTokensPrompt
from vllm_omni.model_executor.stage_input_processors.tts_utils import (
    extract_language_from_prompt,
    extract_language_from_request,
    extract_speaker_from_prompt,
    extract_speaker_from_request,
)

logger = logging.getLogger(__name__)

# Pooling output layer keys: "0" = word embedding, "24" = accept_hidden_layer
_EMBED_LAYER_KEY = "0"
_HIDDEN_LAYER_KEY = "24"
# Per-model REPLACE-keys for the full-payload accumulator.  Keys in this
# set use REPLACE semantics (subsequent emissions discard prior chunks)
# instead of CONCAT.  qwen3-omni currently has none — model_outputs is
# not emitted by the thinker/talker forward.
_FULL_PAYLOAD_REPLACE_KEYS: frozenset[str] = frozenset()

_QWEN3_CODEC_CODEBOOK_SIZE = 2048
_QWEN3_CODEC_PAD_TOKEN_ID = 4196
_QWEN3_CODEC_BOS_TOKEN_ID = 4197
_QWEN3_CODEC_EOS_TOKEN_ID = 4198

# ChatML special token IDs
_IM_START_TOKEN_ID = 151644
_SYSTEM_TOKEN_ID = 8948


def _get_async_chunk_pending_prefill_boot(
    transfer_manager: Any,
) -> dict[str, dict[str, Any]]:
    state = getattr(transfer_manager, "_pending_prefill_boot", None)
    if state is None:
        state = {}
        setattr(transfer_manager, "_pending_prefill_boot", state)
    return state


def _get_prefill_part_state(
    transfer_manager: Any,
) -> dict[str, dict[str, Any]]:
    state = getattr(transfer_manager, "_prefill_part_state", None)
    if state is None:
        state = {}
        setattr(transfer_manager, "_prefill_part_state", state)
    return state


def _is_prompt_prefill_complete(sent_prompt_tokens: int, prompt_len: int) -> bool:
    return sent_prompt_tokens >= prompt_len


def _is_final_prefill_embed_chunk(chunk_start: int, embed_rows: int, prompt_len: int) -> bool:
    """True when this chunk's embedding rows cover the remaining prompt positions."""
    remaining = max(0, prompt_len - chunk_start)
    return remaining > 0 and embed_rows >= remaining


def _release_pending_prefill_boot(
    transfer_manager: Any,
    external_req_id: str,
    decode_embed: torch.Tensor,
    decode_meta: dict[str, Any],
) -> dict[str, Any] | None:
    pending_prefill_boot = _get_async_chunk_pending_prefill_boot(transfer_manager)
    pending_payload = pending_prefill_boot.pop(external_req_id, None)
    if pending_payload is None:
        return None

    payload = dict(pending_payload)
    embed = payload.get("embed", {})
    embed = dict(embed) if isinstance(embed, dict) else {}
    embed["decode"] = decode_embed
    payload["embed"] = embed

    meta = payload.get("meta", {})
    meta = dict(meta) if isinstance(meta, dict) else {}
    if isinstance(decode_meta, dict):
        meta.update({key: decode_meta[key] for key in ("finished", "thinker_finished") if key in decode_meta})
    payload["meta"] = meta

    transfer_manager.request_payload[external_req_id] = payload
    return payload


def _gate_chunked_prefill_chunk(
    transfer_manager: Any,
    request: Any,
    payload_data: dict[str, Any],
    external_req_id: str,
) -> bool:
    if payload_data.get("meta", {}).get("is_final_prefill_chunk", False):
        pending_payload = transfer_manager.request_payload.get(external_req_id, payload_data)
        pending_prefill_boot = _get_async_chunk_pending_prefill_boot(transfer_manager)
        pending_prefill_boot[external_req_id] = {
            key: (dict(value) if isinstance(value, dict) else value) for key, value in pending_payload.items()
        }
        transfer_manager._pending_load_reqs.append(request)
        with transfer_manager._recv_cond:
            transfer_manager._recv_cond.notify()
        return True

    accumulated = transfer_manager.request_payload.get(external_req_id, payload_data)
    cumulative_embeds = accumulated.get("embed", {}).get("prefill")
    available_tokens = cumulative_embeds.shape[0] if isinstance(cumulative_embeds, torch.Tensor) else 0
    remaining_prompt_tokens = max(request.num_prompt_tokens - request.num_computed_tokens, 0)
    next_scheduler_slice = min(
        transfer_manager.scheduler_max_num_batched_tokens,
        remaining_prompt_tokens,
    )
    ready_tokens = available_tokens - request.num_computed_tokens
    if ready_tokens >= next_scheduler_slice:
        return False

    transfer_manager._pending_load_reqs.append(request)
    with transfer_manager._recv_cond:
        transfer_manager._recv_cond.notify()
    return True


def _evict_prefill_tensors(
    transfer_manager: Any,
    external_req_id: str,
) -> None:
    acc = transfer_manager.request_payload.get(external_req_id)
    if acc is None or not isinstance(acc.get("embed", {}).get("prefill"), torch.Tensor):
        return
    cleaned_embed = {
        k: v for k, v in acc.get("embed", {}).items() if k not in ("prefill", "decode", "tts_bos", "tts_eos", "tts_pad")
    }
    cleaned_hs = {k: v for k, v in acc.get("hidden_states", {}).items() if k != "output"}
    cleaned_ids = {k: v for k, v in acc.get("ids", {}).items() if k not in ("all", "prompt")}
    cleaned = {k: v for k, v in acc.items() if k not in ("embed", "hidden_states", "ids")}
    cleaned["embed"] = cleaned_embed
    cleaned["hidden_states"] = cleaned_hs
    cleaned["ids"] = cleaned_ids
    transfer_manager.request_payload[external_req_id] = cleaned


def _layer_tensor(layers: dict[Any, Any], key: str) -> torch.Tensor | None:
    """Fetch layer tensor with tolerant key lookup (str/int)."""
    if not isinstance(layers, dict):
        return None
    key_int = int(key)
    val = layers.get(key_int)
    if val is None:
        val = layers.get(key)
    return val if isinstance(val, torch.Tensor) else None


@dataclass
class _ThinkerPoolingOutput:
    thinker_emb: torch.Tensor
    thinker_hid: torch.Tensor
    thinker_tts: dict[str, Any]
    speaker: Any
    language: Any


def _parse_thinker2talker_pooling_output(
    multimodal_output: OmniPayload | dict[str, Any],
    request: OmniEngineCoreRequest,
    *,
    log_prefix: str = "thinker2talker_async_chunk",
) -> _ThinkerPoolingOutput | None:
    """Extract thinker pooling layers and request metadata for thinker→talker handoff."""
    request_id = request.external_req_id
    if not isinstance(multimodal_output, Mapping):
        logger.debug("%s: skip non-dict multimodal_output for req=%s", log_prefix, request_id)
        return None

    thinker_hs = multimodal_output.get("hidden_states", {})
    thinker_layers = thinker_hs.get("layers", {}) if isinstance(thinker_hs, dict) else {}
    thinker_tts_raw = multimodal_output.get("embed", {})
    thinker_tts = thinker_tts_raw if isinstance(thinker_tts_raw, dict) else {}
    thinker_emb = _layer_tensor(thinker_layers, _EMBED_LAYER_KEY)
    thinker_hid = _layer_tensor(thinker_layers, _HIDDEN_LAYER_KEY)
    if thinker_emb is None or thinker_hid is None:
        logger.debug(
            "%s: missing thinker layers for req=%s (embed=%s hidden=%s)",
            log_prefix,
            request_id,
            thinker_emb is not None,
            thinker_hid is not None,
        )
        return None

    return _ThinkerPoolingOutput(
        thinker_emb=thinker_emb,
        thinker_hid=thinker_hid,
        thinker_tts=thinker_tts,
        speaker=extract_speaker_from_request(request),
        language=extract_language_from_request(request),
    )


def _compute_talker_prompt_ids_length(info: OmniPayload, device: torch.device | str = "cuda") -> int:
    im_start_token_id = 151644
    system_token_id = 8948
    user_token_id = 872
    assistant_token_id = 77091

    ids = info.get("ids", {})
    thinker_sequences = torch.tensor(ids["all"], dtype=torch.long, device=device).unsqueeze(0)  # [1, T]

    input_ids = torch.tensor(ids["prompt"], dtype=torch.long, device=device).unsqueeze(0)  # [1, T]

    im_start_indexes = torch.cat(
        [
            torch.nonzero(input_ids[0] == im_start_token_id).squeeze(1),
            torch.tensor([thinker_sequences.shape[-1]], device=input_ids.device, dtype=input_ids.dtype),
        ],
        dim=0,
    )

    sum_user_len = 0
    assistant_len = 0
    for i in range(len(im_start_indexes) - 1):
        s = int(im_start_indexes[i].item())
        e = int(im_start_indexes[i + 1].item())
        role = int(input_ids[0, s + 1].item())
        if role == system_token_id:
            continue
        elif role == user_token_id:
            sum_user_len += e - s
        elif role == assistant_token_id and i == len(im_start_indexes) - 2:
            assistant_len += 9  # 3 + 4 + 1 + 1
        else:
            pass

    return sum_user_len + assistant_len


# =========================
# Common helpers
# =========================


def _ensure_list(x):
    """Convert ConstantList / tensor-like to Python list."""
    if hasattr(x, "_x"):
        return list(x._x)
    elif not isinstance(x, list):
        return x
    return list(x)


def _as_tensor_or_none(value: Any) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, list) and value and isinstance(value[0], torch.Tensor):
        return value[0].detach().cpu()
    return None


def _is_valid_qwen3_codec_token_id(token_id: Any) -> bool:
    try:
        token_id = int(token_id)
    except (TypeError, ValueError):
        return False
    return 0 <= token_id < _QWEN3_CODEC_CODEBOOK_SIZE


def _extract_qwen3_full_payload_codec_rows(
    code_predictor_codes: torch.Tensor,
    output_token_ids: list[int],
) -> tuple[torch.Tensor, dict[str, int]]:
    """Filter full-payload codec rows by the authoritative output ids."""
    if code_predictor_codes.ndim != 2 or code_predictor_codes.numel() == 0:
        return code_predictor_codes, {
            "raw_rows": int(code_predictor_codes.shape[0]) if code_predictor_codes.ndim > 0 else 0,
            "aligned_rows": 0,
            "valid_rows": 0,
            "trailing_placeholder_count": 0,
        }

    trailing_placeholder_count = 0
    while (
        trailing_placeholder_count < len(output_token_ids) and output_token_ids[-1 - trailing_placeholder_count] == -1
    ):
        trailing_placeholder_count += 1

    aligned_len = min(int(code_predictor_codes.shape[0]), len(output_token_ids))
    if aligned_len <= 0:
        return code_predictor_codes[:0], {
            "raw_rows": int(code_predictor_codes.shape[0]),
            "aligned_rows": 0,
            "valid_rows": 0,
            "trailing_placeholder_count": trailing_placeholder_count,
        }

    aligned_rows = code_predictor_codes[-aligned_len:]
    aligned_token_ids = output_token_ids[-aligned_len:]
    aligned_token_mask = torch.tensor(
        [_is_valid_qwen3_codec_token_id(token_id) for token_id in aligned_token_ids],
        dtype=torch.bool,
        device=aligned_rows.device,
    )
    row_valid_mask = (aligned_rows.max(dim=1).values < _QWEN3_CODEC_CODEBOOK_SIZE) & (
        aligned_rows.min(dim=1).values >= 0
    )
    filtered_rows = aligned_rows[aligned_token_mask & row_valid_mask]
    if filtered_rows.numel() == 0:
        filtered_rows = aligned_rows[:0]
    return filtered_rows, {
        "raw_rows": int(code_predictor_codes.shape[0]),
        "aligned_rows": aligned_len,
        "valid_rows": int(filtered_rows.shape[0]) if filtered_rows.ndim > 0 else 0,
        "trailing_placeholder_count": trailing_placeholder_count,
    }


# =========================
# PD disaggregation helpers
# =========================


def _get_prefill_multimodal_output(
    request_id: str,
    streaming_context: Any | None,
) -> dict[str, Any] | None:
    bridge_states = getattr(streaming_context, "bridge_states", None)
    if not isinstance(bridge_states, dict):
        return None
    by_req = bridge_states.get("pd_prefill_multimodal_output_by_req")
    if not isinstance(by_req, dict):
        return None
    prefill_mm = by_req.get(request_id)
    return prefill_mm if isinstance(prefill_mm, Mapping) else None


def _merge_pd_embeddings(
    decode_emb: torch.Tensor,
    decode_hid: torch.Tensor,
    prefill_mm: dict[str, Any],
    device: torch.device,
    expected_total: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge prefill prompt embeddings with decode generated embeddings.

    In PD mode the prefill engine processes the prompt and the decode engine
    generates tokens starting from position 1.  This function concatenates
    them, removing the overlapping token(s):

        merged = prefill[:P] + decode[overlap:]

    where overlap = P + D - expected_total.
    """
    try:
        p_layers = prefill_mm.get("hidden_states", {}).get("layers", {})
        p_emb = p_layers[int(_EMBED_LAYER_KEY)].detach().to(device=device, dtype=torch.float)
        p_hid = p_layers[int(_HIDDEN_LAYER_KEY)].detach().to(device=device, dtype=torch.float)
    except (KeyError, AttributeError, TypeError) as exc:
        available_keys = list(prefill_mm.keys()) if isinstance(prefill_mm, Mapping) else type(prefill_mm).__name__
        logger.error(
            "_merge_pd_embeddings: failed to extract prefill embeddings (%s). "
            "Expected keys %r and %r, got: %s. "
            "Falling back to decode-only embeddings – talker user-segment will be degraded.",
            exc,
            _EMBED_LAYER_KEY,
            _HIDDEN_LAYER_KEY,
            available_keys,
        )
        return decode_emb, decode_hid

    if p_emb.shape[0] == 0 or decode_emb.shape[0] == 0:
        return decode_emb, decode_hid

    raw_total = p_emb.shape[0] + decode_emb.shape[0]
    overlap = max(0, raw_total - expected_total) if expected_total is not None else 0

    merged_emb = torch.cat([p_emb, decode_emb[overlap:]], dim=0)
    merged_hid = torch.cat([p_hid, decode_hid[overlap:]], dim=0)
    return merged_emb, merged_hid


def _resolve_tts_token_embedding(
    key: str,
    *,
    thinker_mm: dict[str, Any],
    prefill_mm: dict[str, Any] | None,
    device: torch.device,
) -> torch.Tensor | None:
    """Return TTS BOS/EOS/PAD embedding tensors for the talker projection path.

    Values are taken from the current thinker (decode) ``multimodal_output``; in
    PD mode, missing keys may be filled from the paired prefill stage output.
    """
    val = thinker_mm.get("embed", {}).get(key)
    if val is None and prefill_mm is not None:
        val = prefill_mm.get("embed", {}).get(key)
    return val.detach().to(device=device, dtype=torch.float) if val is not None else None


# =========================
# Streaming input helpers
# =========================


def _construct_thinker2talker_streaming_input_async_chunk(
    is_finished: bool,
    request,
    thinker_emb,
    thinker_hid,
    transfer_manager,
) -> OmniPayloadStruct | None:
    """Build Thinker -> Talker payloads for realtime streaming input chunks.

    A resumable realtime request reuses the same logical request id across
    audio segments. The first streaming prefill chunk is cached and returns ``None`` so the
    connector does not emit an incomplete downstream chunk. The following
    decode chunk flushes that cached prefill together with the current Thinker
    output, keeping Talker ids and tensor rows aligned.
    """
    request_id = request.external_req_id
    output_token_ids = request.output_token_ids
    # Convert ConstantList to regular list for OmniSerializer serialization
    output_token_ids = _ensure_list(output_token_ids)
    speaker = extract_speaker_from_request(request)
    language = extract_language_from_request(request)
    finished = torch.tensor(is_finished, dtype=torch.bool)
    emb_cpu = thinker_emb.detach().cpu()
    hid_cpu = thinker_hid.detach().cpu()

    if output_token_ids:
        if thinker_emb.shape[0] > 1:
            # if thinker_emb.shape[0] > 1, new streaming input segment is added
            # and will transfer prefill embeddings and hidden states to talker.
            new_prompt_len = thinker_emb.shape[0]
            payload = OmniPayloadStruct(
                meta=MetaStruct(finished=finished),
                embed=EmbeddingsStruct(prefill=emb_cpu),
                hidden_states=HiddenStatesStruct(output=hid_cpu),
                ids=IdsStruct(
                    all=_ensure_list(request.all_token_ids[-new_prompt_len - 1 :]),
                    prompt=_ensure_list(request.prompt_token_ids[-new_prompt_len:]),
                ),
                speaker=speaker,
                language=language,
            )
            transfer_manager._pending_streaming_prefills[request_id] = to_dict(payload)
            return None
        else:
            save_payload = transfer_manager._pending_streaming_prefills.pop(request_id, None)
            if save_payload is not None:
                saved_prefill = save_payload.get("embed", {}).get("prefill")
                saved_output = save_payload.get("hidden_states", {}).get("output")
                if isinstance(saved_prefill, torch.Tensor) and isinstance(saved_output, torch.Tensor):
                    return OmniPayloadStruct(
                        meta=MetaStruct(finished=finished),
                        embed=EmbeddingsStruct(prefill=torch.cat((saved_prefill, emb_cpu), dim=0)),
                        hidden_states=HiddenStatesStruct(output=torch.cat((saved_output, hid_cpu), dim=0)),
                        ids=IdsStruct(
                            all=save_payload.get("ids", {}).get("all"),
                            prompt=save_payload.get("ids", {}).get("prompt"),
                        ),
                        speaker=speaker,
                        language=language,
                    )
            return OmniPayloadStruct(
                meta=MetaStruct(
                    finished=finished,
                ),
                embed=EmbeddingsStruct(decode=emb_cpu),
                hidden_states=HiddenStatesStruct(output=hid_cpu),
                speaker=speaker,
                language=language,
            )
    else:
        if not is_finished:
            # do not send async chunk mode placeholder token or embedding/hidden of the stop token
            return None
        return OmniPayloadStruct(
            meta=MetaStruct(finished=finished),
            embed=EmbeddingsStruct(decode=emb_cpu),
            hidden_states=HiddenStatesStruct(output=hid_cpu),
            speaker=speaker,
            language=language,
        )


@dataclass
class _Thinker2TalkerStreamingState:
    last_prompt_len: int = 0
    last_output_len: int = 0
    merged_sequences: list[int] = field(default_factory=list)


@dataclass
class _Qwen3OmniStreamingState:
    thinker2talker: _Thinker2TalkerStreamingState = field(default_factory=_Thinker2TalkerStreamingState)
    talker2code2wav_last_seq_len: int = 0


def _get_qwen3_streaming_state(
    request_id: str,
    streaming_context: Any | None,
) -> _Qwen3OmniStreamingState:
    bridge_states = getattr(streaming_context, "bridge_states", None)
    per_model_state = bridge_states.setdefault("qwen3_omni", {})
    state = per_model_state.get(request_id)
    if state is None:
        state = _Qwen3OmniStreamingState()
        per_model_state[request_id] = state
    return state


def _get_streaming_talker_tokens(
    request_id: str,
    prompt_token_ids: list[int],
    output_token_ids: list[int],
    new_prompt_len_snapshot: int | None = None,
    streaming_context: Any | None = None,
    *,
    clear_state: bool = False,
) -> tuple[list[int], list[int]]:
    """Return prompt/output token deltas for the current streaming segment.

    In non-async-chunk streaming, Thinker's prompt may already include the
    next input segment. Remove that new prompt tail before building the Talker
    delta for the previous segment.

    Returns:
        inc_prompt: prompt token delta for this segment.
        inc_output: output token delta for this segment.
    """
    state = _get_qwen3_streaming_state(request_id, streaming_context).thinker2talker
    if new_prompt_len_snapshot:
        prompt_token_ids = prompt_token_ids[:-new_prompt_len_snapshot]
    cur_prompt_len = len(prompt_token_ids)
    cur_output_len = len(output_token_ids)

    inc_prompt = prompt_token_ids[state.last_prompt_len :]
    inc_output = output_token_ids[state.last_output_len :]

    state.last_prompt_len = cur_prompt_len
    state.last_output_len = cur_output_len

    if clear_state:
        state.last_prompt_len = 0
        state.last_output_len = 0
        state.merged_sequences.clear()

    return inc_prompt, inc_output


def _get_streaming_codec_delta_len(
    cur_seq_len: int,
    request_id: str,
    talker_output: Any,
    streaming_context: Any | None = None,
) -> int:
    """Return newly added seq_len for talker->code2wav in streaming mode."""
    state = _get_qwen3_streaming_state(request_id, streaming_context)
    prev_seq_len = state.talker2code2wav_last_seq_len
    seq_len = cur_seq_len - prev_seq_len
    state.talker2code2wav_last_seq_len = cur_seq_len + 1
    if bool(getattr(talker_output, "finished", False)):
        # Final segment: clear history to avoid cross-session carry-over.
        state.talker2code2wav_last_seq_len = 0
    return seq_len


# =========================
# Thinker -> Talker
# =========================


def thinker2talker_async_chunk(
    transfer_manager: Any,
    multimodal_output: OmniPayload | dict[str, Any],
    request: OmniEngineCoreRequest,
    is_finished: bool = False,
) -> OmniPayloadStruct | None:
    """
    Process thinker outputs to create talker inputs.
    1. thinker's text generation outputs (token IDs + hidden states)
    2. Split hidden states into: prompt embeddings + generated embeddings
    3. Package for talker with additional information
    """

    request_id = request.external_req_id
    chunk_id = transfer_manager.put_req_chunk[request_id]
    pooling = _parse_thinker2talker_pooling_output(multimodal_output, request)
    if pooling is None:
        return None
    thinker_emb = pooling.thinker_emb
    thinker_hid = pooling.thinker_hid
    thinker_tts = pooling.thinker_tts
    speaker = pooling.speaker
    language = pooling.language

    def _maybe_cpu(t: Any) -> torch.Tensor | None:
        return t.detach().cpu() if isinstance(t, torch.Tensor) else None

    if chunk_id == 0:
        all_token_ids = _ensure_list(request.all_token_ids)
        prompt_token_ids = _ensure_list(request.prompt_token_ids)
        payload = OmniPayloadStruct(
            embed=EmbeddingsStruct(
                prefill=thinker_emb.detach().cpu(),
                tts_bos=_maybe_cpu(thinker_tts.get("tts_bos")),
                tts_eos=_maybe_cpu(thinker_tts.get("tts_eos")),
                tts_pad=_maybe_cpu(thinker_tts.get("tts_pad")),
            ),
            hidden_states=HiddenStatesStruct(output=thinker_hid.detach().cpu()),
            ids=IdsStruct(all=all_token_ids, prompt=prompt_token_ids),
            meta=MetaStruct(finished=torch.tensor(is_finished, dtype=torch.bool)),
            speaker=speaker,
            language=language,
        )
        if transfer_manager.request_payload.get(request_id) is None:
            if not is_finished:
                transfer_manager.request_payload[request_id] = to_dict(payload)
                return None
        else:
            save_payload = transfer_manager.request_payload.pop(request_id)
            payload.embed.prefill = torch.cat(
                (save_payload.get("embed", {}).get("prefill"), payload.embed.prefill), dim=0
            )
            payload.hidden_states.output = torch.cat(
                (save_payload.get("hidden_states", {}).get("output"), payload.hidden_states.output), dim=0
            )
            prefill_shape = payload.embed.prefill.shape[0]
            if not is_finished and prefill_shape <= len(prompt_token_ids):
                transfer_manager.request_payload[request_id] = to_dict(payload)
                return None
    else:
        if request.resumable:
            return _construct_thinker2talker_streaming_input_async_chunk(
                is_finished, request, thinker_emb, thinker_hid, transfer_manager
            )
        if thinker_emb.shape[0] > 1:
            logger.warning(
                "Unexpected multiple embeddings in thinker2talker_async_chunk for chunk_id %d: "
                "request_id %s, num_computed_tokens%d %s. Expected shape [1, D].",
                chunk_id,
                request_id,
                request.num_computed_tokens,
                thinker_emb.shape,
            )
            return None
        meta = MetaStruct(finished=torch.tensor(is_finished, dtype=torch.bool))
        payload = OmniPayloadStruct(
            meta=meta,
            embed=EmbeddingsStruct(decode=thinker_emb.detach().cpu()),
            speaker=speaker,
            language=language,
        )
    return payload


def thinker2talker_async_chunk_chunked_prefill(
    transfer_manager: Any,
    multimodal_output: OmniPayload,
    request: OmniEngineCoreRequest,
    is_finished: bool = False,
) -> OmniPayloadStruct | None:
    request_id = request.external_req_id
    prompt_token_ids = _ensure_list(request.prompt_token_ids)
    state_map = _get_prefill_part_state(transfer_manager)
    state = state_map.setdefault(request_id, {"sent_prompt_tokens": 0})

    sent_prompt_tokens = state.get("sent_prompt_tokens", 0)
    prompt_len = len(prompt_token_ids)
    prefill_complete = _is_prompt_prefill_complete(sent_prompt_tokens, prompt_len)

    held = state.get("held_final_prefill")
    if held is not None and (prefill_complete or is_finished):
        state.pop("held_final_prefill", None)
        final_prefill_chunk = bool(prefill_complete)
        return OmniPayloadStruct(
            embed=EmbeddingsStruct(
                prefill=held["embeds"],
                tts_bos=held["tts_bos"],
                tts_eos=held["tts_eos"],
                tts_pad=held["tts_pad"],
            ),
            hidden_states=HiddenStatesStruct(output=held["hidden"]),
            ids=IdsStruct(all=held["filtered_ids"], prompt=held["filtered_ids"]),
            meta=MetaStruct(
                # When the prompt is complete, gate this final prefill chunk
                # until the first decode token arrives so pos-8 can be filled
                # from the real first text token.
                finished=torch.tensor(bool(is_finished and not final_prefill_chunk), dtype=torch.bool),
                is_final_prefill_chunk=final_prefill_chunk,
                override_keys=[("ids", "all"), ("ids", "prompt")],
            ),
            speaker=held["speaker"],
            language=held["language"],
        )

    if prefill_complete:
        return thinker2talker_async_chunk(transfer_manager, multimodal_output, request, is_finished)

    pooling = _parse_thinker2talker_pooling_output(multimodal_output, request)
    if pooling is None:
        return None
    thinker_emb = pooling.thinker_emb
    thinker_hid = pooling.thinker_hid
    thinker_tts = pooling.thinker_tts
    speaker = pooling.speaker
    language = pooling.language
    chunk_start = state["sent_prompt_tokens"]
    embeds_cpu = thinker_emb.detach().cpu()
    hidden_cpu = thinker_hid.detach().cpu()
    state["sent_prompt_tokens"] = chunk_start + embeds_cpu.shape[0]
    im_starts = [i for i, token_id in enumerate(prompt_token_ids) if token_id == _IM_START_TOKEN_ID]
    im_starts.append(len(prompt_token_ids))
    system_ranges = [
        (s, e)
        for s, e in zip(im_starts, im_starts[1:])
        if s + 1 < len(prompt_token_ids) and prompt_token_ids[s + 1] == _SYSTEM_TOKEN_ID
    ]
    if system_ranges:

        def _in_system(pos: int) -> bool:
            return any(s <= pos < e for s, e in system_ranges)

        filtered_ids = [token_id for pos, token_id in enumerate(prompt_token_ids) if not _in_system(pos)]
        keep_mask = torch.tensor(
            [not _in_system(chunk_start + i) for i in range(embeds_cpu.shape[0])], dtype=torch.bool
        )
        embeds_cpu, hidden_cpu = embeds_cpu[keep_mask], hidden_cpu[keep_mask]
    else:
        filtered_ids = prompt_token_ids
    if embeds_cpu.shape[0] == 0:
        return None
    assistant_region_start = im_starts[-2] if len(im_starts) >= 2 else len(prompt_token_ids)
    chunk_end_raw = state.get("sent_prompt_tokens", 0)
    prompt_rows_complete = chunk_end_raw >= prompt_len
    n_assistant = max(0, chunk_end_raw - max(chunk_start, assistant_region_start))
    if n_assistant > 0:
        n_assistant = min(n_assistant, embeds_cpu.shape[0])
        n_user = embeds_cpu.shape[0] - n_assistant
        held_embeds = embeds_cpu[n_user:]
        held_hidden = hidden_cpu[n_user:]
        prev = state.get("held_final_prefill")
        if prev is not None:
            held_embeds = torch.cat([prev["embeds"], held_embeds], dim=0)
            held_hidden = torch.cat([prev["hidden"], held_hidden], dim=0)
        state["held_final_prefill"] = {
            "embeds": held_embeds,
            "hidden": held_hidden,
            "filtered_ids": filtered_ids,
            "tts_bos": _as_tensor_or_none(thinker_tts.get("tts_bos")),
            "tts_eos": _as_tensor_or_none(thinker_tts.get("tts_eos")),
            "tts_pad": _as_tensor_or_none(thinker_tts.get("tts_pad")),
            "speaker": speaker,
            "language": language,
        }
        if prompt_rows_complete:
            # This chunk completed the assistant bootstrap source rows. Emit
            # the final prefill payload now and let the receiver gate it until
            # the first decode token arrives, instead of flushing it on the
            # first-token call and dropping that token.
            state.pop("held_final_prefill", None)
            if n_user > 0:
                emit_embeds = torch.cat([embeds_cpu[:n_user], held_embeds], dim=0)
                emit_hidden = torch.cat([hidden_cpu[:n_user], held_hidden], dim=0)
            else:
                emit_embeds = held_embeds
                emit_hidden = held_hidden
        else:
            if n_user <= 0:
                return None
            emit_embeds = embeds_cpu[:n_user]
            emit_hidden = hidden_cpu[:n_user]
    else:
        emit_embeds = embeds_cpu
        emit_hidden = hidden_cpu

    meta = MetaStruct(
        finished=torch.tensor(False, dtype=torch.bool),
        override_keys=[("ids", "all"), ("ids", "prompt")],
    )
    # Straddling chunk: only mark final once the assistant bootstrap source rows
    # are included in the emitted payload.
    if n_assistant <= 0 or is_finished or prompt_rows_complete:
        if prompt_rows_complete or _is_final_prefill_embed_chunk(chunk_start, embeds_cpu.shape[0], prompt_len):
            meta.is_final_prefill_chunk = True
    return OmniPayloadStruct(
        embed=EmbeddingsStruct(
            prefill=emit_embeds,
            tts_bos=_as_tensor_or_none(thinker_tts.get("tts_bos")),
            tts_eos=_as_tensor_or_none(thinker_tts.get("tts_eos")),
            tts_pad=_as_tensor_or_none(thinker_tts.get("tts_pad")),
        ),
        hidden_states=HiddenStatesStruct(output=emit_hidden),
        ids=IdsStruct(all=filtered_ids, prompt=filtered_ids),
        meta=meta,
        speaker=speaker,
        language=language,
    )


def async_chunk_handle_ar_payload(
    transfer_manager: Any,
    request: Any,
    req_id: str,
    external_req_id: str,
    payload_data: dict[str, Any],
    meta: dict[str, Any],
    payload_finished: bool,
) -> bool | None:
    embed_data = payload_data.get("embed", {})
    if not isinstance(embed_data, dict):
        embed_data = {}
    has_prefill_embeds = isinstance(embed_data.get("prefill"), torch.Tensor)
    has_decode_embed = isinstance(embed_data.get("decode"), torch.Tensor)

    if has_decode_embed:
        boot_payload = _release_pending_prefill_boot(transfer_manager, external_req_id, embed_data["decode"], meta)
        if boot_payload is not None:
            request.additional_information = boot_payload
            transfer_manager._finished_load_reqs.add(req_id)
            _evict_prefill_tensors(transfer_manager, external_req_id)
            return True

    if has_prefill_embeds:
        # accumulate prefill chunks until the decode chunk arrives, then merge and send
        merged_payload = _update_request_payload(transfer_manager, external_req_id, payload_data)
        request.additional_information = merged_payload
        prefill_boundary = payload_finished or has_decode_embed
        if not prefill_boundary:
            if _gate_chunked_prefill_chunk(transfer_manager, request, payload_data, external_req_id):
                return True
        if prefill_boundary:
            _evict_prefill_tensors(transfer_manager, external_req_id)
    else:
        # decode-only
        request.additional_information = payload_data

    return None


def async_chunk_cleanup_state(
    transfer_manager: Any,
    external_req_id: str,
) -> None:
    _get_async_chunk_pending_prefill_boot(transfer_manager).pop(external_req_id, None)
    prefill_part_state = getattr(transfer_manager, "_prefill_part_state", None)
    if isinstance(prefill_part_state, dict):
        prefill_part_state.pop(external_req_id, None)


def hook_for_chunked_prefill() -> dict[str, Any]:
    return {
        "async_chunk_handle_ar_payload_func": async_chunk_handle_ar_payload,
        "async_chunk_cleanup_state_func": async_chunk_cleanup_state,
    }


def _update_request_payload(transfer_manager: Any, req_id: str, payload_data: dict[str, Any]) -> dict[str, Any]:
    """Merge talker-side async prefill chunks: cat embedding rows, replace ids/meta flags."""
    if req_id not in transfer_manager.request_payload:
        transfer_manager.request_payload[req_id] = payload_data
        return payload_data

    meta = payload_data.get("meta")
    replace_keys = {
        tuple(k) if isinstance(k, list) else k
        for k in (meta.pop("override_keys", []) if isinstance(meta, dict) else [])
    }
    cat_keys = {("embed", "prefill"), ("hidden_states", "output")}

    origin = transfer_manager.request_payload[req_id]
    merged = dict(origin)
    for type_key, new_val in payload_data.items():
        if not isinstance(new_val, dict):
            merged[type_key] = new_val
            continue
        origin_sub = origin.get(type_key)
        if not isinstance(origin_sub, dict):
            merged[type_key] = dict(new_val)
            continue

        merged_sub = dict(origin_sub)
        for qual, value in new_val.items():
            key = (type_key, qual)
            old = origin_sub.get(qual)
            if key in replace_keys or (type_key == "meta" and qual in ("finished", "is_segment_finished")):
                merged_sub[qual] = value
            elif key in cat_keys and isinstance(value, torch.Tensor) and isinstance(old, torch.Tensor):
                merged_sub[qual] = torch.cat([old, value], dim=0)
            else:
                merged_sub[qual] = value
        merged[type_key] = merged_sub

    transfer_manager.request_payload[req_id] = merged
    return merged


def thinker2talker_full_payload(
    transfer_manager: Any,
    pooling_output: dict[str, Any],
    request: OmniEngineCoreRequest,
) -> dict[str, Any] | None:
    """Pack complete thinker output for the non-async connector path."""
    rid = getattr(request, "request_id", None)
    if not isinstance(pooling_output, Mapping):
        logger.warning(
            "thinker2talker_full_payload: pooling_output not a dict (type=%s) for req=%s; consumer wait gate may hang.",
            type(pooling_output).__name__,
            rid,
        )
        return None

    layers = {
        0: pooling_output.get("hidden_states.layer_0"),
        24: pooling_output.get("hidden_states.layer_24"),
    }
    thinker_emb = _layer_tensor(layers, _EMBED_LAYER_KEY)
    thinker_hid = _layer_tensor(layers, _HIDDEN_LAYER_KEY)
    if thinker_emb is None:
        hidden = pooling_output.get("hidden")
        thinker_emb = hidden if isinstance(hidden, torch.Tensor) else None
    if thinker_emb is None or thinker_hid is None:
        logger.warning(
            "thinker2talker_full_payload: missing thinker tensors for req=%s "
            "(embed=%s hidden=%s keys=%s); consumer wait gate may hang.",
            rid,
            thinker_emb is not None,
            thinker_hid is not None,
            list(pooling_output.keys()),
        )
        return None

    prompt_token_ids = _ensure_list(getattr(request, "prompt_token_ids", []) or [])
    all_token_ids = _ensure_list(getattr(request, "all_token_ids", None) or [])
    if not all_token_ids:
        output_token_ids = _ensure_list(getattr(request, "output_token_ids", []) or [])
        all_token_ids = list(prompt_token_ids) + list(output_token_ids)

    # Trim the trailing stop-token row from the accumulated thinker output.
    # The accumulator captures one hidden-state row per executed thinker
    # forward (prefill + every decode step including the one that emitted
    # the stop_token), so for a finished request thinker_emb has exactly one
    # row more than the rows the talker should consume. async_chunk's
    # chunk-0 path naturally captures only the prefill / non-stop portion,
    # which is why the [async_chunk] parametrization passes while [default]
    # over-generates one codec frame on short outputs (e.g.
    # test_one_word_prompt_001[default]: audio extends "London" with
    # spurious phonemes).
    if isinstance(thinker_emb, torch.Tensor) and thinker_emb.shape[0] > 0:
        thinker_emb_prefill = thinker_emb[:-1]
    else:
        thinker_emb_prefill = thinker_emb
    if isinstance(thinker_hid, torch.Tensor) and thinker_hid.shape[0] > 0:
        thinker_hid_prefill = thinker_hid[:-1]
    else:
        thinker_hid_prefill = thinker_hid

    payload: OmniPayload = {
        "embed": {
            "prefill": thinker_emb_prefill.detach().cpu(),
            "tts_bos": _as_tensor_or_none(pooling_output.get("embed.tts_bos")),
            "tts_eos": _as_tensor_or_none(pooling_output.get("embed.tts_eos")),
            "tts_pad": _as_tensor_or_none(pooling_output.get("embed.tts_pad")),
        },
        "hidden_states": {"output": thinker_hid_prefill.detach().cpu()},
        "ids": {"all": list(all_token_ids), "prompt": list(prompt_token_ids)},
        "meta": {"finished": torch.tensor(True, dtype=torch.bool)},
    }
    payload["next_stage_prompt_len"] = _compute_talker_prompt_ids_length(payload, device="cpu")
    speaker = extract_speaker_from_request(request)
    if speaker is not None:
        payload["speaker"] = speaker
    language = extract_language_from_request(request)
    if language is not None:
        payload["language"] = language
    return payload


def thinker2talker(
    source_outputs: list[Any],
    prompt: OmniTokensPrompt | TextPrompt | None = None,
    requires_multimodal_data: bool = False,
    streaming_context: Any | None = None,
) -> list[OmniTokensPrompt]:
    """
    Process thinker outputs to create talker inputs.

    Workflow:
    1. Extract thinker's text generation outputs (token IDs + hidden states)
    2. Split hidden states into: prompt embeddings + generated embeddings
    3. Package for talker with additional information

    In PD disaggregation mode, merges prefill-stage prompt embeddings with
    decode-stage generated embeddings before handing off to the talker.

    Args:
        prompt: Original prompt data
        requires_multimodal_data: Whether multimodal data is required

    Returns:
        List of OmniTokensPrompt for talker stage
    """
    thinker_outputs = source_outputs
    talker_inputs: list[OmniTokensPrompt] = []

    device = torch.device(current_platform.device_type)

    # Process each thinker output
    for i, thinker_output in enumerate(thinker_outputs):
        output = thinker_output.outputs[0]
        req_id = str(getattr(thinker_output, "request_id", f"idx-{i}"))
        prompt_token_ids = _ensure_list(thinker_output.prompt_token_ids)
        output_ids = _ensure_list(output.cumulative_token_ids)
        is_streaming_session = bool(getattr(streaming_context, "enabled", False))
        if is_streaming_session:
            prompt_token_ids, output_ids = _get_streaming_talker_tokens(
                req_id,
                prompt_token_ids,
                output_ids,
                getattr(streaming_context, "new_prompt_len_snapshot", None),
                streaming_context,
                clear_state=bool(getattr(thinker_output, "finished", False)),
            )
        thinker_sequences = prompt_token_ids + output_ids
        thinker_input_ids = prompt_token_ids
        new_seq_length = len(prompt_token_ids + output_ids) - 1
        thinker_mm_raw = getattr(output, "multimodal_output", None)
        if not isinstance(thinker_mm_raw, Mapping):
            logger.debug("thinker2talker: skip req=%s due to empty multimodal_output", req_id)
            continue
        thinker_mm: OmniPayload = thinker_mm_raw
        mm_hs = thinker_mm.get("hidden_states", {})
        mm_layers = mm_hs.get("layers", {}) if isinstance(mm_hs, Mapping) else {}
        emb_layer = _layer_tensor(mm_layers, _EMBED_LAYER_KEY)
        hid_layer = _layer_tensor(mm_layers, _HIDDEN_LAYER_KEY)
        if emb_layer is None or hid_layer is None:
            logger.debug("thinker2talker: skip req=%s due to missing hidden-state layers", req_id)
            continue
        thinker_emb = emb_layer.detach().to(device=device, dtype=torch.float)[-new_seq_length:]
        thinker_hid = hid_layer.detach().to(device=device, dtype=torch.float)[-new_seq_length:]

        prefill_mm: dict[str, Any] | None = None
        prefill_mm = _get_prefill_multimodal_output(req_id, streaming_context)

        if prefill_mm is not None:
            expected_total = len(prompt_token_ids) + len(output_ids)
            try:
                thinker_emb, thinker_hid = _merge_pd_embeddings(
                    thinker_emb, thinker_hid, prefill_mm, device, expected_total=expected_total
                )
            except Exception as exc:
                logger.warning("[PD] Could not merge prefill embeddings: %s", exc)

        payload = OmniPayloadStruct(
            embed=EmbeddingsStruct(
                prefill=thinker_emb,
                tts_bos=_resolve_tts_token_embedding(
                    "tts_bos", thinker_mm=thinker_mm, prefill_mm=prefill_mm, device=device
                ),
                tts_eos=_resolve_tts_token_embedding(
                    "tts_eos", thinker_mm=thinker_mm, prefill_mm=prefill_mm, device=device
                ),
                tts_pad=_resolve_tts_token_embedding(
                    "tts_pad", thinker_mm=thinker_mm, prefill_mm=prefill_mm, device=device
                ),
            ),
            hidden_states=HiddenStatesStruct(output=thinker_hid),
            ids=IdsStruct(all=thinker_sequences, prompt=thinker_input_ids),
            speaker=extract_speaker_from_prompt(prompt, index=i),
            language=extract_language_from_prompt(prompt, index=i),
        )
        info = to_dict(payload)
        prompt_len = _compute_talker_prompt_ids_length(info, device=device)

        talker_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=[0] * prompt_len,
                additional_information=info,
                multi_modal_data=None,
                mm_processor_kwargs=None,
            )
        )

    return talker_inputs


def thinker2talker_token_only(
    source_outputs: list[Any],
    prompt: OmniTokensPrompt | TextPrompt | None = None,
    requires_multimodal_data: bool = False,
    streaming_context: Any | None = None,
) -> list[OmniTokensPrompt]:
    """Non-async-chunk Stage-1 input builder for the connector data plane.

    The worker connector (Stage-0 ``thinker2talker_full_payload`` →
    ``_sync_local_stage_payloads``) supplies the bulk talker conditioning
    tensors (embed / hidden_states / ids) via ``model_intermediate_buffer``.
    The orchestrator only needs to ship a placeholder prefill prompt of the
    correct length so the scheduler can allocate KV-cache slots.

    Small per-request voice metadata (``speaker`` / ``language``) is forwarded
    here from the user prompt so the worker's line-408 buffer seed picks it
    up. The connector-side ``extract_speaker_from_request`` reads the
    strongly-typed ``request.additional_information.entries["speaker"]`` which
    currently does not always round-trip the user-supplied voice; until that
    plumbing is normalized, providing the small fields directly preserves
    voice selection (regression discovered on Buildkite 9668:
    ``test_speaker_002[default]`` lost the preset voice).
    """
    talker_inputs: list[OmniTokensPrompt] = []
    for i, thinker_output in enumerate(source_outputs):
        output = thinker_output.outputs[0]
        req_id = str(getattr(thinker_output, "request_id", f"idx-{i}"))
        # Skip-on-missing parity with thinker2talker_full_payload: if the
        # connector builder would drop this request (no MM dict or missing
        # hidden-state layers), do the same here so the worker buffer
        # presence agrees with the orchestrator's scheduling decision.
        thinker_mm_raw = getattr(output, "multimodal_output", None)
        if not isinstance(thinker_mm_raw, Mapping):
            logger.debug("thinker2talker_token_only: skip req=%s due to empty multimodal_output", req_id)
            continue
        mm_hs = thinker_mm_raw.get("hidden_states", {})
        mm_layers = mm_hs.get("layers", {}) if isinstance(mm_hs, Mapping) else {}
        if _layer_tensor(mm_layers, _EMBED_LAYER_KEY) is None or _layer_tensor(mm_layers, _HIDDEN_LAYER_KEY) is None:
            logger.debug("thinker2talker_token_only: skip req=%s due to missing hidden-state layers", req_id)
            continue
        prompt_token_ids = _ensure_list(thinker_output.prompt_token_ids)
        output_ids = _ensure_list(output.cumulative_token_ids)
        is_streaming_session = bool(getattr(streaming_context, "enabled", False))
        if is_streaming_session:
            prompt_token_ids, output_ids = _get_streaming_talker_tokens(
                req_id,
                prompt_token_ids,
                output_ids,
                getattr(streaming_context, "new_prompt_len_snapshot", None),
                streaming_context,
                clear_state=bool(getattr(thinker_output, "finished", False)),
            )
        thinker_sequences = prompt_token_ids + output_ids
        thinker_input_ids = prompt_token_ids
        info_for_len = {"ids": {"all": thinker_sequences, "prompt": thinker_input_ids}}
        prompt_len = _compute_talker_prompt_ids_length(info_for_len, device="cpu")

        # Forward only small voice metadata; bulk tensors come from the
        # connector path via _sync_local_stage_payloads.
        small_info: dict[str, Any] = {}
        speaker = extract_speaker_from_prompt(prompt, index=i)
        if speaker is not None:
            small_info["speaker"] = speaker
        language = extract_language_from_prompt(prompt, index=i)
        if language is not None:
            small_info["language"] = language

        talker_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=[0] * prompt_len,
                additional_information=(small_info if small_info else None),
                multi_modal_data=None,
                mm_processor_kwargs=None,
            )
        )
    return talker_inputs


# =========================
# Talker -> Code2Wav
# =========================


def talker2code2wav_async_chunk(
    transfer_manager: Any,
    multimodal_output: OmniPayload | dict[str, Any],
    request: OmniEngineCoreRequest,
    is_finished: bool = False,
) -> OmniPayloadStruct | None:
    """
    Multimodal output version.
    """
    if not isinstance(multimodal_output, Mapping):
        return None
    talker_codes = multimodal_output.get("codes", {})
    if not isinstance(talker_codes, dict):
        return None
    code_predictor_codes = talker_codes.get("audio")
    if code_predictor_codes is None:
        return None

    if code_predictor_codes.numel() == 0:
        return None

    if not code_predictor_codes.any():
        return None

    connector = getattr(transfer_manager, "connector", None)
    raw_cfg = getattr(connector, "config", {}) or {}
    cfg = raw_cfg.get("extra", raw_cfg) if isinstance(raw_cfg, dict) else {}
    chunk_size_config = int(cfg.get("codec_chunk_frames", 25))
    left_context_size_config = int(cfg.get("codec_left_context_frames", 25))
    configured_initial_chunk_size = int(cfg.get("initial_codec_chunk_frames") or 0)

    sampling_params = getattr(request, "sampling_params", None)
    stop_token_ids = set(getattr(sampling_params, "stop_token_ids", None) or [])
    stop_token_id = getattr(sampling_params, "stop_token_id", None)
    if stop_token_id is not None:
        stop_token_ids.add(stop_token_id)
    first_codebook = int(code_predictor_codes[0, 0].item())
    if first_codebook in stop_token_ids:
        logger.debug("skip stop-token codec frame: first_codebook=%s", first_codebook)
        return None

    request_id = request.external_req_id
    chunk_id = transfer_manager.put_req_chunk[request_id]
    transfer_manager.code_prompt_token_ids[request_id].append(code_predictor_codes)
    length = len(transfer_manager.code_prompt_token_ids[request_id])

    if configured_initial_chunk_size > 0:
        if chunk_id == 0:
            chunk_size_config = configured_initial_chunk_size
        else:
            length -= configured_initial_chunk_size

    chunk_length = length % chunk_size_config
    if chunk_length != 0 and not is_finished:
        return None

    context_length = chunk_length if chunk_length != 0 else chunk_size_config
    # ensure left context does not exceed available length
    if configured_initial_chunk_size > 0 and chunk_id == 1:
        left_context_size = configured_initial_chunk_size
        end_index = length + configured_initial_chunk_size
    else:
        left_context_size = max(0, min(length - context_length, left_context_size_config))
        end_index = min(length, left_context_size + context_length)

    codes = (
        torch.cat(transfer_manager.code_prompt_token_ids[request_id][-end_index:], dim=0).transpose(0, 1).reshape(-1)
    )

    return OmniPayloadStruct(
        codes=CodesStruct(audio=codes),
        meta=MetaStruct(
            left_context_size=left_context_size,
            finished=torch.tensor(is_finished, dtype=torch.bool),
        ),
    )


def talker2code2wav_full_payload(
    transfer_manager: Any,
    pooling_output: dict[str, Any],
    request: OmniEngineCoreRequest,
) -> dict[str, Any] | None:
    """Pack complete talker codec output for the non-async connector path."""
    rid = getattr(request, "request_id", None)
    if not isinstance(pooling_output, Mapping):
        logger.warning(
            "talker2code2wav_full_payload: pooling_output not a dict "
            "(type=%s) for req=%s; consumer wait gate may hang.",
            type(pooling_output).__name__,
            rid,
        )
        return None
    code_predictor_codes = pooling_output.get("codes.audio")
    if code_predictor_codes is None:
        codes = pooling_output.get("codes")
        if isinstance(codes, dict):
            code_predictor_codes = codes.get("audio")
    if code_predictor_codes is None:
        logger.warning(
            "talker2code2wav_full_payload: missing codes.audio (keys=%s) for req=%s; consumer wait gate may hang.",
            list(pooling_output.keys()),
            rid,
        )
        return None
    if not isinstance(code_predictor_codes, torch.Tensor):
        code_predictor_codes = torch.as_tensor(code_predictor_codes)
    if code_predictor_codes.numel() == 0:
        logger.warning(
            "talker2code2wav_full_payload: empty codes.audio for req=%s; consumer wait gate may hang.",
            rid,
        )
        return None

    output_token_ids = _ensure_list(getattr(request, "output_token_ids", []) or [])
    raw_shape = tuple(code_predictor_codes.shape)
    code_predictor_codes, codec_stats = _extract_qwen3_full_payload_codec_rows(
        code_predictor_codes.to(torch.long),
        list(output_token_ids),
    )
    if code_predictor_codes.numel() == 0:
        logger.warning(
            "talker2code2wav_full_payload: no valid codec rows after filtering "
            "(raw_shape=%s output_ids_len=%d aligned_rows=%s valid_rows=%s) for req=%s; "
            "consumer wait gate may hang.",
            raw_shape,
            len(output_token_ids),
            codec_stats["aligned_rows"],
            codec_stats["valid_rows"],
            rid,
        )
        return None

    codec_codes = code_predictor_codes.transpose(0, 1).cpu().reshape(-1).tolist()
    logger.debug(
        "talker2code2wav_full_payload: raw_shape=%s output_ids_len=%s aligned_rows=%s "
        "valid_rows=%s placeholders=%s flattened_len=%s pad4196=%s bos4197=%s eos4198=%s",
        raw_shape,
        len(output_token_ids),
        codec_stats["aligned_rows"],
        codec_stats["valid_rows"],
        codec_stats["trailing_placeholder_count"],
        len(codec_codes),
        sum(1 for tid in output_token_ids if tid == _QWEN3_CODEC_PAD_TOKEN_ID),
        sum(1 for tid in output_token_ids if tid == _QWEN3_CODEC_BOS_TOKEN_ID),
        sum(1 for tid in output_token_ids if tid == _QWEN3_CODEC_EOS_TOKEN_ID),
    )
    return {
        "codes": {"audio": codec_codes},
        "code_predictor_codes": codec_codes,
        "meta": {"finished": torch.tensor(True, dtype=torch.bool)},
    }


def talker2code2wav(
    source_outputs: list[Any],
    _prompt: OmniTokensPrompt | TextPrompt | None = None,
    _requires_multimodal_data: bool = False,
    streaming_context: Any | None = None,
) -> list[OmniTokensPrompt]:
    """
    Process talker outputs to create code2wav inputs.

    Workflow:
    1. Extract talker's codec code outputs (8-layer RVQ codes)
    2. Flatten codes for code2wav input
    3. Package for code2wav stage

    Args:
    Returns:
        List of OmniTokensPrompt for code2wav stage
    """
    talker_outputs = source_outputs
    code2wav_inputs: list[OmniTokensPrompt] = []
    # Process each talker output
    for i, talker_output in enumerate(talker_outputs):
        output = talker_output.outputs[0]
        req_id = str(getattr(talker_output, "request_id", f"idx-{i}"))
        cur_seq_len = len(output.cumulative_token_ids) - 1
        seq_len = cur_seq_len
        is_streaming_session = bool(getattr(streaming_context, "enabled", False))
        if is_streaming_session:
            seq_len = _get_streaming_codec_delta_len(cur_seq_len, req_id, talker_output, streaming_context)
        mm_raw = getattr(output, "multimodal_output", None)
        if not isinstance(mm_raw, Mapping):
            logger.debug("talker2code2wav: skip req=%s due to empty multimodal_output", req_id)
            continue
        mm: OmniPayload = mm_raw
        if "codes" not in mm or not isinstance(mm.get("codes"), dict) or "audio" not in mm["codes"]:
            logger.debug("talker2code2wav: skip req=%s due to missing codes.audio", req_id)
            continue
        # Extract codec codes from talker output
        # Expected shape: [8, seq_len] (8-layer RVQ codes)
        codec_codes = (
            mm["codes"]["audio"][-seq_len:].to(torch.long).transpose(0, 1).cpu().to(torch.long).reshape(-1).tolist()
        )  # 16, seq_len
        code2wav_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=codec_codes,
                multi_modal_data=None,
                mm_processor_kwargs=None,
            )
        )

    return code2wav_inputs
