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

from vllm_omni.data_entry_keys import (
    CodesStruct,
    EmbeddingsStruct,
    HiddenStatesStruct,
    IdsStruct,
    MetaStruct,
    OmniPayload,
    OmniPayloadStruct,
    to_dict,
    to_struct,
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
_QWEN3_TALKER_TEXT_VOCAB_SIZE = 3072


def _layer_tensor(layers: dict[Any, Any], key: str) -> torch.Tensor | None:
    """Fetch layer tensor with tolerant key lookup (str/int)."""
    if not isinstance(layers, dict):
        return None
    key_int = int(key)
    val = layers.get(key_int)
    if val is None:
        val = layers.get(key)
    return val if isinstance(val, torch.Tensor) else None


def _build_talker_scheduler_prompt_ids(
    info: OmniPayload,
    device: torch.device | str = "cuda",
    *,
    vocab_size: int = _QWEN3_TALKER_TEXT_VOCAB_SIZE,
) -> list[int]:
    """Build the MRv1-equivalent neutral Stage-1 sampler history."""
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

    prompt_len = 0
    for i in range(len(im_start_indexes) - 1):
        s = int(im_start_indexes[i].item())
        e = int(im_start_indexes[i + 1].item())
        role = int(input_ids[0, s + 1].item())
        if role == system_token_id:
            continue
        elif role == user_token_id:
            prompt_len += e - s
        elif role == assistant_token_id and i == len(im_start_indexes) - 2:
            prompt_len += 9  # 3 + 4 + 1 + 1
        else:
            pass

    # MRv1 schedules a length-only placeholder prompt. These ids are sampler
    # history, not model inputs: thinker text ids must not be projected or
    # clamped into the Talker codec vocabulary for repetition penalties.
    del vocab_size
    return [0] * prompt_len


def _compute_talker_prompt_ids_length(info: OmniPayload, device: torch.device | str = "cuda") -> int:
    return len(_build_talker_scheduler_prompt_ids(info, device=device))


def _attach_talker_scheduler_prompt_metadata(payload: OmniPayloadStruct) -> None:
    """Attach only the Stage-1 scheduler prompt length."""
    ids = payload.ids
    if ids is None or ids.prompt is None or ids.all is None:
        return
    scheduler_prompt_ids = _build_talker_scheduler_prompt_ids(
        {"ids": {"all": ids.all, "prompt": ids.prompt}},
        device="cpu",
    )
    if not scheduler_prompt_ids:
        return
    if payload.meta is None:
        payload.meta = MetaStruct()
    payload.meta.next_stage_prompt_len = len(scheduler_prompt_ids)


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
            _attach_talker_scheduler_prompt_metadata(payload)
            transfer_manager._pending_streaming_prefills[request_id] = to_dict(payload)
            return None
        else:
            save_payload = transfer_manager._pending_streaming_prefills.pop(request_id, None)
            if save_payload is not None:
                saved_prefill = save_payload.get("embed", {}).get("prefill")
                saved_output = save_payload.get("hidden_states", {}).get("output")
                if isinstance(saved_prefill, torch.Tensor) and isinstance(saved_output, torch.Tensor):
                    payload = OmniPayloadStruct(
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
                    _attach_talker_scheduler_prompt_metadata(payload)
                    return payload
            decode_token_start, decode_token_end = _thinker_decode_token_span(
                len(output_token_ids),
                int(emb_cpu.shape[0]),
            )
            return OmniPayloadStruct(
                meta=MetaStruct(
                    finished=finished,
                ),
                embed=EmbeddingsStruct(
                    decode=emb_cpu,
                    decode_token_start=decode_token_start,
                    decode_token_end=decode_token_end,
                ),
                hidden_states=HiddenStatesStruct(output=hid_cpu),
                ids=IdsStruct(output=output_token_ids),
                speaker=speaker,
                language=language,
            )
    else:
        if not is_finished:
            # do not send async chunk mode placeholder token or embedding/hidden of the stop token
            return None
        return OmniPayloadStruct(meta=MetaStruct(finished=finished))


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


def _copy_qwen3_thinker_batch_to_cpu(
    embeddings: torch.Tensor,
    hidden_states: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Copy one Thinker model-step conditioning batch to CPU."""
    return (
        embeddings.detach().to(device="cpu"),
        hidden_states.detach().to(device="cpu"),
    )


def _thinker_decode_token_span(output_token_count: int, row_count: int) -> tuple[int, int]:
    """Map model-output rows to absolute Thinker output-token coordinates.

    A forward row belongs to the token processed as model input. The final
    entry in ``output_token_ids`` was sampled from those rows and has not been
    processed by Thinker yet, so it must not be included in the row span.
    """
    end = int(output_token_count) - 1
    start = end - int(row_count)
    if row_count <= 0 or start < 0:
        raise ValueError(f"invalid Thinker decode row accounting: output_tokens={output_token_count} rows={row_count}")
    return start, end


def _build_thinker2talker_async_chunk_payload(
    transfer_manager: Any,
    request: OmniEngineCoreRequest,
    thinker_emb_cpu: torch.Tensor,
    thinker_hid_cpu: torch.Tensor,
    thinker_embed: Mapping[str, Any],
    *,
    is_finished: bool,
) -> OmniPayloadStruct | None:
    """Apply the per-request Thinker chunk state machine to CPU tensors."""
    request_id = request.external_req_id
    chunk_id = transfer_manager.put_req_chunk[request_id]
    speaker = extract_speaker_from_request(request)
    language = extract_language_from_request(request)

    def _maybe_cpu(t: Any) -> torch.Tensor | None:
        return t.detach().cpu() if isinstance(t, torch.Tensor) else None

    if chunk_id == 0:
        all_token_ids = _ensure_list(request.all_token_ids)
        prompt_token_ids = _ensure_list(request.prompt_token_ids)
        payload = OmniPayloadStruct(
            embed=EmbeddingsStruct(
                prefill=thinker_emb_cpu,
                tts_bos=_maybe_cpu(thinker_embed.get("tts_bos")),
                tts_eos=_maybe_cpu(thinker_embed.get("tts_eos")),
                tts_pad=_maybe_cpu(thinker_embed.get("tts_pad")),
            ),
            hidden_states=HiddenStatesStruct(output=thinker_hid_cpu),
            ids=IdsStruct(all=all_token_ids, prompt=prompt_token_ids),
            meta=MetaStruct(finished=torch.tensor(is_finished, dtype=torch.bool)),
            speaker=speaker,
            language=language,
        )
        _attach_talker_scheduler_prompt_metadata(payload)
        if transfer_manager.request_payload.get(request_id) is None:
            if not is_finished:
                transfer_manager.request_payload[request_id] = to_dict(payload)
                return None
        else:
            save_payload = transfer_manager.request_payload.pop(request_id)
            payload.embed.prefill = torch.cat(
                (save_payload.get("embed", {}).get("prefill"), payload.embed.prefill),
                dim=0,
            )
            payload.hidden_states.output = torch.cat(
                (
                    save_payload.get("hidden_states", {}).get("output"),
                    payload.hidden_states.output,
                ),
                dim=0,
            )
            prefill_shape = payload.embed.prefill.shape[0]
            if not is_finished and prefill_shape <= len(prompt_token_ids):
                transfer_manager.request_payload[request_id] = to_dict(payload)
                return None
        return payload

    if request.resumable:
        return _construct_thinker2talker_streaming_input_async_chunk(
            is_finished,
            request,
            thinker_emb_cpu,
            thinker_hid_cpu,
            transfer_manager,
        )
    output_token_ids = _ensure_list(request.output_token_ids)
    output_token_count = int(getattr(request, "output_token_count", len(output_token_ids)))
    decode_token_start, decode_token_end = _thinker_decode_token_span(
        output_token_count,
        int(thinker_emb_cpu.shape[0]),
    )
    return OmniPayloadStruct(
        meta=MetaStruct(finished=torch.tensor(is_finished, dtype=torch.bool)),
        embed=EmbeddingsStruct(
            decode=thinker_emb_cpu,
            decode_token_start=decode_token_start,
            decode_token_end=decode_token_end,
        ),
        hidden_states=HiddenStatesStruct(output=thinker_hid_cpu),
        ids=IdsStruct(output=output_token_ids or None),
        speaker=speaker,
        language=language,
    )


def thinker2talker_async_chunk(
    transfer_manager: Any,
    multimodal_output: OmniPayload | dict[str, Any] | None,
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
    if not isinstance(multimodal_output, Mapping):
        if is_finished:
            cached_payload = transfer_manager.request_payload.pop(request_id, None)
            if cached_payload is not None:
                meta = cached_payload.setdefault("meta", {})
                meta["finished"] = torch.tensor(True, dtype=torch.bool)
                return to_struct(cached_payload)
            return OmniPayloadStruct(meta=MetaStruct(finished=torch.tensor(True, dtype=torch.bool)))
        logger.debug("thinker2talker_async_chunk: skip non-dict multimodal_output for req=%s", request_id)
        return None

    thinker_hs = multimodal_output.get("hidden_states", {})
    thinker_layers = thinker_hs.get("layers", {}) if isinstance(thinker_hs, dict) else {}
    thinker_embed_raw = multimodal_output.get("embed", {})
    thinker_embed = thinker_embed_raw if isinstance(thinker_embed_raw, dict) else {}
    thinker_emb = _layer_tensor(thinker_layers, _EMBED_LAYER_KEY)
    thinker_hid = _layer_tensor(thinker_layers, _HIDDEN_LAYER_KEY)
    if thinker_emb is None or thinker_hid is None:
        logger.debug(
            "thinker2talker_async_chunk: missing thinker layers for req=%s (embed=%s hidden=%s)",
            request_id,
            thinker_emb is not None,
            thinker_hid is not None,
        )
        return None
    thinker_emb_cpu, thinker_hid_cpu = _copy_qwen3_thinker_batch_to_cpu(
        thinker_emb,
        thinker_hid,
    )
    return _build_thinker2talker_async_chunk_payload(
        transfer_manager,
        request,
        thinker_emb_cpu,
        thinker_hid_cpu,
        thinker_embed,
        is_finished=is_finished,
    )


def thinker2talker_async_chunk_batch(
    transfer_manager: Any,
    pooling_outputs: list[OmniPayload | dict[str, Any] | None],
    requests: list[OmniEngineCoreRequest],
    is_finished: list[bool],
) -> list[OmniPayloadStruct | None]:
    """Build Thinker conditioning chunks with batched embedding D2H."""
    if not (len(pooling_outputs) == len(requests) == len(is_finished)):
        raise ValueError("batch thinker inputs must have identical lengths")

    extracted: list[tuple[torch.Tensor, torch.Tensor, Mapping[str, Any]] | None] = []
    embeddings: list[torch.Tensor] = []
    hidden_states: list[torch.Tensor] = []
    for pooling_output in pooling_outputs:
        item = None
        if isinstance(pooling_output, Mapping):
            thinker_hs = pooling_output.get("hidden_states", {})
            thinker_layers = thinker_hs.get("layers", {}) if isinstance(thinker_hs, Mapping) else {}
            thinker_embed_raw = pooling_output.get("embed", {})
            thinker_embed = thinker_embed_raw if isinstance(thinker_embed_raw, Mapping) else {}
            thinker_emb = _layer_tensor(thinker_layers, _EMBED_LAYER_KEY)
            thinker_hid = _layer_tensor(thinker_layers, _HIDDEN_LAYER_KEY)
            if thinker_emb is not None and thinker_hid is not None and thinker_emb.shape[0] == thinker_hid.shape[0]:
                item = (thinker_emb, thinker_hid, thinker_embed)
                embeddings.append(thinker_emb)
                hidden_states.append(thinker_hid)
        extracted.append(item)

    if not embeddings:
        return [
            thinker2talker_async_chunk(
                transfer_manager,
                pooling_output,
                request,
                is_finished=finished,
            )
            for pooling_output, request, finished in zip(pooling_outputs, requests, is_finished)
        ]

    embeddings_cpu, hidden_states_cpu = _copy_qwen3_thinker_batch_to_cpu(
        torch.cat(embeddings, dim=0),
        torch.cat(hidden_states, dim=0),
    )
    offset = 0
    payloads: list[OmniPayloadStruct | None] = []
    for item, pooling_output, request, finished in zip(extracted, pooling_outputs, requests, is_finished):
        if item is None:
            payloads.append(
                thinker2talker_async_chunk(
                    transfer_manager,
                    pooling_output,
                    request,
                    is_finished=finished,
                )
            )
            continue
        thinker_emb, _thinker_hid, thinker_embed = item
        row_count = int(thinker_emb.shape[0])
        payloads.append(
            _build_thinker2talker_async_chunk_payload(
                transfer_manager,
                request,
                embeddings_cpu[offset : offset + row_count],
                hidden_states_cpu[offset : offset + row_count],
                thinker_embed,
                is_finished=finished,
            )
        )
        offset += row_count
    return payloads


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

    # Drop the terminal stop-token row only when more than one row was
    # accumulated; trimming a single row would ship 0 conditioning tensors
    # while ids still has tokens and break talker prefill alignment.
    if isinstance(thinker_emb, torch.Tensor) and thinker_emb.shape[0] > 1:
        thinker_emb_prefill = thinker_emb[:-1]
    else:
        thinker_emb_prefill = thinker_emb
    if isinstance(thinker_hid, torch.Tensor) and thinker_hid.shape[0] > 1:
        thinker_hid_prefill = thinker_hid[:-1]
    else:
        thinker_hid_prefill = thinker_hid

    emb_rows = int(thinker_emb_prefill.shape[0]) if isinstance(thinker_emb_prefill, torch.Tensor) else 0
    hid_rows = int(thinker_hid_prefill.shape[0]) if isinstance(thinker_hid_prefill, torch.Tensor) else 0
    if len(all_token_ids) > 0 and (emb_rows == 0 or hid_rows == 0):
        logger.warning(
            "thinker2talker_full_payload: empty thinker conditioning for req=%s "
            "(ids_len=%s embed_rows=%s hidden_rows=%s); withholding payload.",
            rid,
            len(all_token_ids),
            emb_rows,
            hid_rows,
        )
        return None

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
    """Build native MRv2 Stage-1 scheduler placeholders only.

    Bulk thinker tensors are delivered by the connector into
    ``model_intermediate_buffer``.  This function only sizes scheduler
    placeholders and forwards small request metadata.
    """
    return thinker2talker_token_only(
        source_outputs=source_outputs,
        prompt=prompt,
        requires_multimodal_data=requires_multimodal_data,
        streaming_context=streaming_context,
    )


def thinker2talker_token_only(
    source_outputs: list[Any],
    prompt: OmniTokensPrompt | TextPrompt | None = None,
    requires_multimodal_data: bool = False,
    streaming_context: Any | None = None,
) -> list[OmniTokensPrompt]:
    """Orchestrator-side placeholder builder for Stage-1 (Talker) when
    ``async_chunk=False``.

    After the communication-layer refactor, this function only allocates a
    placeholder ``prompt_token_ids`` of the correct length so the scheduler can
    reserve KV-cache slots. It does **not** forward bulk tensors.

    Bulk talker conditioning is sent through the connector. Speaker and
    language are also copied from the original prompt so they survive when
    Stage-0 request metadata is unavailable to the connector payload.

    ``prompt`` / ``requires_multimodal_data`` are kept for call-site signature
    compatibility with other orchestrator input processors; they are unused.
    """
    talker_inputs: list[OmniTokensPrompt] = []
    for i, thinker_output in enumerate(source_outputs):
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
        info_for_len = {"ids": {"all": thinker_sequences, "prompt": thinker_input_ids}}
        scheduler_prompt_ids = _build_talker_scheduler_prompt_ids(info_for_len, device="cpu")

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
                prompt_token_ids=scheduler_prompt_ids,
                additional_information=(small_info if small_info else None),
                multi_modal_data=None,
                mm_processor_kwargs=None,
            )
        )
    return talker_inputs


# =========================
# Talker -> Code2Wav
# =========================


def _eof_payload(transfer_manager: Any) -> OmniPayloadStruct:
    """Return an EOF marker payload when the request is finished with no codes."""
    return OmniPayloadStruct(
        codes=CodesStruct(audio=torch.empty((0,), dtype=torch.long)),
        meta=MetaStruct(
            left_context_size=0,
            finished=torch.tensor(True, dtype=torch.bool),
            is_segment_finished=torch.tensor(True, dtype=torch.bool),
        ),
    )


def _filter_qwen3_async_codec_rows(
    code_predictor_codes: torch.Tensor,
    output_token_ids: list[int],
) -> tuple[torch.Tensor, dict[str, int]]:
    """Keep only real codec rows before sending an async Code2Wav chunk.

    Talker can expose rows for placeholder / terminal positions. Those rows are
    not valid codec ids and must not be written into vLLM V2 int32 token state.
    Async chunks cannot rely on request.output_token_ids as the row mask here,
    so keep the contract local to Code2Wav: every emitted row must be in the
    codec id range.
    """
    code_predictor_codes = code_predictor_codes.detach()
    if code_predictor_codes.dtype != torch.long:
        code_predictor_codes = code_predictor_codes.to(dtype=torch.long)
    # In async-chunk mode, request.output_token_ids is not a reliable codec-row
    # mask at this point in the pipeline. Keep the contract local to Code2Wav:
    # only real codec ids in [0, codebook_size) may enter the token buffer.
    if code_predictor_codes.ndim != 2 or code_predictor_codes.numel() == 0:
        return code_predictor_codes[:0], {
            "raw_rows": int(code_predictor_codes.shape[0]) if code_predictor_codes.ndim > 0 else 0,
            "aligned_rows": 0,
            "valid_rows": 0,
            "trailing_placeholder_count": 0,
        }
    row_valid_mask = (code_predictor_codes.max(dim=1).values < _QWEN3_CODEC_CODEBOOK_SIZE) & (
        code_predictor_codes.min(dim=1).values >= 0
    )
    filtered = code_predictor_codes[row_valid_mask]
    return filtered, {
        "raw_rows": int(code_predictor_codes.shape[0]),
        "aligned_rows": int(code_predictor_codes.shape[0]),
        "valid_rows": int(filtered.shape[0]),
        "trailing_placeholder_count": 0,
    }


def _copy_qwen3_codec_batch_to_cpu(rows: torch.Tensor) -> torch.Tensor:
    """Copy one model-step codec batch to CPU with a single synchronization."""
    return rows.detach().to(dtype=torch.long, device="cpu")


def _codec_frame_valid_mask(
    multimodal_output: Mapping[str, Any],
    *,
    row_count: int,
    device: torch.device,
) -> torch.Tensor | None:
    """Return the producer's authoritative per-row codec validity mask.

    ``None`` identifies a legacy payload. Those payloads retain their existing
    value-based placeholder compatibility below; native MRv2 always supplies
    this field and never infers validity from codec IDs.
    """
    meta = multimodal_output.get("meta", {})
    if not isinstance(meta, Mapping):
        return None
    raw_validity = meta.get("codec_frame_valid")
    if raw_validity is None:
        return None
    if isinstance(raw_validity, torch.Tensor):
        if raw_validity.numel() == 0:
            raise ValueError("codec_frame_valid must not be empty")
        validity = raw_validity.detach().to(device=device, dtype=torch.bool).reshape(-1)
    elif isinstance(raw_validity, bool):
        validity = torch.tensor([raw_validity], dtype=torch.bool, device=device)
    else:
        raise TypeError(f"codec_frame_valid must be a bool or tensor, got {type(raw_validity).__name__}")
    if validity.numel() == 1:
        return validity.expand(row_count)
    if validity.numel() != row_count:
        raise ValueError(
            f"codec_frame_valid changed the codec row axis: expected={row_count} actual={validity.numel()}"
        )
    return validity


def _codec_stop_token_ids(request: Any) -> set[int]:
    sampling_params = getattr(request, "sampling_params", None)
    stop_token_ids = set(getattr(sampling_params, "stop_token_ids", None) or [])
    stop_token_id = getattr(sampling_params, "stop_token_id", None)
    if stop_token_id is not None:
        stop_token_ids.add(stop_token_id)
    return {int(token_id) for token_id in stop_token_ids if 0 <= int(token_id) < _QWEN3_CODEC_CODEBOOK_SIZE}


def talker2code2wav_async_chunk_batch(
    transfer_manager: Any,
    pooling_outputs: list[OmniPayload | dict[str, Any] | None],
    requests: list[OmniEngineCoreRequest],
    is_finished: list[bool],
) -> list[OmniPayloadStruct | None]:
    """Build one Talker model-step worth of Code2Wav chunks in a batch.

    Codec tensors remain on GPU until all request rows from the step have been
    concatenated. The hot path then performs one D2H copy and filters the tiny
    codec matrix on CPU before updating each request's chunk window.
    """
    if not (len(pooling_outputs) == len(requests) == len(is_finished)):
        raise ValueError("batch codec inputs must have identical lengths")

    tensors: list[torch.Tensor] = []
    row_counts = [0] * len(requests)
    validity_masks: list[torch.Tensor | None] = [None] * len(requests)
    for index, pooling_output in enumerate(pooling_outputs):
        if not isinstance(pooling_output, Mapping):
            continue
        talker_codes = pooling_output.get("codes", {})
        if not isinstance(talker_codes, Mapping):
            continue
        rows = talker_codes.get("audio")
        if not isinstance(rows, torch.Tensor) or rows.ndim != 2 or rows.numel() == 0:
            continue
        row_counts[index] = int(rows.shape[0])
        validity_masks[index] = _codec_frame_valid_mask(
            pooling_output,
            row_count=row_counts[index],
            device=rows.device,
        )
        tensors.append(rows.detach())

    cpu_rows_by_request: list[torch.Tensor | None] = [None] * len(requests)
    if tensors:
        combined_cpu = _copy_qwen3_codec_batch_to_cpu(torch.cat(tensors, dim=0))
        offset = 0
        for index, row_count in enumerate(row_counts):
            if row_count == 0:
                continue
            rows_cpu = combined_cpu[offset : offset + row_count]
            offset += row_count
            valid_mask = (rows_cpu >= 0).all(dim=1) & (rows_cpu < _QWEN3_CODEC_CODEBOOK_SIZE).all(dim=1)
            explicit_validity = validity_masks[index]
            if explicit_validity is not None:
                valid_mask &= explicit_validity.to(device="cpu", dtype=torch.bool)
            cpu_rows_by_request[index] = rows_cpu[valid_mask]

    payloads: list[OmniPayloadStruct | None] = []
    for request, rows_cpu, finished, explicit_validity in zip(
        requests,
        cpu_rows_by_request,
        is_finished,
        validity_masks,
        strict=True,
    ):
        request_id = request.external_req_id
        if rows_cpu is not None and rows_cpu.numel() > 0 and (explicit_validity is not None or bool(rows_cpu.any())):
            stop_token_ids = _codec_stop_token_ids(request)
            first_codebook = int(rows_cpu[0, 0])
            if first_codebook not in stop_token_ids:
                token_frames = transfer_manager.code_prompt_token_ids[request_id]
                token_frames.extend(row.reshape(-1) for row in rows_cpu.unbind(0))
        payloads.append(
            _build_qwen3_async_code2wav_payload_from_buffer(
                transfer_manager,
                request_id,
                is_finished=finished,
            )
        )
    return payloads


def _build_qwen3_async_code2wav_payload_from_buffer(
    transfer_manager: Any,
    request_id: str,
    *,
    is_finished: bool,
) -> OmniPayloadStruct | None:
    token_frames = transfer_manager.code_prompt_token_ids[request_id]
    length = len(token_frames)
    if length == 0:
        if is_finished:
            return _eof_payload(transfer_manager)
        return None

    connector = getattr(transfer_manager, "connector", None)
    raw_cfg = getattr(connector, "config", {}) or {}
    cfg = raw_cfg.get("extra", raw_cfg) if isinstance(raw_cfg, dict) else {}
    chunk_size_config = int(cfg.get("codec_chunk_frames", 25))
    left_context_size_config = int(cfg.get("codec_left_context_frames", 25))
    configured_initial_chunk_size = int(cfg.get("initial_codec_chunk_frames") or 0)

    emitted_by_req = getattr(transfer_manager, "_qwen3_omni_emitted_frames", None)
    if emitted_by_req is None:
        emitted_by_req = {}
        transfer_manager._qwen3_omni_emitted_frames = emitted_by_req
    previous_emit = int(emitted_by_req.get(request_id, 0))
    if configured_initial_chunk_size > 0 and previous_emit == 0:
        target_emit = min(length, configured_initial_chunk_size) if is_finished else configured_initial_chunk_size
    else:
        target_emit = length if is_finished else previous_emit + chunk_size_config

    if length < target_emit:
        return None
    if target_emit <= previous_emit:
        if is_finished:
            return _eof_payload(transfer_manager)
        return None

    context_length = target_emit - previous_emit
    left_context_size = max(0, min(previous_emit, left_context_size_config))
    window_start = max(0, target_emit - context_length - left_context_size)
    window_frames = token_frames[window_start:target_emit]
    codes = torch.stack(window_frames, dim=0).transpose(0, 1).reshape(-1)
    emitted_by_req[request_id] = target_emit

    return OmniPayloadStruct(
        codes=CodesStruct(audio=codes),
        meta=MetaStruct(
            left_context_size=left_context_size,
            finished=torch.tensor(is_finished, dtype=torch.bool),
            is_segment_finished=torch.tensor(is_finished, dtype=torch.bool),
        ),
    )


def talker2code2wav_async_chunk_prepare(
    transfer_manager: Any,
    multimodal_output: OmniPayload | dict[str, Any],
    request: OmniEngineCoreRequest,
    is_finished: bool = False,
) -> OmniPayloadStruct | None:
    """
    Multimodal output version.
    """
    request_id = request.external_req_id
    if not isinstance(multimodal_output, Mapping):
        return _build_qwen3_async_code2wav_payload_from_buffer(
            transfer_manager,
            request_id,
            is_finished=is_finished,
        )
    talker_codes = multimodal_output.get("codes", {})
    if not isinstance(talker_codes, dict):
        return _build_qwen3_async_code2wav_payload_from_buffer(
            transfer_manager,
            request_id,
            is_finished=is_finished,
        )
    code_predictor_codes = talker_codes.get("audio")
    if code_predictor_codes is None:
        return _build_qwen3_async_code2wav_payload_from_buffer(
            transfer_manager,
            request_id,
            is_finished=is_finished,
        )

    if code_predictor_codes.numel() == 0:
        return _build_qwen3_async_code2wav_payload_from_buffer(
            transfer_manager,
            request_id,
            is_finished=is_finished,
        )

    explicit_validity = _codec_frame_valid_mask(
        multimodal_output,
        row_count=int(code_predictor_codes.shape[0]) if code_predictor_codes.ndim == 2 else 1,
        device=code_predictor_codes.device,
    )
    if explicit_validity is not None:
        if code_predictor_codes.ndim == 2:
            code_predictor_codes = code_predictor_codes[explicit_validity]
        elif not bool(explicit_validity[-1].item()):
            code_predictor_codes = code_predictor_codes[:0]
        if code_predictor_codes.numel() == 0:
            return _build_qwen3_async_code2wav_payload_from_buffer(
                transfer_manager,
                request_id,
                is_finished=is_finished,
            )
    elif not code_predictor_codes.any():
        return _build_qwen3_async_code2wav_payload_from_buffer(
            transfer_manager,
            request_id,
            is_finished=is_finished,
        )

    raw_shape = tuple(code_predictor_codes.shape)
    output_token_ids = _ensure_list(getattr(request, "output_token_ids", []) or [])
    code_predictor_codes, codec_stats = _filter_qwen3_async_codec_rows(
        code_predictor_codes,
        list(output_token_ids),
    )
    if code_predictor_codes.numel() == 0:
        logger.debug(
            "talker2code2wav_async_chunk: no valid codec rows after filtering "
            "(raw_shape=%s output_ids_len=%d aligned_rows=%s valid_rows=%s placeholders=%s) for req=%s",
            raw_shape,
            len(output_token_ids),
            codec_stats["aligned_rows"],
            codec_stats["valid_rows"],
            codec_stats["trailing_placeholder_count"],
            getattr(request, "request_id", None),
        )
        return _build_qwen3_async_code2wav_payload_from_buffer(
            transfer_manager,
            request_id,
            is_finished=is_finished,
        )

    sampling_params = getattr(request, "sampling_params", None)
    stop_token_ids = set(getattr(sampling_params, "stop_token_ids", None) or [])
    stop_token_id = getattr(sampling_params, "stop_token_id", None)
    if stop_token_id is not None:
        stop_token_ids.add(stop_token_id)
    codec_stop_token_ids = {
        int(token_id) for token_id in stop_token_ids if 0 <= int(token_id) < _QWEN3_CODEC_CODEBOOK_SIZE
    }
    if codec_stop_token_ids:
        first_codebook = int(code_predictor_codes[0, 0].detach().cpu().item())
        if first_codebook in codec_stop_token_ids:
            logger.debug("skip stop-token codec frame: first_codebook=%s", first_codebook)
            return _build_qwen3_async_code2wav_payload_from_buffer(
                transfer_manager,
                request_id,
                is_finished=is_finished,
            )

    token_frames = transfer_manager.code_prompt_token_ids[request_id]
    code_predictor_codes_cpu = code_predictor_codes.to(dtype=torch.long, device="cpu")
    token_frames.extend(row.reshape(-1) for row in code_predictor_codes_cpu.unbind(0))
    return _build_qwen3_async_code2wav_payload_from_buffer(
        transfer_manager,
        request_id,
        is_finished=is_finished,
    )


def talker2code2wav_async_chunk(
    transfer_manager: Any,
    multimodal_output: OmniPayload | dict[str, Any],
    request: OmniEngineCoreRequest,
    is_finished: bool = False,
) -> OmniPayloadStruct | None:
    """Compatibility wrapper for callers that still process in save_loop.

    OmniChunkTransferAdapter discovers ``talker2code2wav_async_chunk_prepare``
    and runs it before enqueueing the save task. That path appends codec rows,
    checks chunk readiness, and builds the payload in the scheduler thread so
    save_loop only handles real connector puts. Direct callers keep the same
    behavior through this wrapper.
    """
    return talker2code2wav_async_chunk_prepare(
        transfer_manager=transfer_manager,
        multimodal_output=multimodal_output,
        request=request,
        is_finished=is_finished,
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
        "meta": {"finished": torch.tensor(True, dtype=torch.bool)},
    }
