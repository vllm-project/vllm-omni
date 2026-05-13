# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2025 The Qwen team.
"""Stage input processor for Qwen3 Omni MoE: Thinker → Talker transition."""

import logging
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


def _layer_tensor(layers: dict[Any, Any], key: str) -> torch.Tensor | None:
    """Fetch layer tensor with tolerant key lookup (str/int)."""
    if not isinstance(layers, dict):
        return None
    key_int = int(key)
    val = layers.get(key_int)
    if val is None:
        val = layers.get(key)
    return val if isinstance(val, torch.Tensor) else None


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
    return prefill_mm if isinstance(prefill_mm, dict) else None


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
        available_keys = list(prefill_mm.keys()) if isinstance(prefill_mm, dict) else type(prefill_mm).__name__
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
) -> tuple[list[int], list[int], list[int], list[int]]:
    """Return streaming token slices and merged token views for thinker->talker.
       e.g. For the second streaming input request:
       merged_sequences: [input_prompt 1, output_tokens 1[:-1], input_prompt 2, output_tokens 2]
      thinker_input_ids: [input_prompt 1, output_tokens 1[:-1], input_prompt 2]
    Returns:
        inc_prompt: prompt token delta for this segment.
        inc_output: output token delta for this segment.
        merged_sequences: full thinker_sequences to send downstream.
        thinker_input_ids: full thinker_input_ids paired with merged_sequences.
    """
    state = _get_qwen3_streaming_state(request_id, streaming_context).thinker2talker
    if new_prompt_len_snapshot:
        prompt_token_ids = prompt_token_ids[:-new_prompt_len_snapshot]
    cur_prompt_len = len(prompt_token_ids)
    cur_output_len = len(output_token_ids)

    inc_prompt = prompt_token_ids[state.last_prompt_len :]
    inc_output = output_token_ids[state.last_output_len :]
    delta_sequences = inc_prompt + inc_output
    cached_sequences = state.merged_sequences

    merged_sequences = cached_sequences + delta_sequences
    thinker_input_ids = cached_sequences + inc_prompt

    # Persist history for next segment. Drop the latest sampled token to keep
    # thinker_input_ids / thinker_sequences alignment with next-step append.
    cached_sequences.extend(delta_sequences[:-1])

    state.last_prompt_len = cur_prompt_len
    state.last_output_len = cur_output_len

    if clear_state:
        state.last_prompt_len = 0
        state.last_output_len = 0
        state.merged_sequences.clear()

    return inc_prompt, inc_output, merged_sequences, thinker_input_ids


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
    pooling_output: OmniPayload,
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
    if not isinstance(pooling_output, dict):
        logger.debug("thinker2talker_async_chunk: skip non-dict pooling_output for req=%s", request_id)
        return None

    thinker_hs = pooling_output.get("hidden_states", {})
    thinker_layers = thinker_hs.get("layers", {}) if isinstance(thinker_hs, dict) else {}
    thinker_embed_raw = pooling_output.get("embed", {})
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
    speaker = extract_speaker_from_request(request)
    language = extract_language_from_request(request)

    def _maybe_cpu(t: Any) -> torch.Tensor | None:
        return t.detach().cpu() if isinstance(t, torch.Tensor) else None

    if chunk_id == 0:
        all_token_ids = _ensure_list(request.all_token_ids)
        prompt_token_ids = _ensure_list(request.prompt_token_ids)
        payload = OmniPayloadStruct(
            embed=EmbeddingsStruct(
                prefill=thinker_emb.detach().cpu(),
                tts_bos=_maybe_cpu(thinker_embed.get("tts_bos")),
                tts_eos=_maybe_cpu(thinker_embed.get("tts_eos")),
                tts_pad=_maybe_cpu(thinker_embed.get("tts_pad")),
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
    else:
        output_token_ids = _ensure_list(request.output_token_ids)
        meta = MetaStruct(finished=torch.tensor(is_finished, dtype=torch.bool))
        if output_token_ids:
            meta.override_keys = [("embed", "decode"), ("ids", "output")]
            payload = OmniPayloadStruct(
                meta=meta,
                embed=EmbeddingsStruct(decode=thinker_emb.detach().cpu()),
                ids=IdsStruct(output=output_token_ids),
                speaker=speaker,
                language=language,
            )
        else:
            # When prefilling a chunked thinker, thinker_hidden_states needs to be updated.
            payload = OmniPayloadStruct(
                meta=meta,
                embed=EmbeddingsStruct(prefill=thinker_emb.detach().cpu()),
                hidden_states=HiddenStatesStruct(output=thinker_hid.detach().cpu()),
                speaker=speaker,
                language=language,
            )
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
            prompt_token_ids, output_ids, thinker_sequences, thinker_input_ids = _get_streaming_talker_tokens(
                req_id,
                prompt_token_ids,
                output_ids,
                getattr(streaming_context, "new_prompt_len_snapshot", None),
                streaming_context,
                clear_state=bool(getattr(thinker_output, "finished", False)),
            )
        else:
            thinker_sequences = prompt_token_ids + output_ids
            thinker_input_ids = prompt_token_ids
        new_seq_length = len(prompt_token_ids + output_ids) - 1
        thinker_mm_raw = getattr(output, "multimodal_output", None)
        if not isinstance(thinker_mm_raw, dict):
            logger.debug("thinker2talker: skip req=%s due to empty multimodal_output", req_id)
            continue
        thinker_mm: OmniPayload = thinker_mm_raw
        mm_hs = thinker_mm.get("hidden_states", {})
        mm_layers = mm_hs.get("layers", {}) if isinstance(mm_hs, dict) else {}
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


# =========================
# Talker -> Code2Wav
# =========================


def talker2code2wav_async_chunk(
    transfer_manager: Any,
    pooling_output: OmniPayload,
    request: OmniEngineCoreRequest,
    is_finished: bool = False,
) -> OmniPayloadStruct | None:
    """
    Pooling version.
    """
    if not isinstance(pooling_output, dict):
        return None
    talker_codes = pooling_output.get("codes", {})
    if not isinstance(talker_codes, dict):
        return None
    code_predictor_codes = talker_codes.get("audio")
    if code_predictor_codes is None:
        return None

    connector = getattr(transfer_manager, "connector", None)
    raw_cfg = getattr(connector, "config", {}) or {}
    cfg = raw_cfg.get("extra", raw_cfg) if isinstance(raw_cfg, dict) else {}
    chunk_size_config = int(cfg.get("codec_chunk_frames", 25))
    left_context_size_config = int(cfg.get("codec_left_context_frames", 25))

    if code_predictor_codes is None:
        return None
    if isinstance(code_predictor_codes, torch.Tensor):
        if code_predictor_codes.numel() == 0:
            return None
    elif hasattr(code_predictor_codes, "__len__"):
        if len(code_predictor_codes) == 0:
            return None

    if isinstance(code_predictor_codes, torch.Tensor):
        if not code_predictor_codes.any():
            return None
    else:
        code_tensor = torch.tensor(code_predictor_codes, dtype=torch.long)
        if not code_tensor.any():
            return None

    codec_codes = code_predictor_codes.to(torch.long).transpose(0, 1).cpu().to(torch.long).reshape(-1).tolist()
    if sum(codec_codes) == 0:
        return None

    request_id = request.external_req_id
    transfer_manager.code_prompt_token_ids[request_id].append(codec_codes)
    length = len(transfer_manager.code_prompt_token_ids[request_id])

    chunk_length = length % chunk_size_config
    if chunk_length != 0 and not is_finished:
        return None

    context_length = chunk_length if chunk_length != 0 else chunk_size_config
    # ensure left context does not exceed available length
    left_context_size = max(0, min(length - context_length, left_context_size_config))
    end_index = min(length, left_context_size + context_length)

    codes = torch.tensor(transfer_manager.code_prompt_token_ids[request_id][-end_index:]).transpose(0, 1).reshape(-1)

    return OmniPayloadStruct(
        codes=CodesStruct(audio=codes),
        meta=MetaStruct(
            left_context_size=left_context_size,
            finished=torch.tensor(is_finished, dtype=torch.bool),
        ),
    )


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
        if not isinstance(mm_raw, dict):
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
