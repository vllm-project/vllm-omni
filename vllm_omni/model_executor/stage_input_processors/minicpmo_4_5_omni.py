# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage input processor for MiniCPM-o 4.5: Thinker (LLM) -> Talker (TTS).

This bridge converts thinker hidden states and token ids into the talker
prompt payload. The talker runs Token2Wav internally; MiniCPM-o 4.5 does not
have an independent code2wav pipeline stage.
"""

import logging
from collections.abc import Callable, Mapping

import torch
from vllm.inputs import TextPrompt

from vllm_omni.experimental.fullduplex.engine.intermediate import (
    build_duplex_intermediate_buffer,
    set_ref_audio,
    set_tts_handoff,
)
from vllm_omni.inputs.data import OmniTokensPrompt

logger = logging.getLogger(__name__)


def _extract_first_audio_ref(multi_modal_data):
    if not isinstance(multi_modal_data, dict):
        return None
    audio_data = multi_modal_data.get("audio")
    if audio_data is None:
        return None
    if isinstance(audio_data, list):
        if not audio_data:
            return None
        audio_data = audio_data[0]

    samples = None
    sample_rate = None
    if isinstance(audio_data, tuple) and len(audio_data) >= 2:
        samples, sample_rate = audio_data[0], audio_data[1]
    elif isinstance(audio_data, dict):
        sample_rate = audio_data.get("sample_rate") or audio_data.get("sampling_rate") or audio_data.get("sr")
        for key in ("audio", "wav", "samples", "array", "waveform"):
            if key in audio_data:
                samples = audio_data[key]
                break
    if samples is None or sample_rate is None:
        return None

    waveform = torch.as_tensor(samples, dtype=torch.float32)
    if waveform.ndim > 1:
        if waveform.shape[0] <= 2 and waveform.shape[-1] > waveform.shape[0]:
            waveform = waveform.mean(dim=0)
        else:
            waveform = waveform.mean(dim=-1)
    return waveform.reshape(-1).cpu(), int(sample_rate)


def _extract_native_runtime_ref_audio(data_plane_metadata):
    if not isinstance(data_plane_metadata, dict):
        return None
    runtime_config = data_plane_metadata.get("runtime_config")
    if not isinstance(runtime_config, dict):
        return None
    from vllm_omni.experimental.fullduplex.minicpmo45.input import decode_native_ref_audio_from_config

    waveform = decode_native_ref_audio_from_config({"extra_body": runtime_config})
    if waveform is None:
        return None
    sample_rate = runtime_config.get("ref_audio_sample_rate_hz") or 16000
    return torch.as_tensor(waveform, dtype=torch.float32).reshape(-1).cpu(), int(sample_rate)


def _coerce_token_id_list(value):
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu().reshape(-1).tolist()
    if isinstance(value, list) and value and isinstance(value[0], list):
        value = value[0]
    if isinstance(value, tuple):
        value = list(value)
    if not isinstance(value, list):
        return None
    out = []
    for item in value:
        try:
            out.append(int(item))
        except (TypeError, ValueError):
            return None
    return out


def _to_transport_list(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if isinstance(value, torch.Tensor):
        return value.tolist()
    return value


def _coerce_int(value):
    if hasattr(value, "detach"):
        flat = value.detach().cpu().reshape(-1)
        if flat.numel() == 0:
            return None
        value = flat[0].item()
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _special_token_ids_from_mm_output(mm_output):
    if not isinstance(mm_output, Mapping):
        return {}
    meta = mm_output.get("meta")
    if not isinstance(meta, dict):
        meta = {}
    special_token_ids = mm_output.get("special_token_ids")
    if not isinstance(special_token_ids, dict):
        special_token_ids = {}
    flat_meta = {
        key.removeprefix("meta."): value
        for key, value in mm_output.items()
        if isinstance(key, str) and key.startswith("meta.")
    }
    return {
        key: value
        for key, value in (
            (key, _coerce_int(value))
            for source in (special_token_ids, meta, flat_meta)
            for key, value in source.items()
        )
        if value is not None and value >= 0
    }


def _has_native_duplex_prompt_metadata(mm_output):
    if not isinstance(mm_output, Mapping):
        return False
    return mm_output.get("duplex_prompt_token_ids") is not None or mm_output.get("ids.prompt") is not None


def _require_native_tts_boundary_metadata(special_token_ids):
    if special_token_ids.get("tts_bos_token_id") is None:
        raise ValueError(
            "MiniCPM-o native duplex TTS handoff requires tokenizer-derived "
            "<|tts_bos|> metadata; refusing to infer special token ids."
        )


def _native_tts_boundary_token_ids(special_token_ids):
    # Official duplex conditions the talker on mid-unit <|speak|> tokens AND
    # the <|turn_eos|> token+hidden (its embedding is the trained stop
    # signal); only chunk terminators and framing tokens bound the slice.
    return {
        token_id
        for token_id in (
            special_token_ids.get("tts_eos_token_id"),
            special_token_ids.get("tts_pad_token_id"),
            special_token_ids.get("listen_token_id"),
            special_token_ids.get("chunk_eos_token_id"),
            special_token_ids.get("chunk_tts_eos_token_id"),
            special_token_ids.get("unit_token_id"),
            special_token_ids.get("unit_end_token_id"),
        )
        if token_id is not None
    }


def _native_duplex_segment_output_ids(
    output_ids: list[int],
    output_text: str,
    streaming_context,
    *,
    request_id: str,
) -> tuple[list[int], str]:
    """Slice the cumulative thinker output down to the current segment.

    Tracks how many output tokens were already handed to the talker in the
    orchestrator's streaming bridge state. Segment transcript is decoded from
    the same token delta, not from a character cursor over cumulative text.
    """
    bridge_states = getattr(streaming_context, "bridge_states", None)
    if not isinstance(bridge_states, dict):
        return output_ids, output_text
    state = bridge_states.setdefault("minicpmo45_tts_handoff", {})
    duplex_state = bridge_states.get("duplex")
    turn_id = duplex_state.get("model_turn_id", duplex_state.get("turn_id")) if isinstance(duplex_state, dict) else None
    if not isinstance(turn_id, int):
        turn_id = None
    if state.get("request_id") != request_id:
        state["request_id"] = request_id
        state["sent_output_len"] = 0
        state["sent_output_ids"] = []
        state["acc_tts_ids"] = []
        state["acc_tts_hidden"] = []
    sent_len = state.get("sent_output_len", 0)
    prev_output_ids = state.get("sent_output_ids", [])
    prev_turn_id = state.get("turn_id")
    if not isinstance(sent_len, int) or sent_len < 0 or sent_len > len(output_ids):
        # Shrunken cumulative output = epoch reset after barge-in: the talker
        # condition history is stale too.
        sent_len = 0
        state["acc_tts_ids"] = []
        state["acc_tts_hidden"] = []
    elif sent_len and isinstance(prev_output_ids, list):
        prev_prefix = prev_output_ids[:sent_len]
        current_prefix = output_ids[:sent_len]
        if current_prefix != prev_prefix:
            # Some runtimes restart output token ids after a turn boundary while
            # others keep returning cumulative ids. Detect restart from the
            # token prefix instead of clearing the cursor at turn_eos.
            sent_len = 0
            state["acc_tts_ids"] = []
            state["acc_tts_hidden"] = []
    if isinstance(prev_turn_id, int) and isinstance(turn_id, int) and prev_turn_id != turn_id:
        # The thinker output list is cumulative across clean turns, so the
        # output cursor must stay put. The talker condition is per assistant
        # turn, though; carrying it across turn_id boundaries makes stage 1
        # replay the previous turn after its consumed cursor has been reset.
        state["acc_tts_ids"] = []
        state["acc_tts_hidden"] = []
    segment_ids = output_ids[sent_len:]
    decode_token_ids = getattr(streaming_context, "source_token_decoder", None)
    if isinstance(decode_token_ids, Callable):
        decode_ids = [int(token_id) for token_id in segment_ids]
        try:
            try:
                segment_text = str(decode_token_ids(decode_ids, skip_special_tokens=True))
            except TypeError:
                segment_text = str(decode_token_ids(decode_ids))
        except Exception:
            logger.exception("Failed to decode MiniCPM-o duplex token delta for request_id=%s", request_id)
            segment_text = ""
    elif sent_len == 0:
        # Non-orchestrator unit tests and legacy callers may not install a
        # decoder. Never slice cumulative text with a stale character cursor.
        segment_text = output_text
    else:
        logger.warning(
            "MiniCPM-o native duplex token delta decoder missing for request_id=%s; "
            "suppressing cumulative transcript fallback",
            request_id,
        )
        segment_text = ""
    state["sent_output_len"] = len(output_ids)
    state["sent_output_ids"] = list(output_ids)
    state["turn_id"] = turn_id
    return segment_ids, segment_text


def _reset_native_tts_handoff(streaming_context) -> None:
    bridge_states = getattr(streaming_context, "bridge_states", None)
    if not isinstance(bridge_states, dict):
        return
    state = bridge_states.get("minicpmo45_tts_handoff")
    if not isinstance(state, dict):
        return
    # Keep the output cursor: the thinker can keep reporting cumulative
    # output after turn_eos on the same resumable request. If a runtime really
    # restarts token ids, _native_duplex_segment_output_ids detects the prefix
    # mismatch and resets the slice cursor there.
    state["acc_tts_ids"] = []
    state["acc_tts_hidden"] = []


def _native_duplex_data_plane_metadata(streaming_context) -> dict[str, object] | None:
    bridge_states = getattr(streaming_context, "bridge_states", None)
    if not isinstance(bridge_states, dict):
        return None
    duplex_state = bridge_states.get("duplex")
    if not isinstance(duplex_state, dict):
        return None

    metadata: dict[str, object] = {"data_plane": True}
    session_id = duplex_state.get("session_id")
    if isinstance(session_id, str) and session_id:
        metadata["session_id"] = session_id
    incarnation = duplex_state.get("incarnation")
    if isinstance(incarnation, int):
        metadata["incarnation"] = incarnation
    epoch = duplex_state.get("epoch")
    if isinstance(epoch, int):
        metadata["epoch"] = epoch
    turn_id = duplex_state.get("model_turn_id", duplex_state.get("turn_id"))
    if isinstance(turn_id, int):
        metadata["turn_id"] = turn_id
    session_config = duplex_state.get("session_config")
    if isinstance(session_config, dict):
        metadata["session_config"] = dict(session_config)
    runtime_config = duplex_state.get("runtime_config")
    if isinstance(runtime_config, dict):
        metadata["runtime_config"] = dict(runtime_config)
    return metadata


def _accumulate_native_tts_handoff(streaming_context, new_ids, new_hidden):
    """Hand the talker the FULL accumulated condition on every handoff.

    The runner's streaming buffer update is not merge-safe for a resumable
    stage-1 request: in-place updates merge sub-keys, but a resume prefill
    REPLACES the buffer, silently dropping every earlier segment's tts
    tokens/hiddens (observed losing alternating reply segments — the talker
    vocalized text it never saw between islands it did). Accumulate here so
    the latest handoff always carries the complete history and downstream
    replace semantics are lossless; the talker consumes by cursor.
    """
    bridge_states = getattr(streaming_context, "bridge_states", None)
    if not isinstance(bridge_states, dict):
        return new_ids, new_hidden
    state = bridge_states.setdefault("minicpmo45_tts_handoff", {})
    acc_ids = state.setdefault("acc_tts_ids", [])
    acc_hidden = state.setdefault("acc_tts_hidden", [])
    if new_ids:
        acc_ids.extend(int(t) for t in new_ids)
        if new_hidden:
            acc_hidden.extend(new_hidden)
    if not acc_ids:
        return None, None
    return list(acc_ids), list(acc_hidden)


def _build_tts_scheduler_prompt_token_ids(
    tts_token_ids: torch.Tensor | None,
    llm_output_ids: list[int],
    prompt_token_ids: list[int],
) -> list[int]:
    if tts_token_ids is not None:
        ids = _coerce_token_id_list(tts_token_ids)
        if ids:
            return ids
    if llm_output_ids:
        return llm_output_ids
    if prompt_token_ids:
        return prompt_token_ids[-1:]
    raise ValueError("MiniCPM-o TTS stage requires at least one scheduler prompt token")


def llm2tts(
    source_outputs,
    prompt: OmniTokensPrompt | TextPrompt = None,
    requires_multimodal_data: bool = False,
    _streaming_context=None,
):
    """Convert thinker stage output to talker stage input for MiniCPMO Omni.

    Extracts from thinker output:
      - Full hidden states (prompt + generated) for speaker embedding extraction
      - Prompt token IDs (for finding spk_bos/spk_eos positions)
      - Generated token IDs (for decoding TTS text)

    The talker model will:
      1. Find <|spk_bos|>/<|spk_eos|> positions in prompt_token_ids
      2. Extract speaker embedding from hidden states at those positions
      3. Decode generated text and extract TTS content
      4. Run ConditionalChatTTS pipeline
    """
    if not source_outputs:
        raise ValueError("source_outputs cannot be empty")

    llm_outputs = source_outputs
    tts_inputs = []

    if not isinstance(prompt, list):
        prompt = [prompt]

    multi_modal_data = {}
    for llm_output, p in zip(llm_outputs, prompt):
        if isinstance(p, dict):
            multi_modal_data[llm_output.request_id] = p.get("multi_modal_data", None)
        else:
            multi_modal_data[llm_output.request_id] = getattr(p, "multi_modal_data", None)

    for llm_output in llm_outputs:
        output = llm_output.outputs[0]
        mm_output = output.multimodal_output if isinstance(output.multimodal_output, Mapping) else {}
        special_token_ids = _special_token_ids_from_mm_output(mm_output)
        prompt_token_ids = (
            _coerce_token_id_list(mm_output.get("duplex_prompt_token_ids"))
            or _coerce_token_id_list(mm_output.get("ids.prompt"))
            or list(llm_output.prompt_token_ids)
        )
        llm_output_ids = getattr(output, "token_ids", None)
        if llm_output_ids is None:
            llm_output_ids = getattr(output, "cumulative_token_ids", [])
        # Always copy: CompletionOutput.token_ids can alias the upstream
        # detokenizer's live token list. Forwarding that exact object as the
        # talker prompt makes the stage-1 streaming update extend the list
        # with itself (state and update bind the same object), doubling the
        # thinker's recorded output every segment until the TTS engine input
        # buffer overflows.
        llm_output_ids = list(llm_output_ids)
        thinker_text = getattr(output, "text", "") or ""
        if _has_native_duplex_prompt_metadata(mm_output):
            # The thinker's resumable duplex request reports cumulative
            # output ids/text, but earlier segments are already folded into
            # the prompt by the scheduler session update, and the forwarded
            # hidden states only cover the current prompt + current segment.
            # Hand stage 1 exactly one segment per handoff so token/hidden
            # alignment holds, the talker prompt grows linearly with new
            # tokens instead of quadratically, and downstream transcripts
            # carry per-unit deltas instead of re-sending the whole reply.
            llm_output_ids, thinker_text = _native_duplex_segment_output_ids(
                llm_output_ids,
                thinker_text,
                _streaming_context,
                request_id=str(llm_output.request_id),
            )
        prompt_token_ids_len = len(prompt_token_ids)

        latent = mm_output.get("latent", None)
        if latent is None:
            latent = output.hidden_states if hasattr(output, "hidden_states") else None
            if latent is None:
                raise ValueError("No latent or hidden_states found in thinker output")

        thinker_hidden_states = latent.detach()
        if thinker_hidden_states.ndim == 3 and thinker_hidden_states.shape[0] == 1:
            thinker_hidden_states = thinker_hidden_states.squeeze(0)

        # Build full token sequence and extract TTS region
        full_token_ids = prompt_token_ids + llm_output_ids

        tts_bos_id = special_token_ids.get("tts_bos_token_id")
        is_native_duplex_handoff = _has_native_duplex_prompt_metadata(mm_output)
        if is_native_duplex_handoff:
            _require_native_tts_boundary_metadata(special_token_ids)
        tts_end_ids = _native_tts_boundary_token_ids(special_token_ids)

        # Plain-chat (use_tts_template) fallback: non-duplex requests do not
        # surface special_token_ids, so use MiniCPM-o 4.5's fixed boundaries.
        if tts_bos_id is None and not is_native_duplex_handoff:
            tts_bos_id = 151703
            tts_end_ids = set(tts_end_ids) | {151704, 151645}

        tts_bos_idx = None
        # For native duplex the resumable prompt folds every earlier unit, so
        # a <|tts_bos|> from an already-spoken reply can sit mid-prompt; only
        # a boundary folded as the FINAL prompt token (this unit's decision)
        # or one inside the current segment may start the slice, or stale
        # text would be re-handed to the talker on text-less continuations.
        search_start = max(0, prompt_token_ids_len - 1) if is_native_duplex_handoff else 0
        for idx_t in range(search_start, len(full_token_ids)):
            if full_token_ids[idx_t] == tts_bos_id:
                tts_bos_idx = idx_t + 1

        tts_eos_idx = None
        if tts_bos_idx is not None:
            for idx_t in range(tts_bos_idx, len(full_token_ids)):
                if full_token_ids[idx_t] in tts_end_ids:
                    tts_eos_idx = idx_t
                    break

        tts_token_ids_slice = tts_hidden_slice = None
        native_segment_end = False
        if tts_bos_idx is not None and thinker_hidden_states.shape[0] > tts_bos_idx:
            end_idx = tts_eos_idx if tts_eos_idx is not None else thinker_hidden_states.shape[0]
            if is_native_duplex_handoff and tts_eos_idx is not None:
                boundary_token = full_token_ids[tts_eos_idx]
                native_segment_end = boundary_token in {
                    special_token_ids.get("chunk_eos_token_id"),
                    special_token_ids.get("chunk_tts_eos_token_id"),
                }
            tts_token_ids_slice = torch.tensor(full_token_ids[tts_bos_idx:end_idx], dtype=torch.long)
            tts_hidden_slice = thinker_hidden_states[tts_bos_idx:end_idx].to(torch.float32).contiguous()
        elif is_native_duplex_handoff:
            # Official MiniCPM-o duplex does not prefill an assistant
            # <|tts_bos|> boundary before generation. A segment delta can
            # start with SEVERAL unit decisions (forced/model listens from
            # chunks that produced no handoff accumulate ahead of the speak),
            # so skip the leading listen run, then the speak decision itself;
            # hidden states for TTS start after it and stop at the next
            # chunk terminator.
            listen_id = special_token_ids.get("listen_token_id")
            speak_id = special_token_ids.get("speak_token_id")
            out_ids = llm_output_ids
            j = 0
            while j < len(out_ids) and out_ids[j] == listen_id:
                j += 1
            if j < len(out_ids) and out_ids[j] == speak_id:
                out_start = j + 1
                out_end = len(out_ids)
                turn_eos_id = special_token_ids.get("turn_eos_token_id")
                for idx_t in range(out_start, len(out_ids)):
                    token_id = out_ids[idx_t]
                    if turn_eos_id is not None and token_id == turn_eos_id:
                        # Native duplex trains the talker on the <|turn_eos|>
                        # embedding itself, but any text sampled after it
                        # belongs to a stale tail and must not enter TTS.
                        out_end = idx_t + 1
                        break
                    if token_id in tts_end_ids:
                        out_end = idx_t
                        native_segment_end = token_id in {
                            special_token_ids.get("chunk_eos_token_id"),
                            special_token_ids.get("chunk_tts_eos_token_id"),
                        }
                        break
                # Map output indices onto hidden rows by END alignment: the
                # leading decision tokens of the delta may ALSO be folded into
                # the resumable prompt (they belong to earlier non-forwarded
                # segments), so prompt_len + delta over-counts them and
                # front-aligned indexing truncates the slice. The hidden
                # tensor's last len(out_ids) rows are the delta's rows.
                hidden_base = int(thinker_hidden_states.shape[0]) - len(out_ids)
                if hidden_base >= 0 and out_end > out_start:
                    tts_token_ids_slice = torch.tensor(out_ids[out_start:out_end], dtype=torch.long)
                    tts_hidden_slice = (
                        thinker_hidden_states[hidden_base + out_start : hidden_base + out_end]
                        .to(torch.float32)
                        .contiguous()
                    )
            elif j < len(out_ids) and out_ids[j] not in tts_end_ids:
                # HF streaming_generate does not require an explicit <|speak|>
                # marker. If a unit starts directly with text, the first token
                # is fed back into the LLM but is not included in
                # total_hidden_in_unit; TTS starts from the following token.
                out_start = j + 1
                out_end = len(out_ids)
                turn_eos_id = special_token_ids.get("turn_eos_token_id")
                for idx_t in range(out_start, len(out_ids)):
                    token_id = out_ids[idx_t]
                    if turn_eos_id is not None and token_id == turn_eos_id:
                        out_end = idx_t + 1
                        break
                    if token_id in tts_end_ids:
                        out_end = idx_t
                        native_segment_end = token_id in {
                            special_token_ids.get("chunk_eos_token_id"),
                            special_token_ids.get("chunk_tts_eos_token_id"),
                        }
                        break
                hidden_base = int(thinker_hidden_states.shape[0]) - len(out_ids)
                if hidden_base >= 0 and out_end > out_start:
                    tts_token_ids_slice = torch.tensor(out_ids[out_start:out_end], dtype=torch.long)
                    tts_hidden_slice = (
                        thinker_hidden_states[hidden_base + out_start : hidden_base + out_end]
                        .to(torch.float32)
                        .contiguous()
                    )
        model_intermediate_buffer = build_duplex_intermediate_buffer(
            request_id=str(llm_output.request_id),
            prompt_token_ids=prompt_token_ids,
            output_token_ids=llm_output_ids,
            output_text=thinker_text,
            stream_output=is_native_duplex_handoff,
            native_duplex=is_native_duplex_handoff,
        )
        if is_native_duplex_handoff:
            turn_eos_id = special_token_ids.get("turn_eos_token_id")
            meta = model_intermediate_buffer.setdefault("meta", {})
            data_plane_metadata = _native_duplex_data_plane_metadata(_streaming_context)
            if data_plane_metadata is not None:
                model_intermediate_buffer["duplex"] = data_plane_metadata
            meta["native_duplex_segment_text"] = thinker_text
            meta.setdefault("override_keys", []).extend(
                [
                    "llm_output_text",
                    ["meta", "native_duplex_segment_text"],
                ]
            )
            if turn_eos_id is not None:
                # The talker detects turn end from <|turn_eos|> inside the
                # handed condition (official conditions on its embedding).
                meta["turn_eos_token_id"] = int(turn_eos_id)
            if native_segment_end:
                meta["segment_end"] = True
        req_mm_data = multi_modal_data.get(llm_output.request_id)
        ref_audio = _extract_first_audio_ref(req_mm_data)
        if ref_audio is None:
            ref_audio = _extract_native_runtime_ref_audio(
                model_intermediate_buffer.get("duplex"),
            )
        if ref_audio is not None:
            ref_waveform, ref_sr = ref_audio
            set_ref_audio(model_intermediate_buffer, _to_transport_list(ref_waveform), ref_sr)
        handoff_ids = _coerce_token_id_list(tts_token_ids_slice) if tts_token_ids_slice is not None else None
        handoff_hidden = _to_transport_list(tts_hidden_slice) if tts_hidden_slice is not None else None
        native_turn_end_handoff = False
        if is_native_duplex_handoff:
            turn_eos_id = special_token_ids.get("turn_eos_token_id")
            native_turn_end_handoff = turn_eos_id is not None and handoff_ids is not None and turn_eos_id in handoff_ids
            handoff_ids, handoff_hidden = _accumulate_native_tts_handoff(
                _streaming_context,
                handoff_ids,
                handoff_hidden,
            )
            if not handoff_ids:
                continue
        set_tts_handoff(model_intermediate_buffer, handoff_ids, handoff_hidden)
        if native_turn_end_handoff:
            _reset_native_tts_handoff(_streaming_context)

        scheduler_prompt_token_ids = _build_tts_scheduler_prompt_token_ids(
            tts_token_ids_slice,
            llm_output_ids,
            prompt_token_ids,
        )
        tts_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=scheduler_prompt_token_ids,
                model_intermediate_buffer=model_intermediate_buffer,
                multi_modal_data=(
                    multi_modal_data[llm_output.request_id]
                    if requires_multimodal_data and multi_modal_data.get(llm_output.request_id) is not None
                    else None
                ),
                mm_processor_kwargs=None,
            )
        )
        if native_turn_end_handoff:
            bridge_states = getattr(_streaming_context, "bridge_states", None)
            duplex_state = bridge_states.get("duplex") if isinstance(bridge_states, dict) else None
            if isinstance(duplex_state, dict):
                current_model_turn_id = duplex_state.get("model_turn_id", duplex_state.get("turn_id", 0))
                if isinstance(current_model_turn_id, int):
                    duplex_state["model_turn_id"] = current_model_turn_id + 1

    return tts_inputs
