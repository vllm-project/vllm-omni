from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from vllm.logger import init_logger

from vllm_omni.engine import OmniEngineCoreRequest
from vllm_omni.inputs.data import OmniTokensPrompt

# Special token ids from the MiniCPM tokenizer config.
TTS_BOS_ID = 151703
TTS_EOS_ID = 151704
ASYNC_TTS_CHUNK_SIZE = 10
DEBUG_REF_AUDIO_ENV = "VLLM_OMNI_MINICPMO45_DEBUG_REF_AUDIO"
E2E_ARTIFACT_DIR_ENV = "MINICPMO45_E2E_OUTPUT_DIR"

logger = init_logger(__name__)
_ASYNC_TTS_STATE_KEY = "_minicpmo4_5_async_tts_state"
_ASYNC_CODEC_STATE_KEY = "_minicpmo4_5_async_codec_state"


def _debug_ref_audio_enabled() -> bool:
    return os.environ.get(DEBUG_REF_AUDIO_ENV, "").strip() == "1"


def _summarize_ref_audio(ref_audio: dict[str, Any] | None) -> dict[str, Any] | None:
    if ref_audio is None:
        return None

    wav = np.asarray(ref_audio["wav"], dtype=np.float32).reshape(-1)
    sr = int(ref_audio["sr"])
    return {
        "sample_rate": sr,
        "num_samples": int(wav.shape[0]),
        "duration_sec": float(wav.shape[0] / max(sr, 1)),
    }


def _log_ref_audio_debug(hook: str, payload: dict[str, Any]) -> None:
    if not _debug_ref_audio_enabled():
        return
    logger.info(
        "MiniCPM ref-audio-debug %s",
        json.dumps({"hook": hook, **payload}, ensure_ascii=False, sort_keys=True),
    )


def _ensure_list(x: Any) -> list[Any]:
    if hasattr(x, "_x"):
        return list(x._x)
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return list(x)


def _async_debug_request_dir(request_id: str | None) -> Path | None:
    artifact_root = os.environ.get(E2E_ARTIFACT_DIR_ENV, "").strip()
    if not artifact_root:
        return None

    safe_request_id = re.sub(r"[^A-Za-z0-9._-]+", "_", request_id or "unknown_request").strip("_")
    if not safe_request_id:
        safe_request_id = "unknown_request"

    request_dir = Path(artifact_root) / "debug" / "minicpmo4_5_async_chunk" / safe_request_id
    request_dir.mkdir(parents=True, exist_ok=True)
    return request_dir


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        f.write("\n")


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def _write_int_tokens(path: Path, tokens: list[int]) -> None:
    path.write_text(" ".join(str(int(tok)) for tok in tokens), encoding="utf-8")


def _dump_async_chunk_tokens(
    *,
    request_id: str,
    file_prefix: str,
    chunk_index: int,
    tokens: list[int],
    finished: bool,
    flat_tokens: list[int],
    extra: dict[str, Any] | None = None,
) -> None:
    request_dir = _async_debug_request_dir(request_id)
    if request_dir is None:
        return

    payload: dict[str, Any] = {
        "chunk_index": int(chunk_index),
        "finished": bool(finished),
        "token_count": int(len(tokens)),
        "tokens": [int(tok) for tok in tokens],
    }
    if extra:
        payload.update(extra)

    _append_jsonl(request_dir / f"{file_prefix}_chunks.jsonl", payload)
    _write_json(request_dir / f"{file_prefix}_token_ids.json", [int(tok) for tok in flat_tokens])
    _write_int_tokens(request_dir / f"{file_prefix}_token_ids.txt", [int(tok) for tok in flat_tokens])


def _validate_stage_inputs(stage_list: list[Any], engine_input_source: list[int]) -> Any:
    if not engine_input_source:
        raise ValueError("engine_input_source cannot be empty")

    stage_id = engine_input_source[0]
    if stage_id >= len(stage_list):
        raise IndexError(f"Invalid stage_id: {stage_id}")

    stage = stage_list[stage_id]
    if stage.engine_outputs is None:
        raise RuntimeError(f"Stage {stage_id} has no outputs yet")

    return stage.engine_outputs


def _extract_token_hidden_states(latent: Any) -> torch.Tensor:
    """Normalize thinker latent captures into [seq_len, hidden_size]."""
    if isinstance(latent, torch.Tensor):
        if latent.ndim == 2:
            return latent
        if latent.ndim == 3:
            # Handle either [seq, layers, hidden] or [layers, seq, hidden].
            return latent[:, -1, :] if latent.shape[0] >= latent.shape[1] else latent[-1]
        if latent.ndim == 4:
            # Handle [seq, layers, batch, hidden] or [layers, batch, seq, hidden].
            if latent.shape[0] >= latent.shape[1]:
                return latent[:, -1, 0, :]
            return latent[-1, 0, :, :]
        raise ValueError(f"Unsupported latent tensor shape: {tuple(latent.shape)}")

    if isinstance(latent, (list, tuple)):
        rows: list[torch.Tensor] = []
        for token_layers in latent:
            layer_value = token_layers[-1] if isinstance(token_layers, (list, tuple)) else token_layers
            if not isinstance(layer_value, torch.Tensor):
                raise TypeError(f"Unsupported latent element type: {type(layer_value)}")
            if layer_value.ndim == 1:
                rows.append(layer_value)
            elif layer_value.ndim == 2:
                rows.append(layer_value[0])
            elif layer_value.ndim == 3:
                rows.append(layer_value[-1, 0])
            else:
                raise ValueError(f"Unsupported latent element shape: {tuple(layer_value.shape)}")
        return torch.stack(rows, dim=0)

    raise TypeError(f"Unsupported latent type: {type(latent)}")


def _get_request_all_token_ids(request: Any) -> list[int]:
    all_token_ids = getattr(request, "all_token_ids", None)
    if all_token_ids is not None:
        return [int(tok) for tok in _ensure_list(all_token_ids)]

    prompt_token_ids = _ensure_list(getattr(request, "prompt_token_ids", None))
    output_token_ids = _ensure_list(getattr(request, "output_token_ids", None))
    return [int(tok) for tok in prompt_token_ids + output_token_ids]


def _get_async_tts_state(transfer_manager: Any, request_id: str) -> dict[str, Any]:
    state = transfer_manager.request_payload.get(request_id)
    if not isinstance(state, dict) or state.get("_kind") != _ASYNC_TTS_STATE_KEY:
        state = {
            "_kind": _ASYNC_TTS_STATE_KEY,
            "tts_started": False,
            "tts_closed": False,
            "hidden_size": None,
            "pending_llm_tokens": [],
            "pending_hidden_states": None,
            "debug_chunk_index": 0,
            "debug_emitted_llm_tokens": [],
            "emitted_chunk_serial": 0,
            "prompt_state_initialized": False,
            "seen_output_token_count": 0,
        }
        transfer_manager.request_payload[request_id] = state
    return state


def _get_async_codec_state(transfer_manager: Any, request_id: str) -> dict[str, Any]:
    state = transfer_manager.request_payload.get(request_id)
    if not isinstance(state, dict) or state.get("_kind") != _ASYNC_CODEC_STATE_KEY:
        state = {
            "_kind": _ASYNC_CODEC_STATE_KEY,
            "emitted_output_tokens": 0,
            "pending_audio_tokens": [],
            "debug_chunk_index": 0,
            "debug_emitted_audio_tokens": [],
        }
        transfer_manager.request_payload[request_id] = state
    return state


def _empty_tts_hidden_states(hidden_size: int | None) -> torch.Tensor:
    width = 0 if hidden_size is None else int(hidden_size)
    return torch.empty((0, width), dtype=torch.float32)


def _append_async_tts_pending(
    state: dict[str, Any],
    llm_tokens: torch.Tensor,
    tts_hidden_states: torch.Tensor,
) -> None:
    if llm_tokens.numel() == 0:
        return

    pending_llm_tokens = state.setdefault("pending_llm_tokens", [])
    pending_llm_tokens.extend(torch.as_tensor(llm_tokens, dtype=torch.long).tolist())

    hidden_chunk = torch.as_tensor(tts_hidden_states, dtype=torch.float32).detach().cpu().contiguous()
    pending_hidden = state.get("pending_hidden_states")
    if isinstance(pending_hidden, torch.Tensor) and pending_hidden.numel() > 0:
        state["pending_hidden_states"] = torch.cat([pending_hidden, hidden_chunk], dim=0).contiguous()
    else:
        state["pending_hidden_states"] = hidden_chunk
    state["hidden_size"] = int(hidden_chunk.shape[-1])


def _pop_async_tts_pending(
    state: dict[str, Any],
    count: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    pending_llm_tokens = state.get("pending_llm_tokens", [])
    total = len(pending_llm_tokens)
    if total == 0:
        return (
            torch.empty((0,), dtype=torch.long),
            _empty_tts_hidden_states(state.get("hidden_size")),
        )

    take = total if count is None else min(int(count), total)
    llm_tokens = torch.tensor(pending_llm_tokens[:take], dtype=torch.long)

    pending_hidden = state.get("pending_hidden_states")
    if isinstance(pending_hidden, torch.Tensor) and pending_hidden.numel() > 0:
        tts_hidden_states = pending_hidden[:take].to(torch.float32).contiguous()
        remaining_hidden = pending_hidden[take:]
        state["pending_hidden_states"] = remaining_hidden.contiguous() if remaining_hidden.numel() > 0 else None
    else:
        tts_hidden_states = _empty_tts_hidden_states(state.get("hidden_size"))
        state["pending_hidden_states"] = None

    state["pending_llm_tokens"] = pending_llm_tokens[take:]
    return llm_tokens, tts_hidden_states


def _extract_async_tts_delta(
    hidden_states: torch.Tensor,
    chunk_token_ids: list[int],
    state: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    if hidden_states.ndim != 2:
        raise ValueError(f"Expected [seq_len, hidden_size] hidden states, got {tuple(hidden_states.shape)}")

    seq_len = int(hidden_states.shape[0])
    token_len = len(chunk_token_ids)
    if seq_len != token_len:
        usable = min(seq_len, token_len)
        logger.warning(
            "MiniCPM async thinker chunk token/latent mismatch; trimming to tail. tokens=%d latent=%d usable=%d",
            token_len,
            seq_len,
            usable,
        )
        hidden_states = hidden_states[-usable:]
        chunk_token_ids = chunk_token_ids[-usable:]

    emitted_tokens: list[int] = []
    emitted_hidden_states: list[torch.Tensor] = []
    saw_tts_eos = False

    for token_id, hidden_row in zip(chunk_token_ids, hidden_states, strict=False):
        token_id = int(token_id)
        if state["tts_closed"]:
            break
        if not state["tts_started"]:
            if token_id == TTS_BOS_ID:
                state["tts_started"] = True
            continue
        if token_id == TTS_EOS_ID:
            state["tts_closed"] = True
            saw_tts_eos = True
            break
        emitted_tokens.append(token_id)
        emitted_hidden_states.append(hidden_row)

    if not emitted_tokens:
        return (
            torch.empty((0,), dtype=torch.long),
            _empty_tts_hidden_states(state.get("hidden_size")),
            saw_tts_eos,
        )

    token_tensor = torch.tensor(emitted_tokens, dtype=torch.long)
    hidden_tensor = torch.stack(emitted_hidden_states, dim=0).to(torch.float32)
    state["hidden_size"] = int(hidden_tensor.shape[-1])
    return token_tensor, hidden_tensor, saw_tts_eos


def _initialize_async_tts_state_from_prompt(state: dict[str, Any], request: Any) -> None:
    if state.get("prompt_state_initialized", False):
        return

    prompt_token_ids = [int(tok) for tok in _ensure_list(getattr(request, "prompt_token_ids", None))]
    state["prompt_state_initialized"] = True
    if not prompt_token_ids:
        return

    if TTS_BOS_ID in prompt_token_ids:
        state["tts_started"] = True

    # Defensive: if a malformed prompt already contains a closed TTS span,
    # avoid collecting generated tokens for this request.
    if state["tts_started"] and TTS_EOS_ID in prompt_token_ids:
        bos_idx = prompt_token_ids.index(TTS_BOS_ID)
        eos_idx = prompt_token_ids.index(TTS_EOS_ID)
        if eos_idx > bos_idx:
            state["tts_closed"] = True


def thinker2talker(
    stage_list: list[Any],
    engine_input_source: list[int],
    prompt: Any = None,
    requires_multimodal_data: bool = False,
) -> list[OmniTokensPrompt]:
    from vllm_omni.model_executor.models.minicpmo4_5.minicpmo4_5_talker import (
        MiniCPMO4_5TalkerForConditionalGeneration,
    )

    thinker_outputs = _validate_stage_inputs(stage_list, engine_input_source)
    talker_inputs: list[OmniTokensPrompt] = []

    for thinker_output in thinker_outputs:
        output = thinker_output.outputs[0]

        prompt_ids = list(thinker_output.prompt_token_ids)
        gen_ids = list(output.token_ids)
        full_sequence = prompt_ids + gen_ids

        hidden_states = _extract_token_hidden_states(output.multimodal_output["latent"]).detach()

        tts_bos_idx = None
        tts_eos_idx = None
        for i, tok in enumerate(full_sequence):
            if tok == TTS_BOS_ID:
                tts_bos_idx = i + 1
                tts_eos_idx = None
            elif tok == TTS_EOS_ID and tts_bos_idx is not None:
                tts_eos_idx = i
                break

        if tts_bos_idx is None:
            raise ValueError("MiniCPM thinker output is missing <|tts_bos|>.")
        if tts_eos_idx is None:
            tts_eos_idx = len(full_sequence)
        usable_end = min(tts_eos_idx, int(hidden_states.shape[0]))
        if usable_end <= tts_bos_idx:
            raise ValueError(
                "MiniCPM thinker latent span does not cover the TTS content tokens: "
                f"latent={tuple(hidden_states.shape)} tts_range=({tts_bos_idx}, {tts_eos_idx})"
            )

        llm_tokens = torch.tensor(full_sequence[tts_bos_idx:usable_end], dtype=torch.long)
        tts_hidden_states = hidden_states[tts_bos_idx:usable_end].to(torch.float32)

        info = {
            "llm_tokens": llm_tokens,
            "tts_hidden_states": tts_hidden_states,
        }
        prompt_len = MiniCPMO4_5TalkerForConditionalGeneration.estimate_prompt_len_from_additional_information(info)

        talker_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=[0] * prompt_len,
                additional_information=info,
                multi_modal_data=None,
                mm_processor_kwargs=None,
            )
        )

    return talker_inputs


def _extract_ref_audio_from_prompt(prompt: Any, index: int = 0) -> Any:
    if prompt is None:
        return None
    p = prompt[index] if isinstance(prompt, list) and index < len(prompt) else prompt
    if p is None or not isinstance(p, dict):
        return None
    add_info = p.get("additional_information")
    if not isinstance(add_info, dict):
        return None
    raw_ref_audio = add_info.get("ref_audio")
    if raw_ref_audio is None:
        return None
    return _canonicalize_ref_audio(raw_ref_audio)


def _canonicalize_ref_audio(raw_ref_audio: Any) -> dict[str, Any]:
    if isinstance(raw_ref_audio, list) and len(raw_ref_audio) == 1:
        raw_ref_audio = raw_ref_audio[0]

    if isinstance(raw_ref_audio, dict):
        wav = raw_ref_audio.get("wav")
        sr = raw_ref_audio.get("sr")
    elif isinstance(raw_ref_audio, (list, tuple)) and len(raw_ref_audio) == 2:
        wav, sr = raw_ref_audio
    else:
        raise TypeError(f"Unsupported MiniCPM ref_audio payload at stage boundary: {type(raw_ref_audio)}")

    if isinstance(sr, torch.Tensor):
        if sr.numel() != 1:
            raise ValueError("MiniCPM ref_audio sample rate tensor must be scalar.")
        sr = int(sr.item())
    elif not isinstance(sr, int):
        sr = int(sr)

    if isinstance(wav, torch.Tensor):
        wav_np = wav.detach().cpu().float().numpy()
    else:
        wav_np = np.asarray(wav, dtype=np.float32)

    if wav_np.ndim == 0:
        raise ValueError("MiniCPM ref_audio waveform must be at least 1-D.")
    if wav_np.ndim > 1:
        wav_np = wav_np.mean(axis=-1)

    return {
        "wav": np.asarray(wav_np, dtype=np.float32).reshape(-1).tolist(),
        "sr": int(sr),
    }


def talker2code2wav(
    stage_list: list[Any],
    engine_input_source: list[int],
    prompt: Any = None,
    requires_multimodal_data: bool = False,
) -> list[OmniTokensPrompt]:
    """Minimal non-async handoff for the future MiniCPM code2wav stage.

    MiniCPM talker emits audio codec token ids directly, so the full finished
    token sequence is enough to seed the next stage once that decoder exists.
    """
    talker_outputs = _validate_stage_inputs(stage_list, engine_input_source)
    code2wav_inputs: list[OmniTokensPrompt] = []

    for i, talker_output in enumerate(talker_outputs):
        if not talker_output.finished:
            continue
        output = talker_output.outputs[0]
        token_ids = list(output.token_ids)
        if not token_ids:
            continue
        additional_information: dict[str, Any] | None = None
        ref_audio = _extract_ref_audio_from_prompt(prompt, index=i)
        if ref_audio is not None:
            additional_information = {"ref_audio": ref_audio}
        _log_ref_audio_debug(
            "talker2code2wav",
            {
                "request_index": i,
                "request_id": getattr(talker_output, "request_id", None),
                "token_count": len(token_ids),
                "has_ref_audio": ref_audio is not None,
                "ref_audio": _summarize_ref_audio(ref_audio),
            },
        )
        code2wav_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=token_ids,
                additional_information=additional_information,
                multi_modal_data=None,
                mm_processor_kwargs=None,
            )
        )

    return code2wav_inputs


def thinker2talker_async_chunk(
    transfer_manager: Any,
    pooling_output: dict[str, Any],
    request: OmniEngineCoreRequest,
    is_finished: bool = False,
) -> dict[str, Any] | None:
    """Stream only the MiniCPM TTS span from thinker to talker.

    This async path treats ``pooling_output["latent"]`` as the current chunk's
    token-aligned hidden states. It tracks whether ``<|tts_bos|>`` has been
    seen, emits only the spoken tokens between ``<|tts_bos|>`` and
    ``<|tts_eos|>``, and drops everything outside that span.

    Assumption: thinker sampling stops on the same step that emits
    ``<|tts_eos|>``. If it can continue generating past that token, the
    adapter lifecycle may need an explicit early-finish path.
    """
    request_id = request.external_req_id
    state = _get_async_tts_state(transfer_manager, request_id)
    _initialize_async_tts_state_from_prompt(state, request)

    hidden_states = None
    saw_tts_eos = False
    latent_payload = None
    if isinstance(pooling_output, dict):
        latent_payload = pooling_output.get("latent")
        if latent_payload is None:
            # Async chunk callbacks receive the raw per-step hidden slice under
            # "hidden" from GPUARModelRunner.pooler_output.
            latent_payload = pooling_output.get("hidden")

    if latent_payload is not None:
        hidden_states = _extract_token_hidden_states(latent_payload).detach().cpu().to(torch.float32)
        if hidden_states.ndim == 2 and hidden_states.shape[-1] > 0:
            state["hidden_size"] = int(hidden_states.shape[-1])

    output_token_ids = [int(tok) for tok in _ensure_list(getattr(request, "output_token_ids", None))]
    seen_output_token_count = int(state.get("seen_output_token_count", 0) or 0)
    new_output_count = max(0, len(output_token_ids) - seen_output_token_count)

    if hidden_states is not None and hidden_states.numel() > 0 and new_output_count > 0:
        current_chunk_len = int(hidden_states.shape[0])
        usable = min(current_chunk_len, new_output_count)
        if usable != current_chunk_len or usable != new_output_count:
            logger.warning(
                "MiniCPM async thinker output/latent mismatch; trimming to tail. "
                "request_id=%s new_output_count=%d latent=%d usable=%d",
                request_id,
                new_output_count,
                current_chunk_len,
                usable,
            )
        chunk_token_ids = output_token_ids[-usable:]
        llm_tokens, tts_hidden_states, saw_tts_eos = _extract_async_tts_delta(
            hidden_states[-usable:],
            chunk_token_ids,
            state,
        )
        _append_async_tts_pending(state, llm_tokens, tts_hidden_states)

    state["seen_output_token_count"] = len(output_token_ids)

    stream_finished = bool(is_finished or saw_tts_eos)
    pending_count = len(state.get("pending_llm_tokens", []))
    if stream_finished:
        llm_tokens, tts_hidden_states = _pop_async_tts_pending(state, count=None)
        emitted_tokens = [int(tok) for tok in llm_tokens.tolist()]
        chunk_serial = int(state.get("emitted_chunk_serial", 0) or 0)
        if state.get("tts_started", False) and not state.get("tts_closed", False) and not emitted_tokens:
            logger.warning(
                "MiniCPM async thinker finished without emitted TTS tokens. "
                "request_id=%s prompt_has_tts_bos=%s pending_count=%d current_chunk_tokens=%d",
                request_id,
                TTS_BOS_ID in [int(tok) for tok in _ensure_list(getattr(request, "prompt_token_ids", None))],
                pending_count,
                0 if hidden_states is None else int(hidden_states.shape[0]),
            )
        state["debug_emitted_llm_tokens"].extend(emitted_tokens)
        _dump_async_chunk_tokens(
            request_id=request_id,
            file_prefix="thinker_tts",
            chunk_index=int(state["debug_chunk_index"]),
            tokens=emitted_tokens,
            finished=True,
            flat_tokens=list(state["debug_emitted_llm_tokens"]),
            extra={
                "pending_count_before_flush": int(pending_count),
                "saw_tts_eos": bool(saw_tts_eos),
                "is_finished_flag": bool(is_finished),
                "tts_started": bool(state.get("tts_started", False)),
                "tts_closed": bool(state.get("tts_closed", False)),
            },
        )
        state["debug_chunk_index"] = int(state["debug_chunk_index"]) + 1
        state["emitted_chunk_serial"] = chunk_serial + 1
        return {
            "llm_tokens": llm_tokens,
            "tts_hidden_states": tts_hidden_states,
            "async_tts_chunk_id": chunk_serial,
            "global_request_id": request_id,
            "finished": torch.tensor(True, dtype=torch.bool),
        }

    if pending_count < ASYNC_TTS_CHUNK_SIZE:
        return None

    llm_tokens, tts_hidden_states = _pop_async_tts_pending(state, count=ASYNC_TTS_CHUNK_SIZE)
    emitted_tokens = [int(tok) for tok in llm_tokens.tolist()]
    chunk_serial = int(state.get("emitted_chunk_serial", 0) or 0)
    state["debug_emitted_llm_tokens"].extend(emitted_tokens)
    _dump_async_chunk_tokens(
        request_id=request_id,
        file_prefix="thinker_tts",
        chunk_index=int(state["debug_chunk_index"]),
        tokens=emitted_tokens,
        finished=False,
        flat_tokens=list(state["debug_emitted_llm_tokens"]),
        extra={
            "pending_count_before_flush": int(pending_count),
            "chunk_size": int(ASYNC_TTS_CHUNK_SIZE),
            "tts_started": bool(state.get("tts_started", False)),
            "tts_closed": bool(state.get("tts_closed", False)),
        },
    )
    state["debug_chunk_index"] = int(state["debug_chunk_index"]) + 1
    state["emitted_chunk_serial"] = chunk_serial + 1
    return {
        "llm_tokens": llm_tokens,
        "tts_hidden_states": tts_hidden_states,
        "async_tts_chunk_id": chunk_serial,
        "global_request_id": request_id,
        "finished": torch.tensor(False, dtype=torch.bool),
    }


def talker2code2wav_async_chunk(
    transfer_manager: Any,
    pooling_output: dict[str, Any],
    request: OmniEngineCoreRequest,
    is_finished: bool = False,
):
    """Stream MiniCPM talker audio token ids to code2wav in fixed-size chunks."""
    connector = getattr(transfer_manager, "connector", None)
    raw_cfg = getattr(connector, "config", {}) or {}
    cfg = raw_cfg.get("extra", raw_cfg) if isinstance(raw_cfg, dict) else {}
    chunk_size = int(cfg.get("codec_chunk_frames", 25))
    request_id = request.external_req_id
    state = _get_async_codec_state(transfer_manager, request_id)

    output_token_ids = [int(tok) for tok in _ensure_list(getattr(request, "output_token_ids", None))]
    emitted_output_tokens = int(state.get("emitted_output_tokens", 0) or 0)
    if len(output_token_ids) > emitted_output_tokens:
        new_tokens = output_token_ids[emitted_output_tokens:]
        state["pending_audio_tokens"].extend(new_tokens)
        state["emitted_output_tokens"] = len(output_token_ids)

    pending_audio_tokens = state.get("pending_audio_tokens", [])
    if len(pending_audio_tokens) < chunk_size and not is_finished:
        return None

    if is_finished:
        codes = list(pending_audio_tokens)
    else:
        codes = list(pending_audio_tokens[:chunk_size])
        state["pending_audio_tokens"] = pending_audio_tokens[chunk_size:]

    state["debug_emitted_audio_tokens"].extend(int(tok) for tok in codes)
    _dump_async_chunk_tokens(
        request_id=request_id,
        file_prefix="talker_codec",
        chunk_index=int(state["debug_chunk_index"]),
        tokens=[int(tok) for tok in codes],
        finished=bool(is_finished),
        flat_tokens=list(state["debug_emitted_audio_tokens"]),
        extra={
            "chunk_size": int(chunk_size),
            "pending_count_before_flush": int(len(pending_audio_tokens)),
            "left_context_size": 1 if is_finished else 0,
        },
    )
    state["debug_chunk_index"] = int(state["debug_chunk_index"]) + 1

    if is_finished:
        transfer_manager.request_payload.pop(request_id, None)

    return {
        "code_predictor_codes": codes,
        # Only left_context_size is forwarded into runtime_additional_information
        # for Stage 2. Reuse it as a simple EOF marker:
        #   0 -> more chunks expected
        #   1 -> final chunk / flush
        "left_context_size": 1 if is_finished else 0,
        "finished": torch.tensor(is_finished, dtype=torch.bool),
    }
