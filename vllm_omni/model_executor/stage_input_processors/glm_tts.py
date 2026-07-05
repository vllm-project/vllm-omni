# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage input processor for GLM-TTS: AR → DiT Pipeline.

Supports both sync (non-streaming) and async_chunk (streaming) modes.
Adapted for LLM_GENERATION execution type on stage 1 (DiT).
"""

from typing import Any

import torch
from vllm.logger import init_logger

from vllm_omni.data_entry_keys import (
    CodesStruct,
    EmbeddingsStruct,
    MetaStruct,
    OmniPayloadStruct,
)
from vllm_omni.engine.serialization import deserialize_additional_information

logger = init_logger(__name__)

_GLM_TTS_PRIVATE_KEY = "glm_tts"


def _copy_voice_clone_payload(
    src: dict[str, Any],
    dst: dict[str, Any],
    *,
    to_cpu: bool = False,
    skip_existing: bool = False,
) -> None:
    if not skip_existing or "prompt_speech_token" not in dst:
        prompt = src.get("prompt_speech_token")
        if prompt is None:
            prompt = src.get("prompt_token")
        if prompt is not None:
            if to_cpu:
                prompt = _to_cpu_tensor(prompt)
            if prompt is not None:
                dst["prompt_speech_token"] = prompt

    for key in ("prompt_feat", "embedding"):
        if skip_existing and key in dst:
            continue
        val = src.get(key)
        if val is not None:
            if to_cpu:
                val = _to_cpu_tensor(val)
            if val is not None:
                dst[key] = val


def _build_voice_clone_embed_struct(payload: dict[str, Any]) -> EmbeddingsStruct | None:
    prompt_token = payload.get("prompt_speech_token")
    if prompt_token is None:
        prompt_token = payload.get("prompt_token")
    prompt_feat = payload.get("prompt_feat")
    embedding = payload.get("embedding")
    if prompt_token is None and prompt_feat is None and embedding is None:
        return None
    return EmbeddingsStruct(
        speech_token=prompt_token,
        speech_feat=prompt_feat,
        embedding=embedding,
    )


# ---------------------------------------------------------------------------
# Sync processor: collect all speech tokens from AR, pass to DiT
# ---------------------------------------------------------------------------


def ar_to_dit(
    source_outputs: list[Any],
    prompt: Any = None,
    _requires_multimodal_data: bool = False,
) -> list[Any]:
    """Non-streaming: collect speech tokens from AR, pass to DiT.

    Propagates voice cloning data (prompt_token, prompt_feat, embedding)
    from the AR model's multimodal_output to the DiT stage using the same
    nested ``embed`` dict format as the async_chunk path.  This ensures
    ``nested_get(info, "embed", ...)`` in the DiT wrapper reads all three
    fields without relying on backward-compat flat-key fallbacks.
    """
    from vllm_omni.inputs.data import OmniTokensPrompt

    dit_inputs: list[OmniTokensPrompt] = []

    for output in source_outputs:
        mm = output.outputs[0].multimodal_output
        speech_tokens = mm.get("speech_tokens")

        # Extract valid token list (filter -1 placeholder markers from prefill)
        if isinstance(speech_tokens, torch.Tensor):
            valid = speech_tokens.to(torch.long).reshape(-1)
            token_list = valid[valid >= 0].cpu().tolist()
        elif speech_tokens is not None:
            token_list = [t for t in speech_tokens if t >= 0]
        else:
            token_list = []

        if not token_list:
            logger.warning("No valid speech tokens for request %s", output.request_id)

        additional_info: dict[str, Any] = {"speech_tokens": token_list}

        # Build nested embed dict (unified with async_chunk path).
        # _build_voice_clone_embed_struct reads prompt_speech_token/prompt_feat/
        # embedding from *mm* and returns an EmbeddingsStruct; we convert to a
        # plain dict so flatten_payload/unflatten_payload round-trips correctly
        # (key "embed" is in _NESTED_KEYS).
        embed = _build_voice_clone_embed_struct(mm)
        if embed is not None:
            additional_info["embed"] = {
                "speech_token": embed.speech_token,
                "speech_feat": embed.speech_feat,
                "embedding": embed.embedding,
            }

        # Populate meta dict so DiT wrapper uses block-causal attention.
        # Without meta, forward() sets uses_streaming=False → bidirectional
        # attention → garbage audio. The full token sequence is treated as
        # a single "chunk" with stream_finished=True.
        n_tokens = len(token_list) if token_list else 1
        additional_info["meta"] = {
            "req_id": [output.request_id],
            "left_context_size": 0,
            "stream_finished": True,
        }
        additional_info["kv_metadata"] = {
            _GLM_TTS_PRIVATE_KEY: {
                "chunk_sizes_history": [n_tokens],
                "block_pattern": [n_tokens],
            }
        }

        dit_inputs.append(
            OmniTokensPrompt(
                # LLM_GENERATION scheduler needs at least 1 token
                prompt_token_ids=token_list or [0],
                additional_information=additional_info,
            )
        )

    return dit_inputs


# ---------------------------------------------------------------------------
# Helper: extract last speech token from AR model pooling output
# ---------------------------------------------------------------------------


def _extract_last_speech_token(pooling_output: dict[str, Any]) -> int | None:
    """Extract the last valid speech token from AR model output.

    GLM-TTS AR produces one speech token per decode step.
    Returns the token ID (relative to ATS, i.e. 0-based), or None.
    """
    speech_tokens = pooling_output.get("speech_tokens")
    if not isinstance(speech_tokens, torch.Tensor) or speech_tokens.numel() == 0:
        return None
    token_val = int(speech_tokens.reshape(-1).to(torch.long)[-1].item())
    # -1 = invalid/EOA marker
    if token_val < 0:
        return None
    return token_val


def _to_cpu_tensor(value: Any) -> torch.Tensor | None:
    """Convert value to CPU tensor if possible."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, list):
        if not value:
            return None
        if isinstance(value[0], torch.Tensor):
            return value[0].detach().cpu()
    return None


# ---------------------------------------------------------------------------
# Async streaming processor: emit speech token chunks as AR produces them
# ---------------------------------------------------------------------------


def ar_to_dit_async_chunk(
    transfer_manager: Any,
    multimodal_output: dict[str, Any] | None,
    request: Any,
    is_finished: bool = False,
) -> OmniPayloadStruct | None:
    """Async streaming processor: emit speech token chunks as AR produces them.

    Accumulates per-step speech tokens and emits fixed-size cumulative prefixes
    for GLM-TTS flow-cache streaming.

    Follows the CosyVoice3 talker2code2wav_async_chunk transfer pattern:
    - Per-request state tracking (seen tokens, prompt sent flag)
    - First chunk carries voice clone conditioning payload
    - Each chunk includes left_context_size, stream_finished, req_id
    - code_predictor_codes for chunk_transfer_adapter consumption

    Unlike codec left-context decoders, official GLM-TTS sends the cumulative
    AR token prefix to the flow stage on every chunk and reuses diffusion
    latents internally.  Here left_context_size means the stable token prefix
    that has already been emitted, so DiT can crop regenerated audio after
    sampling.

    GLM-TTS produces single-token-per-step (no multi-codebook), so each entry
    in code_prompt_token_ids is a plain int, not a list of codebook values.
    """
    request_id = getattr(request, "external_req_id", None) or getattr(request, "request_id", None)
    if request_id is None:
        raise ValueError("GLM-TTS async chunk request is missing request id")
    finished = bool(is_finished or request.is_finished())
    pooling_output = multimodal_output

    # Read connector chunk config (supports progressive list or single int)
    connector = getattr(transfer_manager, "connector", None)
    raw_cfg = getattr(connector, "config", {}) or {}
    cfg = raw_cfg.get("extra", raw_cfg) if isinstance(raw_cfg, dict) else {}
    chunk_frames_cfg = cfg.get("codec_chunk_frames", 25)
    if isinstance(chunk_frames_cfg, list):
        progressive_chunk_sizes = [int(c) for c in chunk_frames_cfg]
    else:
        progressive_chunk_sizes = [int(chunk_frames_cfg)]
    left_context_size_config = int(cfg.get("codec_left_context_frames", 25))
    crossfade_sec = float(cfg.get("crossfade_sec", 0.1))

    if not progressive_chunk_sizes or any(c <= 0 for c in progressive_chunk_sizes) or left_context_size_config < 0:
        raise ValueError(
            f"Invalid codec chunk config: codec_chunk_frames={chunk_frames_cfg}, "
            f"codec_left_context_frames={left_context_size_config}"
        )

    # Initialize per-request state (like CosyVoice3)
    request_payload = getattr(transfer_manager, "request_payload", None)
    if request_payload is None:
        request_payload = {}
        transfer_manager.request_payload = request_payload
    code_prompt_token_ids = getattr(transfer_manager, "code_prompt_token_ids", None)
    if code_prompt_token_ids is None:
        code_prompt_token_ids = {}
        transfer_manager.code_prompt_token_ids = code_prompt_token_ids
    code_prompt_token_ids.setdefault(request_id, [])
    request_state = request_payload.get(request_id)
    if not isinstance(request_state, dict) or "_glm_tts_async_state" not in request_state:
        # Extract voice clone conditioning from request additional_information
        info = deserialize_additional_information(getattr(request, "additional_information", None))
        prompt_payload: dict[str, Any] = {}
        _copy_voice_clone_payload(info, prompt_payload, to_cpu=True)

        # Also try to extract from pooling_output (first call)
        if isinstance(pooling_output, dict):
            _copy_voice_clone_payload(pooling_output, prompt_payload, to_cpu=True, skip_existing=True)

        request_state = {
            "_glm_tts_async_state": {
                "seen_len": 0,
                "sent_prompt": False,
                "emitted_chunks": 0,
                "emitted_token_len": 0,
                "terminal_sent": False,
                "prompt_payload": prompt_payload,
                "chunk_sizes_history": [],
                "block_pattern": progressive_chunk_sizes,
            }
        }
        request_payload[request_id] = request_state

    state = request_state["_glm_tts_async_state"]
    if state.get("terminal_sent", False):
        return None

    # Helper: get voice clone embed (once per request)
    def _pop_embed() -> EmbeddingsStruct | None:
        if state.get("sent_prompt", False):
            return None
        state["sent_prompt"] = True
        return _build_voice_clone_embed_struct(state.get("prompt_payload", {}))

    # Helper: build terminal or chunk payload
    def _make_payload(
        codes: list[int],
        *,
        tok_offset: int,
        is_terminal: bool,
    ) -> OmniPayloadStruct:
        return OmniPayloadStruct(
            codes=CodesStruct(
                audio=(torch.tensor(codes, dtype=torch.long) if codes else torch.empty(0, dtype=torch.long))
            ),
            meta=MetaStruct(
                finished=torch.tensor(is_terminal, dtype=torch.bool),
                stream_finished=torch.tensor(is_terminal, dtype=torch.bool),
                req_id=[request_id],
                left_context_size=tok_offset,
            ),
            kv_metadata={
                _GLM_TTS_PRIVATE_KEY: {
                    "chunk_sizes_history": list(state.get("chunk_sizes_history", [])),
                    "block_pattern": list(state.get("block_pattern", progressive_chunk_sizes)),
                    "crossfade_sec": crossfade_sec,
                }
            },
            embed=_pop_embed(),
        )

    # Accumulate new speech token from this step
    if isinstance(pooling_output, dict):
        token = _extract_last_speech_token(pooling_output)
        if token is not None:
            code_prompt_token_ids[request_id].append(token)
    elif not finished:
        return None

    token_frames = code_prompt_token_ids[request_id]
    length = len(token_frames)
    emitted_token_len = int(state.get("emitted_token_len", 0))

    # Terminal: no tokens, or finished on/before chunk boundary
    if finished and length <= emitted_token_len:
        state["terminal_sent"] = True
        # Boundary finish is metadata-only: the cumulative prefix has already
        # been emitted, so resending it would make the DiT/vocoder redo work
        # and can duplicate audio.
        return _make_payload([], tok_offset=emitted_token_len, is_terminal=True)

    if length <= 0:
        return None

    # Progressive chunk size: 25 → 50 → 200 (official GLM-TTS pattern)
    chunk_count = int(state.get("emitted_chunks", 0))
    idx = min(chunk_count, len(progressive_chunk_sizes) - 1)
    current_chunk_size = progressive_chunk_sizes[idx]

    # Check if enough tokens accumulated for next chunk
    if not finished and (length - emitted_token_len) < current_chunk_size:
        return None

    # Determine end index for cumulative prefix
    if emitted_token_len == 0:
        end_index = min(length, current_chunk_size)
        token_offset = 0
    else:
        end_index = length if finished else min(length, emitted_token_len + current_chunk_size)
        token_offset = emitted_token_len

    # Track chunk sizes for cache slicing
    chunk_sizes_history: list[int] = list(state.get("chunk_sizes_history", []))
    chunk_sizes_history.append(end_index - emitted_token_len)
    state["chunk_sizes_history"] = chunk_sizes_history
    state["emitted_chunks"] = chunk_count + 1

    if finished:
        state["terminal_sent"] = True
    else:
        state["emitted_token_len"] = max(emitted_token_len, end_index)

    return _make_payload(list(token_frames[:end_index]), tok_offset=token_offset, is_terminal=finished)
